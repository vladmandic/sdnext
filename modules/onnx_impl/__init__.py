from modules.logger import log
from typing import Any, Dict, Optional
import numpy as np
import torch
import diffusers
from installer import installed, install


initialized = False


try:
    import onnxruntime as ort
except Exception as e:
    log.error(f'ONNX import error: {e}')
    ort = None


class DynamicSessionOptions(ort.SessionOptions):
    config: dict | None = None

    def __init__(self):
        super().__init__()
        self.enable_mem_pattern = False

    @classmethod
    def from_sess_options(cls, sess_options: ort.SessionOptions):
        if isinstance(sess_options, DynamicSessionOptions):
            return sess_options.copy()
        return DynamicSessionOptions()

    def enable_static_dims(self, config: dict):
        self.config = config
        self.add_free_dimension_override_by_name("unet_sample_batch", config["hidden_batch_size"])
        self.add_free_dimension_override_by_name("unet_sample_channels", 4)
        self.add_free_dimension_override_by_name("unet_sample_height", config["height"] // 8)
        self.add_free_dimension_override_by_name("unet_sample_width", config["width"] // 8)
        self.add_free_dimension_override_by_name("unet_time_batch", 1)
        self.add_free_dimension_override_by_name("unet_hidden_batch", config["hidden_batch_size"])
        self.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        if config["is_sdxl"] and not config["is_refiner"]:
            self.add_free_dimension_override_by_name("unet_text_embeds_batch", config["hidden_batch_size"])
            self.add_free_dimension_override_by_name("unet_text_embeds_size", 1280)
            self.add_free_dimension_override_by_name("unet_time_ids_batch", config["hidden_batch_size"])
            self.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    def copy(self):
        sess_options = DynamicSessionOptions()
        if self.config is not None:
            sess_options.enable_static_dims(self.config)
        return sess_options


class TorchCompatibleModule:
    device = torch.device("cpu")
    dtype = torch.float32

    def named_modules(self): # dummy
        return ()

    def to(self, *_, **__):
        raise NotImplementedError

    def type(self, *_, **__):
        return self


class TemporalModule(TorchCompatibleModule):
    """
    Replace the models which are not able to be moved to CPU.
    """
    provider: Any
    path: str
    sess_options: ort.SessionOptions

    def __init__(self, provider: Any, path: str, sess_options: ort.SessionOptions):
        self.provider = provider
        self.path = path
        self.sess_options = sess_options

    def to(self, *args, **kwargs):
        from .utils import extract_device

        device = extract_device(args, kwargs)
        if device is not None and device.type != "cpu":
            from .execution_providers import TORCH_DEVICE_TO_EP
            provider = TORCH_DEVICE_TO_EP[device.type] if device.type in TORCH_DEVICE_TO_EP else self.provider
            return OnnxRuntimeModel.load_model(self.path, provider, DynamicSessionOptions.from_sess_options(self.sess_options))
        return self


class OnnxRuntimeModel(TorchCompatibleModule, diffusers.OnnxRuntimeModel):
    config = {} # dummy

    def to(self, *args, **kwargs):
        from modules.onnx_impl.utils import extract_device, move_inference_session

        device = extract_device(args, kwargs)
        if device is not None:
            self.device = device
            self.model = move_inference_session(self.model, device)
        return self


class VAEConfig:
    DEFAULTS = { "scaling_factor": 0.18215 }
    config: dict

    def __init__(self, config: dict):
        self.config = config

    def __getattr__(self, key):
        return self.config.get(key, VAEConfig.DEFAULTS.get(key, None))

    def get(self, key, default):
        return self.config.get(key, VAEConfig.DEFAULTS.get(key, default))


class VAE(TorchCompatibleModule):
    pipeline: Any

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    @property
    def config(self):
        return VAEConfig(self.pipeline.vae_decoder.config)

    @property
    def device(self):
        return self.pipeline.vae_decoder.device

    def encode(self, sample: torch.Tensor, *_, **__):
        sample_np = sample.cpu().numpy()
        return [
            torch.from_numpy(np.concatenate(
                [self.pipeline.vae_encoder(sample=sample_np[i : i + 1])[0] for i in range(sample_np.shape[0])]
            )).to(sample.device)
        ]

    def decode(self, latent_sample: torch.Tensor, *_, **__):
        latents_np = latent_sample.cpu().numpy()
        return [
            torch.from_numpy(np.concatenate(
                [self.pipeline.vae_decoder(latent_sample=latents_np[i : i + 1])[0] for i in range(latents_np.shape[0])]
            )).to(latent_sample.device)
        ]

    def to(self, *args, **kwargs):
        self.pipeline.vae_encoder = self.pipeline.vae_encoder.to(*args, **kwargs)
        self.pipeline.vae_decoder = self.pipeline.vae_decoder.to(*args, **kwargs)
        return self


def check_parameters_changed(p, refiner_enabled: bool):
    from modules import shared, sd_models
    if shared.sd_model.__class__.__name__ == "OnnxRawPipeline" or not shared.sd_model.__class__.__name__.startswith("Onnx"):
        return shared.sd_model
    compile_height = p.height
    compile_width = p.width
    if (shared.compiled_model_state is None or
    shared.compiled_model_state.height != compile_height
    or shared.compiled_model_state.width != compile_width
    or shared.compiled_model_state.batch_size != p.batch_size):
        log.info("Olive: Parameter change detected")
        log.info("Olive: Recompiling base model")
        sd_models.unload_model_weights(op='model')
        sd_models.reload_model_weights(op='model')
        if refiner_enabled:
            log.info("Olive: Recompiling refiner")
            sd_models.unload_model_weights(op='refiner')
            sd_models.reload_model_weights(op='refiner')
    shared.compiled_model_state.height = compile_height
    shared.compiled_model_state.width = compile_width
    shared.compiled_model_state.batch_size = p.batch_size
    return shared.sd_model


def preprocess_pipeline(p):
    from modules import shared, sd_models
    if "ONNX" not in shared.opts.diffusers_pipeline:
        log.warning(f"Unsupported pipeline for 'olive-ai' compile backend: {shared.opts.diffusers_pipeline}. You should select one of the ONNX pipelines.")
        return shared.sd_model
    if hasattr(shared.sd_model, "preprocess"):
        shared.sd_model = shared.sd_model.preprocess(p)
    if hasattr(shared.sd_refiner, "preprocess"):
        if shared.opts.onnx_unload_base:
            sd_models.unload_model_weights(op='model')
        shared.sd_refiner = shared.sd_refiner.preprocess(p)
        if shared.opts.onnx_unload_base:
            sd_models.reload_model_weights(op='model')
            shared.sd_model = shared.sd_model.preprocess(p)
    return shared.sd_model


def ORTPipelinePart_to(self, *args, **kwargs):
    self.parent_pipeline = self.parent_pipeline.to(*args, **kwargs)
    return self


def initialize_onnx():
    global initialized # pylint: disable=global-statement
    if initialized:
        return
    from modules import devices
    if not installed('onnx', quiet=True):
        return
    try: # may fail on onnx import
        import onnx # pylint: disable=unused-import
        from .execution_providers import ExecutionProvider, TORCH_DEVICE_TO_EP, available_execution_providers
        if devices.backend == "rocm":
            TORCH_DEVICE_TO_EP["cuda"] = ExecutionProvider.ROCm
        log.debug(f'ONNX: version={ort.__version__}, available={available_execution_providers}')

    except Exception as e:
        log.error(f'ONNX initialization: {e}')

    initialized = True


def initialize_onnx_pipelines():
    try: # may fail on onnx import
        import onnx # pylint: disable=unused-import
        OnnxRuntimeModel.__module__ = 'diffusers' # OnnxRuntimeModel Hijack.
        diffusers.OnnxRuntimeModel = OnnxRuntimeModel
        from .pipelines.onnx_stable_diffusion_pipeline import OnnxStableDiffusionPipeline
        from .pipelines.onnx_stable_diffusion_img2img_pipeline import OnnxStableDiffusionImg2ImgPipeline
        from .pipelines.onnx_stable_diffusion_inpaint_pipeline import OnnxStableDiffusionInpaintPipeline
        from .pipelines.onnx_stable_diffusion_upscale_pipeline import OnnxStableDiffusionUpscalePipeline
        from .pipelines.onnx_stable_diffusion_xl_pipeline import OnnxStableDiffusionXLPipeline
        from .pipelines.onnx_stable_diffusion_xl_img2img_pipeline import OnnxStableDiffusionXLImg2ImgPipeline
        diffusers.OnnxStableDiffusionPipeline = OnnxStableDiffusionPipeline
        diffusers.OnnxStableDiffusionImg2ImgPipeline = OnnxStableDiffusionImg2ImgPipeline
        diffusers.OnnxStableDiffusionInpaintPipeline = OnnxStableDiffusionInpaintPipeline
        diffusers.OnnxStableDiffusionUpscalePipeline = OnnxStableDiffusionUpscalePipeline
        diffusers.OnnxStableDiffusionXLPipeline = OnnxStableDiffusionXLPipeline
        diffusers.OnnxStableDiffusionXLImg2ImgPipeline = OnnxStableDiffusionXLImg2ImgPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionInpaintPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline
        diffusers.ORTStableDiffusionXLPipeline = diffusers.OnnxStableDiffusionXLPipeline # Huggingface model compatibility
        diffusers.ORTStableDiffusionXLImg2ImgPipeline = diffusers.OnnxStableDiffusionXLImg2ImgPipeline
    except Exception as e:
        log.error(f'ONNX initialization: {e}')


def install_olive():
    if installed("olive-ai"):
        return
    try:
        log.info('Installing Olive')
        install('onnx', 'onnx', ignore=True)
        install('olive-ai', 'olive-ai', ignore=True)
        import olive.workflows # pylint: disable=unused-import
    except Exception as e:
        log.error(f'Olive: Failed to load olive-ai: {e}')
    else:
        log.info('Olive: Please restart webui session.')
