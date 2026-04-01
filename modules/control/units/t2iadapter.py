import os
import time
from typing import Union
import threading
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, T2IAdapter, MultiAdapter, StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline # pylint: disable=unused-import
from modules.logger import log
from modules import errors, sd_models
from modules.control.units import detect


what = 'T2I-Adapter'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
predefined_sd15 = {
    'Segment': ('TencentARC/t2iadapter_seg_sd14v1', {}),
    'Zoe Depth': ('TencentARC/t2iadapter_zoedepth_sd15v1', {}),
    'OpenPose': ('TencentARC/t2iadapter_openpose_sd14v1', {}),
    'KeyPose': ('TencentARC/t2iadapter_keypose_sd14v1', {}),
    'Color': ('TencentARC/t2iadapter_color_sd14v1', {}),
    'Depth v1': ('TencentARC/t2iadapter_depth_sd14v1', {}),
    'Depth v2': ('TencentARC/t2iadapter_depth_sd15v2', {}),
    'Canny v1': ('TencentARC/t2iadapter_canny_sd14v1', {}),
    'Canny v2': ('TencentARC/t2iadapter_canny_sd15v2', {}),
    'Sketch v1': ('TencentARC/t2iadapter_sketch_sd14v1', {}),
    'Sketch v2': ('TencentARC/t2iadapter_sketch_sd15v2', {}),
    # 'Coadapter Canny': 'TencentARC/T2I-Adapter/models/coadapter-canny-sd15v1.pth',
    # 'Coadapter Color': 'TencentARC/T2I-Adapter/models/coadapter-color-sd15v1.pth',
    # 'Coadapter Depth': 'TencentARC/T2I-Adapter/models/coadapter-depth-sd15v1.pth',
    # 'Coadapter Fuser': 'TencentARC/T2I-Adapter/models/coadapter-fuser-sd15v1.pth',
    # 'Coadapter Sketch': 'TencentARC/T2I-Adapter/models/coadapter-sketch-sd15v1.pth',
    # 'Coadapter Style': 'TencentARC/T2I-Adapter/models/coadapter-style-sd15v1.pth',
}
predefined_sdxl = {
    'Canny XL': ('TencentARC/t2i-adapter-canny-sdxl-1.0', { 'use_safetensors': True, 'variant': 'fp16' }),
    'LineArt XL': ('TencentARC/t2i-adapter-lineart-sdxl-1.0', { 'use_safetensors': True, 'variant': 'fp16' }),
    'Sketch XL': ('TencentARC/t2i-adapter-sketch-sdxl-1.0', { 'use_safetensors': True, 'variant': 'fp16' }),
    'Zoe Depth XL': ('TencentARC/t2i-adapter-depth-zoe-sdxl-1.0', { 'use_safetensors': True, 'variant': 'fp16' }),
    'OpenPose XL': ('TencentARC/t2i-adapter-openpose-sdxl-1.0', { 'use_safetensors': True }),
    'Midas Depth XL': ('TencentARC/t2i-adapter-depth-midas-sdxl-1.0', { 'use_safetensors': True, 'variant': 'fp16' }),
}

models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
cache_dir = 'models/control/adapter'
load_lock = threading.Lock()


def list_models(refresh=False):
    import modules.shared
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = {}
    if modules.shared.sd_model_type == 'none':
        models = ['None']
    elif modules.shared.sd_model_type == 'sdxl':
        models = ['None'] + sorted(predefined_sdxl)
    elif modules.shared.sd_model_type == 'sd':
        models = ['None'] + sorted(predefined_sd15)
    else:
        log.warning(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(list(predefined_sd15) + list(predefined_sdxl))
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class AdapterModel(T2IAdapter):
    pass


class Adapter():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: AdapterModel = None
        self.model_id: str = model_id
        self.device = device
        self.dtype = dtype
        self.load_config = { 'cache_dir': cache_dir, 'use_safetensors': False }
        if load_config is not None:
            self.load_config.update(load_config)
        if model_id is not None:
            self.load()

    def __str__(self):
        return f' T2IAdapter(id={self.model_id} model={self.model.__class__.__name__})' if self.model_id and self.model else ''

    def reset(self):
        if self.model is not None:
            debug(f'Control {what} model unloaded')
        self.model = None
        self.model_id = None

    def load(self, model_id: str = None, force: bool = True) -> str:
        with load_lock:
            try:
                t0 = time.time()
                model_id = model_id or self.model_id
                if model_id is None or model_id == 'None':
                    self.reset()
                    return
                if model_id not in all_models:
                    log.error(f'Control {what} unknown model: id="{model_id}" available={list(all_models)}')
                    return
                model_path, model_args = all_models[model_id]
                self.load_config.update(model_args)
                from modules.shared import opts
                if opts.offline_mode:
                    self.load_config["local_files_only"] = True
                    os.environ['HF_HUB_OFFLINE'] = '1'
                else:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    os.unsetenv('HF_HUB_OFFLINE')
                if model_path is None:
                    log.error(f'Control {what} model load failed: id="{model_id}" error=unknown model id')
                    return
                if model_id == self.model_id and not force:
                    # log.debug(f'Control {what} model: id="{model_id}" path="{model_path}" already loaded')
                    return
                log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}"')
                if model_path.endswith('.pth') or model_path.endswith('.pt') or model_path.endswith('.safetensors') or model_path.endswith('.bin'):
                    from huggingface_hub import hf_hub_download
                    parts = model_path.split('/')
                    repo_id = f'{parts[0]}/{parts[1]}'
                    filename = '/'.join(parts[2:])
                    model = hf_hub_download(repo_id, filename, **self.load_config)
                    self.model = T2IAdapter.from_pretrained(model, **self.load_config)
                else:
                    self.model = T2IAdapter.from_pretrained(model_path, **self.load_config)
                if self.device is not None:
                    self.model.to(self.device)
                if self.dtype is not None:
                    self.model.to(self.dtype)
                t1 = time.time()
                self.model_id = model_id
                log.debug(f'Control {what} loaded: id="{model_id}" path="{model_path}" time={t1-t0:.2f}')
                return f'{what} loaded model: {model_id}'
            except Exception as e:
                log.error(f'Control {what} model load failed: id="{model_id}" error={e}')
                errors.display(e, f'Control {what} load')
                return f'{what} failed to load model: {model_id}'


class AdapterPipeline():
    def __init__(self, adapter: Union[T2IAdapter, list[T2IAdapter]], pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline], dtype = None):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline] = None
        if pipeline is None:
            log.error(f'Control {what} pipeline: model not loaded')
            return
        if isinstance(adapter, list) and len(adapter) > 1:
            adapter = MultiAdapter(adapter)
        adapter.to(device=pipeline.device, dtype=pipeline.dtype)
        """
        pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["sd-t2iadapter"] = StableDiffusionAdapterPipeline
        pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["sd-t2iadapter"] = StableDiffusionAdapterPipeline
        pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["sd-t2iadapter"] = StableDiffusionAdapterPipeline
        pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["sdxl-t2iadapter"] = StableDiffusionXLAdapterPipeline
        pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["sdxl-t2iadapter"] = StableDiffusionXLAdapterPipeline
        pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["sdxl-t2iadapter"] = StableDiffusionXLAdapterPipeline
        """
        if pipeline.__class__.__name__ == 'StableDiffusionAdapterPipeline' or pipeline.__class__.__name__ == 'StableDiffusionXLAdapterPipeline':
            pass # already initialized
        if detect.is_sdxl(pipeline):
            self.pipeline = StableDiffusionXLAdapterPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                adapter=adapter,
            )
            sd_models.move_model(self.pipeline, pipeline.device)
            sd_models.apply_balanced_offload(self.pipeline, force=True)
        elif detect.is_sd15(pipeline):
            self.pipeline = StableDiffusionAdapterPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
                adapter=adapter,
            )
            sd_models.move_model(self.pipeline, pipeline.device)
            sd_models.apply_balanced_offload(self.pipeline, force=True)
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return
        if dtype is not None and self.pipeline is not None:
            self.pipeline.dtype = dtype
        t1 = time.time()
        if self.pipeline is not None:
            log.debug(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')
        else:
            log.error(f'Control {what} pipeline: not initialized')


    def restore(self):
        self.pipeline = None
        return self.orig_pipeline
