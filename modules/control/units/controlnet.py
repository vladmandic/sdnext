import os
import time
import threading
from typing import Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline, StableDiffusion3Pipeline, ControlNetModel
from modules.control.units import detect
from modules.shared import log, opts, cmd_opts, state, listdir
from modules import errors, sd_models, devices, model_quant
from modules.processing import StableDiffusionProcessingControl


what = 'ControlNet'
debug = os.environ.get('SD_CONTROL_DEBUG', None) is not None
debug_log = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
predefined_sd15 = {
    'Canny': "lllyasviel/control_v11p_sd15_canny",
    'Depth': "lllyasviel/control_v11f1p_sd15_depth",
    'HED': "lllyasviel/sd-controlnet-hed",
    'IP2P': "lllyasviel/control_v11e_sd15_ip2p",
    'LineArt': "lllyasviel/control_v11p_sd15_lineart",
    'LineArt Anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'MLDS': "lllyasviel/control_v11p_sd15_mlsd",
    'NormalBae': "lllyasviel/control_v11p_sd15_normalbae",
    'OpenPose': "lllyasviel/control_v11p_sd15_openpose",
    'Scribble': "lllyasviel/control_v11p_sd15_scribble",
    'Segment': "lllyasviel/control_v11p_sd15_seg",
    'Shuffle': "lllyasviel/control_v11e_sd15_shuffle",
    'SoftEdge': "lllyasviel/control_v11p_sd15_softedge",
    'Tile': "lllyasviel/control_v11f1e_sd15_tile",
    'Depth Anything': 'vladmandic/depth-anything',
    'Canny FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_canny.safetensors',
    'Inpaint FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_inpaint.safetensors',
    'LineArt Anime FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_animeline.safetensors',
    'LineArt FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_lineart.safetensors',
    'MLSD FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_mlsd.safetensors',
    'NormalBae FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_normal.safetensors',
    'OpenPose FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_openpose.safetensors',
    'Pix2Pix FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_pix2pix.safetensors',
    'Scribble FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_scribble.safetensors',
    'Segment FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_seg.safetensors',
    'Shuffle FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_shuffle.safetensors',
    'SoftEdge FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_softedge.safetensors',
    'Tile FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_tileE.safetensors',
    'CiaraRowles TemporalNet': "CiaraRowles/TemporalNet",
    'Ciaochaos Recolor': 'ioclab/control_v1p_sd15_brightness',
    'Ciaochaos Illumination': 'ioclab/control_v1u_sd15_illumination/illumination20000.safetensors',
}
predefined_sdxl = {
    'Canny Small XL': 'diffusers/controlnet-canny-sdxl-1.0-small',
    'Canny Mid XL': 'diffusers/controlnet-canny-sdxl-1.0-mid',
    'Canny XL': 'diffusers/controlnet-canny-sdxl-1.0',
    'Depth Zoe XL': 'diffusers/controlnet-zoe-depth-sdxl-1.0',
    'Depth Mid XL': 'diffusers/controlnet-depth-sdxl-1.0-mid',
    'OpenPose XL': 'thibaud/controlnet-openpose-sdxl-1.0/bin',
    'Xinsir Union XL': 'xinsir/controlnet-union-sdxl-1.0',
    'Xinsir ProMax XL': 'brad-twinkl/controlnet-union-sdxl-1.0-promax',
    'Xinsir OpenPose XL': 'xinsir/controlnet-openpose-sdxl-1.0',
    'Xinsir Canny XL': 'xinsir/controlnet-canny-sdxl-1.0',
    'Xinsir Depth XL': 'xinsir/controlnet-depth-sdxl-1.0',
    'Xinsir Scribble XL': 'xinsir/controlnet-scribble-sdxl-1.0',
    'Xinsir Anime Painter XL': 'xinsir/anime-painter',
    'Xinsir Tile XL': 'xinsir/controlnet-tile-sdxl-1.0',
    'NoobAI Canny XL': 'Eugeoter/noob-sdxl-controlnet-canny',
    'NoobAI Lineart Anime XL': 'Eugeoter/noob-sdxl-controlnet-lineart_anime',
    'NoobAI Depth XL': 'Eugeoter/noob-sdxl-controlnet-depth',
    'NoobAI Normal XL': 'Eugeoter/noob-sdxl-controlnet-normal',
    'NoobAI SoftEdge XL': 'Eugeoter/noob-sdxl-controlnet-softedge_hed',
    'NoobAI OpenPose XL': 'einar77/noob-openpose',
    'TTPlanet Tile Realistic XL': 'Yakonrus/SDXL_Controlnet_Tile_Realistic_v2',
    # 'StabilityAI Canny R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-canny-rank128.safetensors',
    # 'StabilityAI Depth R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-depth-rank128.safetensors',
    # 'StabilityAI Recolor R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors',
    # 'StabilityAI Sketch R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors',
    # 'StabilityAI Canny R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-canny-rank256.safetensors',
    # 'StabilityAI Depth R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-depth-rank256.safetensors',
    # 'StabilityAI Recolor R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors',
    # 'StabilityAI Sketch R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors',
}
predefined_f1 = {
    "InstantX Union F1": 'InstantX/FLUX.1-dev-Controlnet-Union',
    "InstantX Canny F1": 'InstantX/FLUX.1-dev-Controlnet-Canny',
    "JasperAI Depth F1": 'jasperai/Flux.1-dev-Controlnet-Depth',
    "BlackForrestLabs Canny LoRA F1": '/huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/flux1-canny-dev-lora.safetensors',
    "BlackForrestLabs Depth LoRA F1": '/huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/flux1-depth-dev-lora.safetensors',
    "JasperAI Surface Normals F1": 'jasperai/Flux.1-dev-Controlnet-Surface-Normals',
    "JasperAI Upscaler F1": 'jasperai/Flux.1-dev-Controlnet-Upscaler',
    "Shakker-Labs Union F1": 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
    "Shakker-Labs Pose F1": 'Shakker-Labs/FLUX.1-dev-ControlNet-Pose',
    "Shakker-Labs Depth F1": 'Shakker-Labs/FLUX.1-dev-ControlNet-Depth',
    "XLabs-AI Canny F1": 'XLabs-AI/flux-controlnet-canny-diffusers',
    "XLabs-AI Depth F1": 'XLabs-AI/flux-controlnet-depth-diffusers',
    "XLabs-AI HED F1": 'XLabs-AI/flux-controlnet-hed-diffusers',
    "LibreFlux Segment F1": 'neuralvfx/LibreFlux-ControlNet',
}
predefined_sd3 = {
    "StabilityAI Canny SD35": 'diffusers-internal-dev/sd35-controlnet-canny-8b',
    "StabilityAI Depth SD35": 'diffusers-internal-dev/sd35-controlnet-depth-8b',
    "StabilityAI Blur SD35": 'diffusers-internal-dev/sd35-controlnet-blur-8b',
    "InstantX Canny SD35": 'InstantX/SD3-Controlnet-Canny',
    "InstantX Pose SD35": 'InstantX/SD3-Controlnet-Pose',
    "InstantX Depth SD35": 'InstantX/SD3-Controlnet-Depth',
    "InstantX Tile SD35": 'InstantX/SD3-Controlnet-Tile',
    "Alimama Inpainting SD35": 'alimama-creative/SD3-Controlnet-Inpainting',
    "Alimama SoftEdge SD35": 'alimama-creative/SD3-Controlnet-Softedge',
}
predefined_qwen = {
    "InstantX Union Qwen": 'InstantX/Qwen-Image-ControlNet-Union',
}
predefined_hunyuandit = {
    "HunyuanDiT Canny": 'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny',
    "HunyuanDiT Pose": 'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Pose',
    "HunyuanDiT Depth": 'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Depth',
}

variants = {
    'NoobAI Canny XL': 'fp16',
    'NoobAI Lineart Anime XL': 'fp16',
    'NoobAI Depth XL': 'fp16',
    'NoobAI Normal XL': 'fp16',
    'NoobAI SoftEdge XL': 'fp16',
    'TTPlanet Tile Realistic XL': 'fp16',
}

subfolders = {
    "LibreFlux Segment F1": 'controlnet',
}

remote_code = {
    "LibreFlux Segment F1": True,
}

models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
all_models.update(predefined_f1)
all_models.update(predefined_sd3)
all_models.update(predefined_qwen)
all_models.update(predefined_hunyuandit)
cache_dir = 'models/control/controlnet'
load_lock = threading.Lock()


def find_models():
    path = os.path.join(opts.control_dir, 'controlnet')
    files = listdir(path)
    folders = [f for f in files if os.path.isdir(f) if os.path.exists(os.path.join(f, 'config.json'))]
    files = [f for f in files if f.endswith('.safetensors')]
    downloaded_models = {}
    for f in files:
        basename = os.path.splitext(os.path.relpath(f, path))[0]
        downloaded_models[basename] = f
    for f in folders:
        basename = os.path.relpath(f, path)
        downloaded_models[basename] = f
    all_models.update(downloaded_models)
    return downloaded_models

find_models()


def api_list_models(model_type: str = None):
    import modules.shared
    model_type = model_type or modules.shared.sd_model_type
    model_list = []
    if model_type == 'sd' or model_type == 'all':
        model_list += list(predefined_sd15)
    if model_type == 'sdxl' or model_type == 'all':
        model_list += list(predefined_sdxl)
    if model_type == 'f1' or model_type == 'all':
        model_list += list(predefined_f1)
    if model_type == 'sd3' or model_type == 'all':
        model_list += list(predefined_sd3)
    if model_type == 'qwen' or model_type == 'all':
        model_list += list(predefined_qwen)
    if model_type == 'hunyuandit' or model_type == 'all':
        model_list += list(predefined_hunyuandit)
    model_list += sorted(find_models())
    return model_list


def list_models(refresh=False):
    import modules.shared
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = {}
    if modules.shared.sd_model_type == 'none':
        models = ['None']
    elif modules.shared.sd_model_type == 'sdxl':
        models = ['None'] + list(predefined_sdxl) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd':
        models = ['None'] + list(predefined_sd15) + sorted(find_models())
    elif modules.shared.sd_model_type == 'f1':
        models = ['None'] + list(predefined_f1) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd3':
        models = ['None'] + list(predefined_sd3) + sorted(find_models())
    elif modules.shared.sd_model_type == 'qwen':
        models = ['None'] + list(predefined_qwen) + sorted(find_models())
    elif modules.shared.sd_model_type == 'hunyuandit':
        models = ['None'] + list(predefined_hunyuandit) + sorted(find_models())
    else:
        log.warning(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(predefined_sd15) + sorted(predefined_sdxl) + sorted(predefined_f1) + sorted(predefined_sd3) + sorted(find_models())
    debug_log(f'Control list {what}: path={cache_dir} models={models}')
    return models


class ControlNet():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: ControlNetModel = None
        self.model_id: str = model_id
        self.device = device
        self.dtype = dtype
        self.load_config = { 'cache_dir': cache_dir }
        if load_config is not None:
            self.load_config.update(load_config)
        if opts.offline_mode:
            self.load_config["local_files_only"] = True
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.unsetenv('HF_HUB_OFFLINE')
        if model_id is not None:
            self.load()

    def __str__(self):
        return f' ControlNet(id={self.model_id} model={self.model.__class__.__name__})' if self.model_id and self.model else ''

    def reset(self):
        if self.model is not None:
            debug_log(f'Control {what} model unloaded')
            self.model = None
            self.model_id = None
            devices.torch_gc(force=True, reason='controlnet')

    def get_class(self, model_id:str=''):
        from modules import shared
        if shared.sd_model_type == 'none':
            _load = shared.sd_model # trigger a load
        if shared.sd_model_type == 'sd':
            from diffusers import ControlNetModel as cls # pylint: disable=reimported
            config = 'lllyasviel/control_v11p_sd15_canny'
        elif shared.sd_model_type == 'sdxl':
            if 'union' in model_id.lower():
                from diffusers import ControlNetUnionModel as cls
                config = 'xinsir/controlnet-union-sdxl-1.0'
            elif 'promax' in model_id.lower():
                from diffusers import ControlNetUnionModel as cls
                config = 'brad-twinkl/controlnet-union-sdxl-1.0-promax'
            else:
                from diffusers import ControlNetModel as cls # pylint: disable=reimported # sdxl shares same model class
                config = 'Eugeoter/noob-sdxl-controlnet-canny'
        elif shared.sd_model_type == 'f1':
            from diffusers import FluxControlNetModel as cls
            config = 'InstantX/FLUX.1-dev-Controlnet-Union'
        elif shared.sd_model_type == 'sd3':
            from diffusers import SD3ControlNetModel as cls
            config = 'InstantX/SD3-Controlnet-Canny'
        elif shared.sd_model_type == 'qwen':
            from diffusers import QwenImageControlNetModel as cls
            config = 'InstantX/Qwen-Image-ControlNet-Union'
        elif shared.sd_model_type == 'hunyuandit':
            from diffusers import HunyuanDiT2DControlNetModel as cls
            config = 'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny'
        else:
            log.error(f'Control {what}: type={shared.sd_model_type} unsupported model')
            return None, None
        return cls, config

    def load_safetensors(self, model_id, model_path, cls, config): # pylint: disable=unused-argument
        name = os.path.splitext(model_path)[0]
        config_path = None
        if not os.path.exists(model_path):
            import huggingface_hub as hf
            parts = model_path.split('/')
            repo_id = f'{parts[0]}/{parts[1]}'
            filename = os.path.splitext('/'.join(parts[2:]))[0]
            model_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.safetensors', cache_dir=cache_dir)
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.yaml', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.json', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
        elif os.path.exists(name + '.yaml'):
            config_path = f'{name}.yaml'
        elif os.path.exists(name + '.json'):
            config_path = f'{name}.json'
        if config_path is not None:
            self.load_config['original_config_file '] = config_path
        self.model = cls.from_single_file(model_path, config=config, **self.load_config)

    def load(self, model_id: str = None, force: bool = False) -> str:
        with load_lock:
            try:
                t0 = time.time()
                model_id = model_id or self.model_id
                if model_id is None or model_id == 'None':
                    self.reset()
                    return
                if model_id not in all_models:
                    log.error(f'Control {what}: id="{model_id}" available={list(all_models)} unknown model')
                    return
                model_path = all_models[model_id]
                if model_path == '':
                    return
                if model_path is None:
                    log.error(f'Control {what} model load: id="{model_id}" unknown model id')
                    return
                if 'lora' in model_id.lower():
                    self.model = model_path
                    return
                if model_id == self.model_id and not force:
                    # log.debug(f'Control {what} model: id="{model_id}" path="{model_path}" already loaded')
                    return
                log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}"')
                cls, config = self.get_class(model_id)
                if cls is None:
                    log.error(f'Control {what} model load: id="{model_id}" unknown base model')
                    return
                self.reset()
                jobid = state.begin(f'Load {what}')
                if model_path.endswith('.safetensors'):
                    self.load_safetensors(model_id, model_path, cls, config)
                else:
                    kwargs = {}
                    if '/bin' in model_path:
                        model_path = model_path.replace('/bin', '')
                        self.load_config['use_safetensors'] = False
                    else:
                        self.load_config['use_safetensors'] = True
                    if variants.get(model_id, None) is not None:
                        kwargs['variant'] = variants[model_id]
                    if subfolders.get(model_id, None) is not None:
                        kwargs['subfolder'] = subfolders[model_id]
                    if remote_code.get(model_id, None) is not None:
                        kwargs['trust_remote_code'] = remote_code[model_id]
                    try:
                        self.model = cls.from_pretrained(model_path, **self.load_config, **kwargs)
                    except Exception as e:
                        log.error(f'Control {what} model load: id="{model_id}" {e}')
                        if debug:
                            errors.display(e, 'Control')
                if self.model is None:
                    return
                if not cmd_opts.lowvram: # lowvram will cause unet<->controlnet to ping-pong but saves more memory
                    self.model.offload_never = True
                if self.dtype is not None:
                    self.model.to(self.dtype)
                if self.device is not None:
                    self.model.to_empty(device=self.device) # model could be sparse
                if "Control" in opts.sdnq_quantize_weights:
                    try:
                        log.debug(f'Control {what} model SDNQ quantize: id="{model_id}"')
                        from modules.model_quant import sdnq_quantize_model
                        self.model = sdnq_quantize_model(self.model)
                    except Exception as e:
                        log.error(f'Control {what} model SDNQ Compression failed: id="{model_id}" {e}')
                elif "Control" in opts.optimum_quanto_weights:
                    try:
                        log.debug(f'Control {what} model Optimum Quanto: id="{model_id}"')
                        model_quant.load_quanto('Load model: type=Control')
                        from modules.model_quant import optimum_quanto_model
                        self.model = optimum_quanto_model(self.model)
                    except Exception as e:
                        log.error(f'Control {what} model Optimum Quanto: id="{model_id}" {e}')
                elif "Control" in opts.torchao_quantization:
                    try:
                        log.debug(f'Control {what} model Torch AO: id="{model_id}"')
                        model_quant.load_torchao('Load model: type=Control')
                        from modules.model_quant import torchao_quantization
                        self.model = torchao_quantization(self.model)
                    except Exception as e:
                        log.error(f'Control {what} model Torch AO: id="{model_id}" {e}')
                if self.device is not None:
                    sd_models.move_model(self.model, self.device)
                if "Control" in opts.cuda_compile:
                    try:
                        from modules.sd_models_compile import compile_torch
                        self.model = compile_torch(self.model, apply_to_components=False, op="Control")
                    except Exception as e:
                        log.warning(f"Control compile error: {e}")
                t1 = time.time()
                self.model_id = model_id
                log.info(f'Control {what} model loaded: id="{self.model_id}" path="{model_path}" cls={cls.__name__} time={t1-t0:.2f}')
                state.end(jobid)
                return f'{what} loaded model: {self.model_id}'
            except Exception as e:
                log.error(f'Control {what} model load: id="{model_id}" {e}')
                errors.display(e, f'Control {what} load')
                return f'{what} failed to load model: {model_id}'


class ControlNetPipeline():
    def __init__(self,
                 controlnet: Union[ControlNetModel, list[ControlNetModel]],
                 pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline],
                 dtype = None,
                 p: StableDiffusionProcessingControl = None, # pylint: disable=unused-argument
                ):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None

        controlnets = controlnet if isinstance(controlnet, list) else [controlnet]
        loras = [cn for cn in controlnets if isinstance(cn, str)]
        controlnets = [cn for cn in controlnets if not isinstance(cn, str)]

        if pipeline is None:
            log.error('Control model pipeline: model not loaded')
            return
        elif detect.is_sdxl(pipeline) and len(controlnets) > 0:
            from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetUnionPipeline
            classes = [c.__class__.__name__ for c in controlnets]
            if any(c == 'ControlNetUnionModel' for c in classes):
                if not all(c == 'ControlNetUnionModel' for c in classes):
                    log.warning(f'Control {what}: units={classes} mixed type is not supported')
                    return
                if isinstance(controlnets, list) and len(controlnets) == 1:
                    controlnets = controlnets[0]
                cls = StableDiffusionXLControlNetUnionPipeline
            else:
                cls = StableDiffusionXLControlNetPipeline
            self.pipeline = cls(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                image_encoder=getattr(pipeline, 'image_encoder', None),
                controlnet=controlnets, # can be a list
            )
        elif detect.is_f1(pipeline) and len(controlnets) > 0:
            from diffusers import FluxControlNetPipeline
            self.pipeline = FluxControlNetPipeline(
                vae=pipeline.vae.to(devices.device),
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                controlnet=controlnets, # can be a list
            )
        elif detect.is_sd3(pipeline) and len(controlnets) > 0:
            from diffusers import StableDiffusion3ControlNetPipeline
            self.pipeline = StableDiffusion3ControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                text_encoder_3=pipeline.text_encoder_3,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                tokenizer_3=pipeline.tokenizer_3,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                controlnet=controlnets, # can be a list
            )
        elif detect.is_sd15(pipeline) and len(controlnets) > 0:
            from diffusers import StableDiffusionControlNetPipeline
            self.pipeline = StableDiffusionControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                image_encoder=getattr(pipeline, 'image_encoder', None),
                requires_safety_checker=False,
                safety_checker=None,
                controlnet=controlnets, # can be a list
            )
            sd_models.move_model(self.pipeline, pipeline.device)
        elif detect.is_qwen(pipeline) and len(controlnets) > 0:
            from diffusers import QwenImageControlNetPipeline
            self.pipeline = QwenImageControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                controlnet=controlnets[0] if isinstance(controlnets, list) else controlnets, # can be a list
            )
        elif detect.is_hunyuandit(pipeline) and len(controlnets) > 0:
            from diffusers import HunyuanDiTControlNetPipeline
            self.pipeline = HunyuanDiTControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer_2=pipeline.tokenizer_2,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                safety_checker=None,
                feature_extractor=None,
                controlnet=controlnets[0] if isinstance(controlnets, list) else controlnets, # can be a list
            )
        elif len(loras) > 0:
            self.pipeline = pipeline
            for lora in loras:
                log.debug(f'Control {what} pipeline: lora="{lora}"')
                lora = lora.replace('/huggingface.co/', '')
                self.pipeline.load_lora_weights(lora)
                """
                if p is not None:
                    p.prompt += f'<lora:{lora}:1.0>'
                """
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return

        if self.pipeline is None:
            log.error(f'Control {what} pipeline: not initialized')
            return
        if dtype is not None:
            self.pipeline = self.pipeline.to(dtype)

        controlnet = None # free up memory
        controlnets = None
        sd_models.copy_diffuser_options(self.pipeline, pipeline)
        if opts.diffusers_offload_mode == 'none':
            sd_models.move_model(self.pipeline, devices.device)
        sd_models.clear_caches()
        sd_models.set_diffuser_offload(self.pipeline, 'model', force=True)

        t1 = time.time()
        debug_log(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')

    def restore(self):
        if self.pipeline is not None and hasattr(self.pipeline, 'unload_lora_weights'):
            self.pipeline.unload_lora_weights()
        self.pipeline = None
        return self.orig_pipeline
