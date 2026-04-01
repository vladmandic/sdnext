import os
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log


def load_transformer(repo_id, diffusers_load_config=None, subfolder='transformer'):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    fn = None

    if 'VACE' in repo_id:
        transformer_cls = diffusers.WanVACETransformer3DModel
    else:
        transformer_cls = diffusers.WanTransformer3DModel

    if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
        from modules import sd_unet
        if shared.opts.sd_unet not in list(sd_unet.unet_dict):
            log.error(f'Load module: type=Transformer not found: {shared.opts.sd_unet}')
            return None
        fn = sd_unet.unet_dict[shared.opts.sd_unet] if os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]) else None

    if fn is not None and 'gguf' in fn.lower():
        log.error('Load model: type=WanAI format="gguf" unsupported')
        transformer = None
    elif fn is not None and 'safetensors' in fn.lower():
        log.debug(f'Load model: type=WanAI {subfolder}="{fn}" quant="{model_quant.get_quant(repo_id)}" args={load_args}')
        transformer = transformer_cls.from_single_file(
            fn,
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    else:
        log.debug(f'Load model: type=WanAI {subfolder}="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
        transformer = transformer_cls.from_pretrained(
            repo_id,
            subfolder=subfolder,
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoder(repo_id, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    repo_id = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' if 'Wan2.' in repo_id else repo_id # always use shared umt5
    log.debug(f'Load model: type=WanAI te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.UMT5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
        sd_models.move_model(text_encoder, devices.cpu)
    return text_encoder


def load_wan(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    boundary_ratio = None
    if 'a14b' in repo_id.lower() or 'fun-14b' in repo_id.lower():
        if shared.opts.model_wan_stage == 'high noise' or shared.opts.model_wan_stage == 'first':
            transformer = load_transformer(repo_id, diffusers_load_config, 'transformer')
            transformer_2 = None
            boundary_ratio = 0.0
        elif shared.opts.model_wan_stage == 'low noise' or shared.opts.model_wan_stage == 'second':
            transformer = None
            transformer_2 = load_transformer(repo_id, diffusers_load_config, 'transformer_2')
            boundary_ratio = 1000.0
        elif shared.opts.model_wan_stage == 'combined' or shared.opts.model_wan_stage == 'both':
            transformer = load_transformer(repo_id, diffusers_load_config, 'transformer')
            transformer_2 = load_transformer(repo_id, diffusers_load_config, 'transformer_2')
            boundary_ratio = shared.opts.model_wan_boundary
        else:
            log.error(f'Load model: type=WanAI stage="{shared.opts.model_wan_stage}" unsupported')
            return None
    else:
        transformer = load_transformer(repo_id, diffusers_load_config, 'transformer')
        transformer_2 = None

    text_encoder = load_text_encoder(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')

    if 'Wan2.2-I2V' in repo_id:
        pipe_cls = diffusers.WanImageToVideoPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["wanai"] = diffusers.WanImageToVideoPipeline
    elif 'Wan2.2-VACE' in repo_id:
        pipe_cls = diffusers.WanVACEPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["wanai"] = diffusers.WanVACEPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["wanai"] = diffusers.WanVACEPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["wanai"] = diffusers.WanVACEPipeline
    else:
        from pipelines.wan.wan_image import WanImagePipeline
        pipe_cls = diffusers.WanPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["wanai"] = diffusers.WanPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["wanai"] = WanImagePipeline
    log.debug(f'Load model: type=WanAI model="{checkpoint_info.name}" repo="{repo_id}" cls={pipe_cls.__name__} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args} stage="{shared.opts.model_wan_stage}" boundary={boundary_ratio}')
    pipe = pipe_cls.from_pretrained(
        repo_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=text_encoder,
        boundary_ratio=boundary_ratio,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'num_frames': 1,
        'output_type': 'np',
    }

    del text_encoder
    del transformer
    del transformer_2

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
