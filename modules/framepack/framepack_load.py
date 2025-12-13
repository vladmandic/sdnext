import os
import time
from modules import shared, devices, errors, sd_models, sd_checkpoint, model_quant


models = {
    'bi-directional': 'lllyasviel/FramePackI2V_HY',
    'forward-only': 'lllyasviel/FramePack_F1_I2V_HY_20250503',
}
default_model = {
    'pipeline': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': '' },
    'vae': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'vae' },
    'text_encoder': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'text_encoder' },
    'tokenizer': {'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'tokenizer' },
    # 'text_encoder': { 'repo': 'Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'subfolder': '' },
    # 'tokenizer': { 'repo': 'Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'subfolder': '' },
    # 'text_encoder': { 'repo': 'xtuner/llava-llama-3-8b-v1_1-transformers', 'subfolder': '' },
    # 'tokenizer': {'repo': 'xtuner/llava-llama-3-8b-v1_1-transformers', 'subfolder': '' },
    'text_encoder_2': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'text_encoder_2' },
    'tokenizer_2': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'tokenizer_2' },
    'feature_extractor': { 'repo': 'lllyasviel/flux_redux_bfl', 'subfolder': 'feature_extractor' },
    'image_encoder': { 'repo': 'lllyasviel/flux_redux_bfl', 'subfolder': 'image_encoder' },
    'transformer': { 'repo': models.get('bi-directional'), 'subfolder': '' },
}
model = default_model.copy()


def split_url(url):
    if url.count('/') == 1:
        url += '/'
    if url.count('/') != 2:
        raise ValueError(f'Invalid URL: {url}')
    url = [section.strip() for section in url.split('/')]
    return { 'repo': f'{url[0]}/{url[1]}', 'subfolder': url[2] }


def set_model(receipe: str=None):
    if receipe is None or receipe == '':
        return
    lines = [line.strip() for line in receipe.split('\n') if line.strip() != '' and ':' in line]
    for line in lines:
        k, v = line.split(':', 1)
        k = k.strip()
        if k not in default_model.keys():
            shared.log.warning(f'FramePack receipe: key={k} invalid')
        model[k] = split_url(v)
        shared.log.debug(f'FramePack receipe: set {k}={model[k]}')


def get_model():
    receipe = ''
    for k, v in model.items():
        receipe += f'{k}: {v["repo"]}/{v["subfolder"]}\n'
    return receipe.strip()


def reset_model():
    global model # pylint: disable=global-statement
    model = default_model.copy()
    shared.log.debug('FramePack receipe: reset')
    return ''


def load_model(variant:str=None, pipeline:str=None, text_encoder:str=None, text_encoder_2:str=None, feature_extractor:str=None, image_encoder:str=None, transformer:str=None):
    shared.state.begin('Load FramePack')
    if variant is not None:
        if variant not in models.keys():
            raise ValueError(f'FramePack: variant="{variant}" invalid')
        model['transformer']['repo'] = models[variant]
    if pipeline is not None:
        model['pipeline'] = split_url(pipeline)
    if text_encoder is not None:
        model['text_encoder'] = split_url(text_encoder)
    if text_encoder_2 is not None:
        model['text_encoder_2'] = split_url(text_encoder_2)
    if feature_extractor is not None:
        model['feature_extractor'] = split_url(feature_extractor)
    if image_encoder is not None:
        model['image_encoder'] = split_url(image_encoder)
    if transformer is not None:
        model['transformer'] = split_url(transformer)
    # shared.log.trace(f'FramePack load: {model}')

    try:
        import diffusers
        from diffusers import HunyuanVideoImageToVideoPipeline, AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
        from modules.framepack.pipeline.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

        class FramepackHunyuanVideoPipeline(HunyuanVideoImageToVideoPipeline): # inherit and override
            def __init__(
                self,
                text_encoder: LlamaModel,
                tokenizer: LlamaTokenizerFast,
                text_encoder_2: CLIPTextModel,
                tokenizer_2: CLIPTokenizer,
                vae: AutoencoderKLHunyuanVideo,
                feature_extractor: SiglipImageProcessor,
                image_processor: SiglipVisionModel,
                transformer: HunyuanVideoTransformer3DModelPacked,
                scheduler,
            ):
                super().__init__(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    transformer=transformer,
                    image_processor=image_processor,
                    scheduler=scheduler,
                )
                self.register_modules(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    feature_extractor=feature_extractor,
                    image_processor=image_processor,
                    transformer=transformer,
                    scheduler=scheduler,
                )

        sd_models.unload_model_weights()
        t0 = time.time()

        sd_models.hf_auth_check(model["transformer"]["repo"])
        sd_models.hf_auth_check(model["text_encoder"]["repo"])
        sd_models.hf_auth_check(model["text_encoder_2"]["repo"])

        offline_config = {}
        if shared.opts.offline_mode:
            offline_config["local_files_only"] = True
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.unsetenv('HF_HUB_OFFLINE')

        shared.log.debug(f'FramePack load: module=llm {model["text_encoder"]}')
        load_args, quant_args = model_quant.get_dit_args({}, module='TE', device_map=True)
        text_encoder = LlamaModel.from_pretrained(model["text_encoder"]["repo"], subfolder=model["text_encoder"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args, **offline_config)
        tokenizer = LlamaTokenizerFast.from_pretrained(model["tokenizer"]["repo"], subfolder=model["tokenizer"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **offline_config)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        sd_models.move_model(text_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=te {model["text_encoder_2"]}')
        text_encoder_2 = CLIPTextModel.from_pretrained(model["text_encoder_2"]["repo"], subfolder=model["text_encoder_2"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, **offline_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model["pipeline"]["repo"], subfolder='tokenizer_2', cache_dir=shared.opts.hfcache_dir, **offline_config)
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval()
        sd_models.move_model(text_encoder_2, devices.cpu)

        shared.log.debug(f'FramePack load: module=vae {model["vae"]}')
        vae = AutoencoderKLHunyuanVideo.from_pretrained(model["vae"]["repo"], subfolder=model["vae"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, **offline_config)
        vae.requires_grad_(False)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        sd_models.move_model(vae, devices.cpu)

        shared.log.debug(f'FramePack load: module=encoder {model["feature_extractor"]} model={model["image_encoder"]}')
        feature_extractor = SiglipImageProcessor.from_pretrained(model["feature_extractor"]["repo"], subfolder=model["feature_extractor"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **offline_config)
        image_encoder = SiglipVisionModel.from_pretrained(model["image_encoder"]["repo"], subfolder=model["image_encoder"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, **offline_config)
        image_encoder.requires_grad_(False)
        image_encoder.eval()
        sd_models.move_model(image_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=transformer {model["transformer"]}')
        dit_repo = model["transformer"]["repo"]
        load_args, quant_args = model_quant.get_dit_args({}, module='Model', device_map=True)
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(dit_repo, subfolder=model["transformer"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args, **offline_config)
        transformer.high_quality_fp32_output_for_inference = False
        transformer.requires_grad_(False)
        transformer.eval()
        sd_models.move_model(transformer, devices.cpu)

        MODELDATA.sd_model = FramepackHunyuanVideoPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            feature_extractor=feature_extractor,
            image_processor=image_encoder,
            transformer=transformer,
            scheduler=None,
        )
        MODELDATA.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(dit_repo) # pylint: disable=attribute-defined-outside-init
        MODELDATA.sd_model.sd_model_checkpoint = dit_repo # pylint: disable=attribute-defined-outside-init

        MODELDATA.sd_model = model_quant.do_post_load_quant(MODELDATA.sd_model, allow=False)
        t1 = time.time()

        diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING['HunyuanVideoTransformer3DModelPacked'] = lambda model_cls, weights: weights # pylint: disable=protected-access
        shared.log.info(f'FramePack load: model={MODELDATA.sd_model.__class__.__name__} variant="{variant}" type={MODELDATA.sd_model_type} time={t1-t0:.2f}')
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        devices.torch_gc(force=True, reason='load')

    except Exception as e:
        shared.log.error(f'FramePack load: {e}')
        errors.display(e, 'FramePack')
        shared.state.end()
        return None

    shared.state.end()
    return variant


def unload_model():
    sd_models.unload_model_weights()
