import os
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te


def load_transformer(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    fn = None

    if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
        from modules import sd_unet
        if shared.opts.sd_unet not in list(sd_unet.unet_dict):
            shared.log.error(f'Load module: type=Transformer not found: {shared.opts.sd_unet}')
            return None
        fn = sd_unet.unet_dict[shared.opts.sd_unet] if os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]) else None

    if fn is not None and 'gguf' in fn.lower():
        shared.log.error('Load model: type=HiDream format="gguf" unsupported')
        transformer = None
        # from modules import ggml
        # transformer = ggml.load_gguf(fn, cls=diffusers.HiDreamImageTransformer2DModel, compute_dtype=devices.dtype)
    elif fn is not None and 'safetensors' in fn.lower():
        shared.log.debug(f'Load model: type=HiDream transformer="{repo_id}" offload={shared.opts.diffusers_offload_mode} quant="{model_quant.get_quant(repo_id)}" args={load_args}')
        transformer = diffusers.HiDreamImageTransformer2DModel.from_single_file(fn, cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args)
    # elif model_quant.check_nunchaku('Model'):
    #     shared.log.error(f'Load model: type=HiDream transformer="{repo_id}" quant="Nunchaku" unsupported')
    #     transformer = None
    else:
        shared.log.debug(f'Load model: type=HiDream transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
        transformer = diffusers.HiDreamImageTransformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoders(repo_id, diffusers_load_config={}):
    if repo_id == 'HiDream-ai/HiDream-E1-Full':
        repo_id = 'HiDream-ai/HiDream-I1-Full' # use I1 for t5 and llm
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=HiDream te3="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder_3",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder_3 is not None:
        sd_models.move_model(text_encoder_3, devices.cpu)

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    llama_repo = shared.opts.model_h1_llama_repo if shared.opts.model_h1_llama_repo != 'Default' else 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    shared.log.debug(f'Load model: type=HiDream te4="{llama_repo}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')

    sd_models.hf_auth_check(llama_repo)
    text_encoder_4 = transformers.LlamaForCausalLM.from_pretrained(
        llama_repo,
        output_hidden_states=True,
        output_attentions=True,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    tokenizer_4 = transformers.PreTrainedTokenizerFast.from_pretrained(
        llama_repo,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder_4 is not None:
        sd_models.move_model(text_encoder_4, devices.cpu)
    return text_encoder_3, text_encoder_4, tokenizer_4


def load_hidream(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    transformer = load_transformer(repo_id, diffusers_load_config)
    text_encoder_3, text_encoder_4, tokenizer_4 = load_text_encoders(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.HiDreamImageTransformer2DModel.__name__}')
        diffusers.HiDreamImageTransformer2DModel.forward = teacache.teacache_hidream_forward # patch must be done before transformer is loaded

    if 'I1' in repo_id:
        cls = diffusers.HiDreamImagePipeline
    elif 'E1' in repo_id:
        from pipelines.hidream.pipeline_hidream_image_editing import HiDreamImageEditingPipeline
        cls = HiDreamImageEditingPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["hidream-e1"] = diffusers.HiDreamImagePipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["hidream-e1"] = HiDreamImageEditingPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["hidream-e1"] = HiDreamImageEditingPipeline
    else:
        shared.log.error(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" not recognized')
        return False

    pipe = cls.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_3=text_encoder_3,
        text_encoder_4=text_encoder_4,
        tokenizer_4=tokenizer_4,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    sd_hijack_te.init_hijack(pipe)
    del text_encoder_3
    del text_encoder_4
    del tokenizer_4
    del transformer

    devices.torch_gc()
    return pipe
