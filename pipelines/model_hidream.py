import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_llama(diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    llama_repo = shared.opts.model_h1_llama_repo if shared.opts.model_h1_llama_repo != 'Default' else 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    log.debug(f'Load model: type=HiDream te4="{llama_repo}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
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
    return text_encoder_4, tokenizer_4


def load_hidream(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=HiDream repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.HiDreamImageTransformer2DModel, load_config=diffusers_load_config, subfolder="transformer")
    text_encoder_3 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_3")
    text_encoder_4, tokenizer_4 = load_llama(diffusers_load_config)

    if shared.opts.teacache_enabled:
        from modules import teacache
        log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.HiDreamImageTransformer2DModel.__name__}')
        diffusers.HiDreamImageTransformer2DModel.forward = teacache.teacache_hidream_forward # patch must be done before transformer is loaded

    if 'I1' in repo_id:
        cls = diffusers.HiDreamImagePipeline
    elif 'E1' in repo_id:
        from pipelines.hidream.pipeline_hidream_image_editing import HiDreamImageEditingPipeline
        cls = HiDreamImageEditingPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["hidream-e1"] = diffusers.HiDreamImagePipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["hidream-e1"] = HiDreamImageEditingPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["hidream-e1"] = HiDreamImageEditingPipeline
        if transformer and 'E1-1' in repo_id:
            transformer.max_seq = 8192
        elif transformer and 'E1' in repo_id:
            transformer.max_seq = 4608
    else:
        log.error(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" not recognized')
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

    del text_encoder_3
    del text_encoder_4
    del tokenizer_4
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
