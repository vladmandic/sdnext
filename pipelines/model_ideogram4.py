import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant
from modules.logger import log
from pipelines import generic


def pin_transformers(transformer, unconditional_transformer) -> bool:
    """Keep both transformers resident under balanced offload when they fit the budget.
    Every denoise step runs both transformers, so balanced offload ping-pongs them across
    PCIe each step. ``offload_never`` makes ``offload_allowed`` skip the per-step pre-sweep so
    they stay resident, but only when both fit ``gpu_memory * max watermark`` (which leaves the
    watermark headroom for activations); otherwise the normal offload path is kept.
    """
    if shared.opts.diffusers_offload_mode != 'balanced' or shared.gpu_memory <= 0 or not shared.opts.model_ideogram4_pin:
        return False
    if transformer is None or unconditional_transformer is None:
        return False
    size_gb = sum(p.numel() * p.element_size() for m in (transformer, unconditional_transformer) for p in m.parameters()) / (1024 ** 3)
    budget_gb = shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory
    fits = size_gb <= budget_gb
    if fits:
        transformer.offload_never = True
        unconditional_transformer.offload_never = True
    log.debug(f'Load model: type=Ideogram4 offload=balanced transformers={size_gb:.2f} budget={budget_gb:.2f} action={"pin" if fits else "default"}')
    return fits


def load_ideogram4(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Ideogram4 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args} conditional={shared.opts.model_ideogram4_enable_cg} enhance={shared.opts.model_ideogram4_enable_pe} pin={shared.opts.model_ideogram4_pin}')

    from pipelines.ideogram.ideogram4 import Ideogram4Pipeline
    generic.set_pipeline('Ideogram4', Ideogram4Pipeline)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    transformer_cls = diffusers.Ideogram4Transformer2DModel
    transformer = generic.load_transformer(repo_id, cls_name=transformer_cls, subfolder="transformer", load_config=diffusers_load_config)
    if shared.opts.model_ideogram4_enable_cg:
        unconditional_transformer = generic.load_transformer(repo_id, cls_name=transformer_cls, subfolder="unconditional_transformer", load_config=diffusers_load_config)
    else:
        unconditional_transformer = None
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3VLModel, load_config=diffusers_load_config)

    components = {
        'transformer': transformer,
        'unconditional_transformer': unconditional_transformer,
        'text_encoder': text_encoder,
    }

    prompt_enhancer_head = None
    if shared.opts.model_ideogram4_enable_pe:
        enhancer_repo_id = "diffusers/qwen3-vl-8b-instruct-lm-head"
        enhancer_cls = diffusers.Ideogram4PromptEnhancerHead
        log.debug(f'Load model: enhancer="{enhancer_repo_id}" cls={enhancer_cls.__name__}')
        prompt_enhancer_head = enhancer_cls.from_pretrained(
            enhancer_repo_id,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir
        )
        components['prompt_enhancer_head'] = prompt_enhancer_head

    pin_transformers(transformer, unconditional_transformer)
    pipe = Ideogram4Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        **components,
        **load_args,
    )

    pipe.task_args = {
        'output_type': 'pil',
        'prompt_upsampling': shared.opts.model_ideogram4_enable_pe,
    }

    del transformer, unconditional_transformer, text_encoder, prompt_enhancer_head
    devices.torch_gc(force=True, reason='load')

    from pipelines.ideogram.patch_qwen import hijack_qwen3vl
    hijack_qwen3vl()

    return pipe
