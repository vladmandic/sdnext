from modules.logger import log
from modules import shared, devices


def load_qwen_nunchaku(repo_id, subfolder=None):
    import nunchaku
    nunchaku_precision = nunchaku.utils.get_precision()
    nunchaku_repo = None
    transformer = None
    four_step = subfolder is not None and '4step' in subfolder
    try:
        from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    except Exception:
        log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" low nunchaku version')
        return None
    if 'pruning' in repo_id.lower() or 'distill' in repo_id.lower():
        return None
    elif repo_id.lower().endswith('qwen-image'):
        nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image/svdq-{nunchaku_precision}_r128-qwen-image.safetensors"
    elif repo_id.lower().endswith('qwen-lightning'):
        if four_step:
            nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image/svdq-{nunchaku_precision}_r128-qwen-image-lightningv1.0-4steps.safetensors"
        else:
            nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image/svdq-{nunchaku_precision}_r128-qwen-image-lightningv1.1-8steps.safetensors"
    elif repo_id.lower().endswith('qwen-image-edit-2509'):
        nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image-edit-2509/svdq-{nunchaku_precision}_r128-qwen-image-edit-2509.safetensors"
    elif repo_id.lower().endswith('qwen-image-edit'):
        nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image-edit/svdq-{nunchaku_precision}_r128-qwen-image-edit.safetensors"
    elif repo_id.lower().endswith('qwen-lightning-edit'):
        if four_step:
            nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image-edit/svdq-{nunchaku_precision}_r128-qwen-image-edit-lightningv1.0-4steps.safetensors"
        else:
            nunchaku_repo = f"nunchaku-ai/nunchaku-qwen-image-edit/svdq-{nunchaku_precision}_r128-qwen-image-edit-lightningv1.0-8steps.safetensors"
    else:
        log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" unsupported')
    if nunchaku_repo is not None:
        log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} offload={shared.opts.nunchaku_offload} attention={shared.opts.nunchaku_attention}')
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            nunchaku_repo,
            offload=shared.opts.nunchaku_offload,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        ) # pylint: disable=no-member
        transformer.quantization_method = 'SVDQuant'
    return transformer
