from modules import shared, devices


def load_qwen_nunchaku(repo_id):
    import nunchaku
    nunchaku_precision = nunchaku.utils.get_precision()
    nunchaku_repo = None
    transformer = None
    try:
        from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    except Exception:
        shared.log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" low nunchaku version')
        return None
    if repo_id.lower().endswith('qwen-image'):
        nunchaku_repo = f"nunchaku-tech/nunchaku-qwen-image/svdq-{nunchaku_precision}_r128-qwen-image.safetensors" # r32 vs R128
    else:
        shared.log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" unsupported')
    if nunchaku_repo is not None:
        shared.log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} offload={shared.opts.nunchaku_offload} attention={shared.opts.nunchaku_attention}')
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(nunchaku_repo, offload=shared.opts.nunchaku_offload, torch_dtype=devices.dtype) # pylint: disable=no-member
        transformer.quantization_method = 'SVDQuant'
    return transformer
