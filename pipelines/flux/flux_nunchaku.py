from modules import shared, devices


def load_flux_nunchaku(repo_id):
    import nunchaku
    nunchaku_precision = nunchaku.utils.get_precision()
    nunchaku_repo = None
    transformer = None
    if 'flux.1-kontext' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{nunchaku_precision}_r32-flux.1-kontext-dev.safetensors"
    elif 'flux.1-dev' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-dev/svdq-{nunchaku_precision}_r32-flux.1-dev.safetensors"
    elif 'flux.1-schnell' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
    elif 'flux.1-fill' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/svdq-fp4-flux.1-fill-dev/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
    elif 'flux.1-depth' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/svdq-int4-flux.1-depth-dev/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
    elif 'shuttle' in repo_id.lower():
        nunchaku_repo = f"mit-han-lab/nunchaku-shuttle-jaguar/svdq-{nunchaku_precision}_r32-shuttle-jaguar.safetensors"
    else:
        shared.log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" unsupported')
    if nunchaku_repo is not None:
        shared.log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} offload={shared.opts.nunchaku_offload} attention={shared.opts.nunchaku_attention}')
        transformer = nunchaku.NunchakuFluxTransformer2dModel.from_pretrained(nunchaku_repo, offload=shared.opts.nunchaku_offload, torch_dtype=devices.dtype)
        transformer.quantization_method = 'SVDQuant'
        if shared.opts.nunchaku_attention:
            transformer.set_attention_impl("nunchaku-fp16")
    return transformer
