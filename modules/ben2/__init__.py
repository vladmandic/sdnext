model = None


def remove(image, refine: bool = True):
    global model # pylint: disable=global-statement
    from modules import shared, devices

    if model is None:
        from huggingface_hub import hf_hub_download
        from .ben2_model import BEN_Base
        model = BEN_Base()
        model_file = hf_hub_download(
            repo_id='PramaLLC/BEN2',
            filename='BEN2_Base.pth',
            cache_dir=shared.opts.hfcache_dir)
        model.loadcheckpoints(model_file)
        model = model.to(device=devices.device, dtype=devices.dtype).eval()

    model = model.to(device=devices.device)
    foreground = model.inference(image, refine_foreground=refine)
    model = model.to(device=devices.cpu)
    if foreground is None:
        return image
    return foreground
