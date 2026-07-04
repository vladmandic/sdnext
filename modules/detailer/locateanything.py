from PIL import Image
from modules import shared, devices
from modules.detailer import DetailerResult, detailer_opt
from modules.logger import log


repo_id = 'nvidia/LocateAnything-3B'
processor = None
tokenizer = None


def dependencies():
    from installer import install
    install('decord')


def load(model_name: str | None = None):
    import transformers
    global tokenizer, processor # pylint: disable=global-statement
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'trust_remote_code': True,
    }
    dependencies()
    tokenizer = transformers.AutoTokenizer.from_pretrained(**load_kwargs)
    processor = transformers.AutoProcessor.from_pretrained(**load_kwargs)
    model = transformers.AutoModel.from_pretrained(**load_kwargs, torch_dtype=devices.dtype)
    model = model.to(devices.device).eval()
    if shared.opts.detailer_unload:
        model.to(devices.cpu)
    log.info(f'Detailer model="{model_name}" cls={model.__class__.__name__} loaded')
    return model_name, model


def predict(
        self,
        model,
        image: Image.Image,
        device = devices.device,
        mask: bool = True,
        offload: bool | None = None,
        p = None,
    ) -> list[DetailerResult]:
    if offload is None:
        offload = shared.opts.detailer_unload
    log.info(f'Detailer cls="{model.__class__.__name__}" image={image} device={device} mask={mask} offload={offload}')
    result = []
    if isinstance(model, str):
        cached = self.models.get(model, None)
        if cached is None:
            _, model = self.load(model)
        else:
            model = cached
    if model is None:
        return result
    model = model.to(device)
    prompt = detailer_opt(p, 'detailer_classes') or ''
    log.debug(f'Detailer prompt="{prompt}"')

    # TODO locateanything: implement when compatible with transformers=5

    if offload:
        model.to(devices.cpu)
    return result
