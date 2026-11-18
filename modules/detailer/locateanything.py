import re
import time
from PIL import Image
import transformers
from modules import shared, devices, sd_offload_aux
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


repo_id = 'SD-Next/LocateAnything-3B'
processor = None


def load(self, model_name: str | None = None): # pylint: disable=unused-argument
    global processor # pylint: disable=global-statement
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'trust_remote_code': True,
    }
    if shared.opts.offline_mode:
        load_kwargs['local_files_only'] = True
    processor = transformers.AutoProcessor.from_pretrained(**load_kwargs)
    model = transformers.AutoModel.from_pretrained(**load_kwargs, torch_dtype=devices.dtype)
    model = model.eval()
    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)
    log.info(f'Detailer model="{model_name}" cls={model.__class__.__name__} loaded')
    return model_name, model


def parse(response: str, image: Image.Image, include_mask: bool = True) -> list[DetailerResult]:
    results = []
    label = 'object'
    for match in re.finditer(r'<ref>(.*?)</ref>|<box>(.*?)</box>', response, re.DOTALL):
        if match.group(1) is not None:
            label = match.group(1).strip() or 'object'
            continue
        box = re.findall(r'<(\d+)>', match.group(2))
        if len(box) != 4:
            continue
        x1, y1 = int(image.width * int(box[0]) / 1000), int(image.height * int(box[1]) / 1000)
        x2, y2 = int(image.width * int(box[2]) / 1000), int(image.height * int(box[3]) / 1000)
        box = (x1, y1, x2, y2)
        mask, cropped = get_mask(box, image, include_mask)
        results.append(DetailerResult(cls=-1, label=label, score=1.0, box=box, mask=mask, item=cropped))
    return results


def predict(
        self,
        name,
        image: Image.Image,
        device = devices.device,
        mask: bool = True,
        offload: bool | None = None,
        p = None,
    ) -> list[DetailerResult]:
    if offload is None:
        offload = shared.opts.detailer_unload
    if image is None:
        return []
    cached = sd_offload_aux.get_aux_model(name)
    if cached is None:
        name, model = load(self, name)
    else:
        model = cached
    if model is None:
        return []
    sd_offload_aux.move_aux_to_gpu(name)
    log.info(f'Detailer cls="{model.__class__.__name__}" image={image} device={device} mask={mask} offload={offload}')
    prompt = detailer_opt(p, 'detailer_classes') or ''
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]

    t0 = time.time()
    with devices.llm_context():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    t1 = time.time()

    try:
        results = parse(response, image, include_mask=mask)
    except Exception as e:
        log.error(f'Detailer parse: response="{response}" {e}')
        results = []
    log.debug(f'Detailer: name="{name}" prompt="{prompt}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
