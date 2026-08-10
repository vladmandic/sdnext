import time
import transformers
from PIL import Image
from modules import shared, devices, sd_models, sd_offload_aux
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


def load(self, model_name: str | None = None) -> tuple[str, transformers.Sam3Model]: # pylint: disable=unused-argument
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached
    repo_id = model_name.lower().replace('-', '/')
    sd_models.hf_auth_check(repo_id, force=True)
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'torch_dtype': devices.dtype,
    }
    model = transformers.Sam3Model.from_pretrained(**load_kwargs)
    model = model.eval()
    model.processor = transformers.Sam3Processor.from_pretrained(**load_kwargs)
    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)
    log.info(f'Load: type=Detailer name="{model_name}" cls="{model.__class__.__name__}" processor="{model.processor.__class__.__name__}"')
    return model_name, model


def predict(
    self,
    name: str,
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
    prompt = detailer_opt(p, 'detailer_classes') or ''
    if not prompt:
        prompt = 'object'

    log.debug(f'Detailer: name="{name}" cls={model.__class__.__name__} prompt="{prompt}" image={image.size} device={device} mask={mask} offload={offload}')
    sd_offload_aux.move_aux_to_gpu(name)

    t0 = time.time()
    results = []

    with devices.llm_context():
        inputs = model.processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        target_sizes = inputs.get("original_sizes").tolist() if "original_sizes" in inputs else [image.size[::-1]]
        results_list = model.processor.post_process_instance_segmentation(
            outputs,
            threshold=detailer_opt(p, 'detailer_conf') or 0.3,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )

    w, h = image.size
    if results_list and len(results_list) > 0:
        res = results_list[0]
        boxes = res.get("boxes", [])
        scores = res.get("scores", [])
        labels = res.get("labels", [])
        masks = res.get("masks", []) if mask else [None] * len(boxes)
        for i, box_tensor in enumerate(boxes):
            box_coords = box_tensor.tolist()
            xmin, ymin, xmax, ymax = map(int, box_coords)
            box = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
            score = float(scores[i].item()) if i < len(scores) else 1.0
            label = str(labels[i].item()) if i < len(labels) else prompt
            masked, cropped = get_mask(box, image, include_mask=mask)
            if mask and detailer_opt(p, 'detailer_segmentation') and (i < len(masks) and masks[i] is not None):
                masked = Image.fromarray(masks[i].detach().cpu().numpy().astype('uint8') * 255)
            cropped = image.crop(box)
            result = DetailerResult(
                box=box,
                label=label,
                score=score,
                cls=-1,
                mask=masked,
                item=cropped
            )
            results.append(result)

    t1 = time.time()
    log.debug(f'Detailer: name="{name}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
