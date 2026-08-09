import time
import re
import torch
import transformers
from PIL import Image
from modules import shared, devices, sd_offload_aux, model_quant
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


def format_grounding_dino_prompt(prompt: str) -> str:
    # Formats user input into Grounding DINO query format. Grounding DINO requires lowercase text separated by periods and ending with a period.
    clean_prompt = prompt.strip().lower()
    if not clean_prompt or clean_prompt == "detect and locate all objects":
        return "object."
    items = [i.strip() for i in re.split(r'[,.]', clean_prompt) if i.strip()]
    if not items:
        return "object."
    return ". ".join(items) + "."


def load(self, model_name: str | None = None) -> tuple[str, transformers.AutoModelForZeroShotObjectDetection]: # pylint: disable=unused-argument
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached
    repo_id = 'IDEA-Research/' + model_name.lower() if '/' not in model_name else model_name
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'torch_dtype': devices.dtype,
    }
    quant_args = model_quant.create_config(module='LLM', modules_to_not_convert=['conv1d'])
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(**load_kwargs, **quant_args)
    model = model.eval()
    model.processor = transformers.AutoProcessor.from_pretrained(**load_kwargs)

    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)

    log.info(f'Load: type=Detailer name="{model_name}" cls="{model.__class__.__name__}" processor="{model.processor.__class__.__name__}"')
    return model_name, model


def parse(data: dict, image: Image.Image, include_mask: bool = True) -> tuple[str, list[DetailerResult]]:
    results = []
    w, h = image.size
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])
    labels = data.get("labels", [])
    for box, score, label in zip(boxes, scores, labels):
        if len(box) == 4:
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            xmin, ymin, xmax, ymax = map(int, box)
            box = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
            mask, cropped = get_mask(box, image, include_mask=include_mask)
            result = DetailerResult(box=box,
                                    label=str(label),
                                    score=float(score),
                                    cls=-1,
                                    mask=mask,
                                    item=cropped
                                   )
            # log.trace(f'Detailer box: {result}')
            results.append(result)
    response = f"Grounding DINO detected {len(results)} objects."
    return response, results


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
    text_input = format_grounding_dino_prompt(prompt)
    threshold = detailer_opt(p, 'detailer_conf')

    log.debug(f'Detailer: name="{name}" cls={model.__class__.__name__} prompt="{text_input}" image={image.size} device={device} mask={mask} offload={offload} threshold={threshold}')
    sd_offload_aux.move_aux_to_gpu(name)

    t0 = time.time()
    with devices.llm_context():
        inputs = model.processor(
            images=image,
            text=text_input,
            return_tensors="pt"
        ).to(model.device, dtype=devices.dtype)
        with torch.autocast(device_type=model.device.type, dtype=devices.dtype):
            outputs = model(**inputs)

    parsed_output = model.processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(image.height, image.width)]
    )[0]

    t1 = time.time()
    response, results = parse(parsed_output, image, include_mask=mask)

    log.debug(f'Detailer: name="{name}" response="{response}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
