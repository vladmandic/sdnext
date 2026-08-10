import time
import json
import transformers
from PIL import Image
from modules import shared, devices, sd_offload_aux, sd_models, model_quant
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


def select_florence_task(prompt: str) -> tuple[str, str]:
    # Analyzes a user prompt and automatically determines whether to use <OPEN_VOCABULARY_DETECTION>, <CAPTION_TO_PHRASE_GROUNDING>, or default <OD>
    clean_prompt = prompt.strip()
    if not clean_prompt or clean_prompt == "Detect and locate all objects":
        return "<OD>", "<OD>"
    """
    import re
    items = [i.strip() for i in clean_prompt.split(",") if i.strip()]
    is_class_list = len(items) > 1 or all(len(item.split()) <= 2 for item in items)
    descriptive_keywords = re.search(r'\b(a|an|the|with|wearing|in|on|next to|holding|under|near)\b', clean_prompt, re.IGNORECASE)
    if is_class_list and not descriptive_keywords:
        task = "<OPEN_VOCABULARY_DETECTION>"
        # formatted = f"{task}{', '.join(items)}"
        formatted = f"{task}{'. '.join(items)}"
    else:
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        formatted = f"{task}{clean_prompt}"
    """
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    formatted = f"{task}{clean_prompt}"
    return task, formatted


def load(self, model_name: str | None = None) -> tuple[str, transformers.AutoModelForCausalLM]: # pylint: disable=unused-argument
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached
    repo_id = 'florence-community/' + model_name if '/' not in model_name else model_name
    sd_models.hf_auth_check(repo_id)

    orig_get_imports = transformers.dynamic_module_utils.get_imports

    def hijack_get_imports(f):
        R = orig_get_imports(f)
        if "flash_attn" in R:
            R.remove("flash_attn")  # flash_attn is optional
        return R

    transformers.dynamic_module_utils.get_imports = hijack_get_imports
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'torch_dtype': devices.dtype,
        'trust_remote_code': True,
    }
    quant_args = model_quant.create_config(module='LLM', modules_to_not_convert=['conv1d'])
    model = transformers.Florence2ForConditionalGeneration.from_pretrained(
        **load_kwargs,
        **quant_args,
        attn_implementation="sdpa"
    )
    model = model.eval()
    model.processor = transformers.AutoProcessor.from_pretrained(**load_kwargs, max_pixels=1024*1024)

    transformers.dynamic_module_utils.get_imports = orig_get_imports
    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)

    log.info(f'Load: type=Detailer name="{model_name}" cls="{model.__class__.__name__}" processor="{model.processor.__class__.__name__}"')
    return model_name, model


def parse(data: dict | str, image: Image.Image, include_mask: bool = True) -> tuple[str, list[DetailerResult]]:
    results = []
    response = ""
    w, h = image.size
    try:
        parsed_data = {}
        if isinstance(data, str):
            clean = data.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]
            parsed_data = json.loads(clean.strip())
        elif isinstance(data, dict):
            parsed_data = data
        detection_data = None
        for key in ["<OPEN_VOCABULARY_DETECTION>", "<CAPTION_TO_PHRASE_GROUNDING>", "<OD>"]:
            if key in parsed_data:
                detection_data = parsed_data[key]
                response = f"task '{key}' executed successfully."
                break
        if detection_data:
            bboxes = detection_data.get("bboxes", [])
            # Fix key mismatch: Florence-2 returns 'bboxes_labels' for OPEN_VOCABULARY_DETECTION and CAPTION_TO_PHRASE_GROUNDING
            labels = detection_data.get("labels") or detection_data.get("bboxes_labels") or ["object"] * len(bboxes)

            for box, label in zip(bboxes, labels):
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = map(int, box)
                    box = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
                    mask, cropped = get_mask(box, image, include_mask=include_mask)
                    result = DetailerResult(box=box, label=label, score=1.0, cls=-1, mask=mask, item=cropped)
                    log.trace(f'Detailer box: {result}')
                    results.append(result)
    except Exception as err:
        log.error(f'Detailer: failed to parse detection output: {err}')
        log.error(f'Detailer: raw output: {data}')

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

    # Dynamic task routing (<OD>, <OPEN_VOCABULARY_DETECTION>, or <CAPTION_TO_PHRASE_GROUNDING>)
    task_prompt, text_input = select_florence_task(prompt)

    log.debug(f'Detailer: name="{name}" cls={model.__class__.__name__} prompt="{text_input}" image={image.size} device={device} mask={mask} offload={offload}')
    sd_offload_aux.move_aux_to_gpu(name)

    t0 = time.time()
    with devices.llm_context():
        inputs = model.processor(
            text=text_input,
            images=image,
            return_tensors="pt"
        ).to(model.device, dtype=devices.dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3, # beams=3 improves grounding recall on small objects
            early_stopping=False,
        )
    generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_output = model.processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    t1 = time.time()
    response, results = parse(parsed_output, image, include_mask=mask)

    token_count = generated_ids.shape[1] if hasattr(generated_ids, 'shape') else 0
    log.debug(f'Detailer: name="{name}" tokens={token_count} response="{response}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
