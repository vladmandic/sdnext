import time
import re
import torch
import transformers
from PIL import Image
from modules import shared, devices, sd_offload_aux, sd_models,model_quant
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


def format_rex_prompt(prompt: str) -> str:
    clean_prompt = prompt.strip()
    if not clean_prompt or clean_prompt.lower() == "detect and locate all objects":
        return "<|grounding|>Locate all objects in the image."
    if clean_prompt.startswith("<|") and "|>" in clean_prompt:
        return clean_prompt
    return f"<|grounding|>{clean_prompt}"


def load(self, model_name: str | None = None) -> tuple[str, transformers.AutoModelForCausalLM]: # pylint: disable=unused-argument
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached

    repo_id = 'IDEA-Research/' + model_name if '/' not in model_name else model_name
    sd_models.hf_auth_check(repo_id)
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'torch_dtype': devices.dtype,
        'trust_remote_code': True,
    }
    quant_args = model_quant.create_config(module='LLM', modules_to_not_convert=['conv1d', 'linear_attn.conv1d', 'embed_tokens', 'lm_head'])
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        **load_kwargs,
        **quant_args,
        attn_implementation="sdpa"
    )
    model = model.eval()
    model.processor = transformers.AutoProcessor.from_pretrained(
        repo_id,
        cache_dir=shared.opts.hfcache_dir,
        trust_remote_code=True
    )

    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)

    log.info(f'Load: type=Detailer name="{model_name}" cls="{model.__class__.__name__}" processor="{model.processor.__class__.__name__}"')
    return model_name, model


def parse(raw_output: str, image: Image.Image, include_mask: bool = True) -> tuple[str, list[DetailerResult]]:
    results = []
    w, h = image.size
    bboxes = []
    labels = []
    box_pattern = re.compile(r'(?:<box>|\(|\[)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:</box>|\)|\])\s*([^<\n,]+)?')
    matches = box_pattern.findall(raw_output)
    for match in matches:
        if len(match) >= 4:
            coords = [int(match[i]) for i in range(4)]
            label = match[4].strip() if len(match) > 4 and match[4].strip() else "object"
            ymin, xmin, ymax, xmax = coords
            if max(coords) <= 1000:
                xmin = int((xmin / 1000.0) * w)
                ymin = int((ymin / 1000.0) * h)
                xmax = int((xmax / 1000.0) * w)
                ymax = int((ymax / 1000.0) * h)
            bboxes.append((xmin, ymin, xmax, ymax))
            labels.append(label)
    for box, label in zip(bboxes, labels):
        if len(box) == 4:
            xmin, ymin, xmax, ymax = map(int, box)
            box = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
            if box[2] > box[0] and box[3] > box[1]:
                mask, cropped = get_mask(box, image, include_mask=include_mask)
                result = DetailerResult(box=box, label=label, score=1.0, cls=-1, mask=mask, item=cropped)
                log.trace(f'Detailer box: {result}')
                results.append(result)
    return raw_output, results


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

    prompt = detailer_opt(p, 'detailer_classes') or ''
    text_input = format_rex_prompt(prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_input},
            ],
        }
    ]

    log.debug(f'Detailer: name="{name}" cls={model.__class__.__name__} prompt="{text_input}" image={image.size} device={device} mask={mask} offload={offload}')
    sd_offload_aux.move_aux_to_gpu(name)

    t0 = time.time()
    with devices.llm_context():
        # Use processor directly with images and text to automatically build image tokens and grid thw
        text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = model.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        target_device = model.device
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device=target_device)
                if torch.is_floating_point(v):
                    v = v.to(dtype=devices.dtype)
                prepared_inputs[k] = v
            else:
                prepared_inputs[k] = v
        generated_ids = model.generate(
            **prepared_inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )

    input_len = prepared_inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    generated_text = model.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    t1 = time.time()
    response, results = parse(generated_text, image, include_mask=mask)

    token_count = generated_ids.shape[1]
    log.debug(f'Detailer: name="{name}" tokens={token_count} response="{response}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
