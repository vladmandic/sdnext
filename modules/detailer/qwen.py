import time
import json
import transformers
from pydantic import BaseModel, Field
from PIL import Image
from modules import shared, devices, sd_offload_aux, sd_models, model_quant
from modules.detailer import DetailerResult, detailer_opt, get_mask
from modules.logger import log


class BoundingBoxItem(BaseModel):
    label: str = Field(..., description="Label of the detected object")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    box_2d: list[int] = Field(..., min_items=4, max_items=4, description="Bounding box coordinates in 0-1000 normalized format: [ymin, xmin, ymax, xmax]")


class ObjectDetectionOutput(BaseModel):
    response: str = Field(..., description="Reasoning summary: state which requested objects were identified and their location before outputting coordinates.")
    objects: list[BoundingBoxItem]


def template(prompt: str, schema: str, min_confidence: float) -> list[dict]:
    def confidence() -> str:
        conf = max(0.0, min(1.0, float(min_confidence)))
        if conf >= 0.85:
            instruction = "Detect only obvious, fully visible target objects."
        elif conf >= 0.65:
            instruction = "Detect clear targets, ignoring faint or ambiguous cases."
        elif conf >= 0.45:
            instruction = "Detect all distinct targets, including partially covered ones."
        elif conf >= 0.25:
            instruction = "Detect candidate targets, including small, blurry, or occluded ones."
        else:
            instruction = "Detect all possible target candidates, background objects, or fragments."
        return instruction

    instructions = (
        "You are an expert vision assistant for object detection.\n"
        f"INSTRUCTIONS:\n"
        f"1. Scan the image carefully for EACH requested class.\n"
        f"2. Add a BoundingBoxItem to 'objects' for EVERY instance found.\n"
        f"3. In the 'response' field, explicitly list which target classes were found and which were missing.\n"
        f"4. {confidence()}\n"
        f"5. You MUST respond strictly with a valid JSON object matching this schema: \n```json\n{schema}\n```\n\n"
        "6. Do not include any Markdown text outside of the JSON string. Add any text explanations or clarifications inside the 'response' field of the JSON object.\n"
        "7. Bounding box coordinates must be in 0-1000 normalized format: [ymin, xmin, ymax, xmax]."
    )
    return [
        { "role": "system", "content": instructions },
        { "role": "user",
          "content": [
            {"type": "image"}, # injected later using processor
            {"type": "text", "text": prompt},
          ],
        },
    ]


def load(self, model_name: str | None = None) -> tuple[str, transformers.Qwen3VLForConditionalGeneration]: # pylint: disable=unused-argument
    cached = sd_offload_aux.get_aux_model(model_name)
    if cached is not None:
        return model_name, cached
    repo_id = 'Qwen/' + model_name if not model_name.startswith('Qwen/') else model_name
    sd_models.hf_auth_check(repo_id)
    load_kwargs = {
        'pretrained_model_name_or_path': repo_id,
        'cache_dir': shared.opts.hfcache_dir,
        'torch_dtype': devices.dtype,
    }
    quant_args = model_quant.create_config(module='LLM', modules_to_not_convert=['conv1d', 'linear_attn.conv1d'])
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs, **quant_args, attn_implementation="sdpa")
    model = model.eval()
    model.processor: transformers.Qwen3VLProcessor = transformers.Qwen3VLProcessor.from_pretrained(**load_kwargs)
    sd_offload_aux.register_aux(model_name, model)
    if shared.opts.detailer_unload:
        sd_offload_aux.offload_aux(model_name)
    log.info(f'Load: type=Detailer name="{model_name}" cls="{model.__class__.__name__}" processor="{model.processor.__class__.__name__}"')
    return model_name, model


def parse(data: str, image: Image.Image, include_mask: bool = True) -> tuple[str, list[DetailerResult]]:
    results = []
    response = ''
    w, h = image.size
    try:
        clean = data.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        parsed = json.loads(clean.strip())
        response = parsed.get("response", "")
        objects = parsed.get("objects", [])
        for item in objects:
            box_2d = item.get("box_2d", [])
            label = item.get("label", "")
            confidence = float(item.get("confidence", 1.0))
            if len(box_2d) == 4:
                xmin, ymin, xmax, ymax = map(int, box_2d)
                xmin, ymin = int((xmin / 1000.0) * w), int((ymin / 1000.0) * h)
                xmax, ymax = int((xmax / 1000.0) * w), int((ymax / 1000.0) * h)
                box = (xmin, ymin, xmax, ymax)
                mask, cropped = get_mask(box, image, include_mask)
                result = DetailerResult(box=box,
                                        label=label,
                                        score=confidence,
                                        cls=-1,
                                        mask=mask,
                                        item=cropped
                                       )
                # log.trace(f'Detailer box: {result}')
                results.append(result)
    except Exception as err:
        log.error(f'Detailer: failed to parse object detection output: {err}')
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
    if not prompt:
        prompt = 'Detect and locate all objects'

    log.debug(f'Detailer: name="{name}" cls={model.__class__.__name__} prompt="{prompt}" image={image.size} device={device} mask={mask} offload={offload}')
    sd_offload_aux.move_aux_to_gpu(name)

    t0 = time.time()
    schema = json.dumps(ObjectDetectionOutput.model_json_schema(), indent=2)
    messages = template(prompt=prompt, schema=schema, min_confidence=detailer_opt(p, 'detailer_conf'))

    with devices.llm_context():
        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = model.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            min_pixels=128 * 28 * 28,
            max_pixels=1280 * 28 * 28,  # Force 1 MP cap
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        eos_id = model.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or isinstance(eos_id, list):
            eos_id = model.processor.tokenizer.eos_token_id
        pad_id = model.processor.tokenizer.pad_token_id if model.processor.tokenizer.pad_token_id is not None else eos_id
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,         # Deterministic decoding keeps bbox integer tokens strict
            temperature=None,        # Forces argmax token selection
            repetition_penalty=1.03, # Breaks coordinate repetition loops without distorting valid coordinates
            no_repeat_ngram_size=0,  # MUST be 0/None—setting this > 0 corrupts valid repeated bbox coordinates
            eos_token_id=eos_id,     # Prevent premature EOS token stopping
            pad_token_id=pad_id,     # Prevent premature EOS token stopping
        )

    prompt_len = inputs["input_ids"].shape[1]
    output_tokens = generated_ids[0][prompt_len:]
    output_text = model.processor.tokenizer.decode(output_tokens, skip_special_tokens=True)
    t1 = time.time()
    response, results = parse(output_text, image, include_mask=mask)

    log.debug(f'Detailer: name="{name}" tokens={output_tokens.shape[0]} response="{response}" items={len(results)} time={t1-t0:.3f}')

    sd_offload_aux.offload_aux(name)
    return results
