import io
import os
import time
import json
import base64
import torch
import transformers
import transformers.dynamic_module_utils
from PIL import Image
from modules import shared, devices, errors, model_quant, sd_models, sd_models_compile


processor = None
model = None
loaded: str = None
quant_args = {}
vlm_models = {
    "Microsoft Florence 2 Base": "microsoft/Florence-2-base", # 0.5GB
    "Microsoft Florence 2 Large": "microsoft/Florence-2-large", # 1.5GB
    "MiaoshouAI PromptGen 1.5 Base": "MiaoshouAI/Florence-2-base-PromptGen-v1.5@c06a5f02cc6071a5d65ee5d294cf3732d3097540", # 1.1GB
    "MiaoshouAI PromptGen 1.5 Large": "MiaoshouAI/Florence-2-large-PromptGen-v1.5@28a42440e39c9c32b83f7ae74ec2b3d1540404f0", # 3.3GB
    "MiaoshouAI PromptGen 2.0 Base": "MiaoshouAI/Florence-2-base-PromptGen-v2.0", # 1.1GB
    "MiaoshouAI PromptGen 2.0 Large": "MiaoshouAI/Florence-2-large-PromptGen-v2.0", # 3.3GB
    "CogFlorence 2.0 Large": "thwri/CogFlorence-2-Large-Freeze", # 1.6GB
    "CogFlorence 2.2 Large": "thwri/CogFlorence-2.2-Large", # 1.6GB
    "Moondream 2": "vikhyatk/moondream2", # 3.7GB
    "Google Gemma 3 4B": "google/gemma-3-4b-it",
    "Google Gemma 3n E2B": "google/gemma-3n-E2B-it", # 1.5GB
    "Google Gemma 3n E4B": "google/gemma-3n-E4B-it", # 1.5GB
    "Google Pix Textcaps": "google/pix2struct-textcaps-base", # 1.1GB
    "Google PaliGemma 2 3B": "google/paligemma2-3b-pt-224",
    "Alibaba Qwen 2.0 VL 2B": "Qwen/Qwen2-VL-2B-Instruct",
    "Alibaba Qwen 2.5 Omni 3B": "Qwen/Qwen2.5-Omni-3B",
    "Alibaba Qwen 2.5 VL 4B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Huggingface Smol VL2 0.5B": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Huggingface Smol VL2 2B": "HuggingFaceTB/SmolVLM-Instruct",
    "Salesforce BLIP Base": "Salesforce/blip-vqa-base", # 1.5GB
    "Salesforce BLIP Large": "Salesforce/blip-vqa-capfilt-large", # 1.5GB
    "Microsoft GIT TextCaps Base": "microsoft/git-base-textcaps", # 0.7GB
    "Microsoft GIT VQA Base": "microsoft/git-base-vqav2", # 0.7GB
    "Microsoft GIT VQA Large": "microsoft/git-large-vqav2", # 1.6GB
    "ToriiGate 0.4 2B": "Minthy/ToriiGate-v0.4-2B",
    "ToriiGate 0.4 7B": "Minthy/ToriiGate-v0.4-7B",
    "ViLT Base": "dandelin/vilt-b32-finetuned-vqa", # 0.5GB
    "JoyCaption Alpha": "fancyfeast/llama-joycaption-alpha-two-hf-llava", # 17.4GB
    "JoyCaption Beta": "fancyfeast/llama-joycaption-beta-one-hf-llava", # 17.4GB
    "JoyTag": "fancyfeast/joytag", # 0.7GB
    "AIDC Ovis2 1B": "AIDC-AI/Ovis2-1B",
    "AIDC Ovis2 2B": "AIDC-AI/Ovis2-2B",
    "AIDC Ovis2 4B": "AIDC-AI/Ovis2-4B",
    "ByteDance Sa2VA 1B": "ByteDance/Sa2VA-1B",
    "ByteDance Sa2VA 4B": "ByteDance/Sa2VA-4B",
    # "OpenGVLab InternVL 2.5 1B": "OpenGVLab/InternVL2_5-1B"
    # "DeepSeek VL2 Tiny": "deepseek-ai/deepseek-vl2-tiny", # broken
    # "nVidia Eagle 2 1B": "nvidia/Eagle2-1B", # not compatible with latest transformers
}
vlm_system = 'You are image captioning expert, creative, unbiased and uncensored.'
vlm_prompts = [
    '<CAPTION>',
    '<DETAILED_CAPTION>',
    '<MORE_DETAILED_CAPTION>',
    '<CAPTION_TO_PHRASE_GROUNDING>',
    '<OD>',
    '<DENSE_REGION_CAPTION>',
    '<REGION_PROPOSAL>',
    '<OCR>',
    '<OCR_WITH_REGION>',
    '<ANALYZE>',
    '<GENERATE_TAGS>',
    '<MIXED_CAPTION>',
    '<MIXED_CAPTION_PLUS>',
]


def b64(image):
    if image is None:
        return ''
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def clean(response, question):
    strip = ['---', '\r', '\t', '**', '"', '“', '”', 'Assistant:', 'Caption:', '<|im_end|>', '<pad>']
    if isinstance(response, dict):
        if 'task' in response:
            response = response['task']
        if 'answer' in response:
            response = response['answer']
        response = json.dumps(response)
    if isinstance(response, list):
        response = response[0]
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    if question in response:
        response = response.split(question, 1)[1]
    while any(s in response for s in strip):
        for s in strip:
            response = response.replace(s, '')
    response = response.replace('\n\n', '\n').replace('  ', ' ').replace('*  ', '- ').strip()
    return response


def get_kwargs():
    kwargs = {
        'max_new_tokens': shared.opts.interrogate_vlm_max_length,
        'do_sample': shared.opts.interrogate_vlm_do_sample,
    }
    if shared.opts.interrogate_vlm_num_beams > 0:
        kwargs['num_beams'] = shared.opts.interrogate_vlm_num_beams
    if shared.opts.interrogate_vlm_temperature > 0:
        kwargs['temperature'] = shared.opts.interrogate_vlm_temperature
    if shared.opts.interrogate_vlm_top_k > 0:
        kwargs['top_k'] = shared.opts.interrogate_vlm_top_k
    if shared.opts.interrogate_vlm_top_p > 0:
        kwargs['top_p'] = shared.opts.interrogate_vlm_top_p
    return kwargs


def qwen(question: str, image: Image.Image, repo: str = None, system_prompt: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        if '2.5' in repo:
            cls_name = transformers.Qwen2_5_VLForConditionalGeneration
        else:
            cls_name = transformers.Qwen2VLForConditionalGeneration
        model = cls_name.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            **quant_args,
        )
        processor = transformers.AutoProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        if 'LLM' in shared.opts.cuda_compile:
            model = sd_models_compile.compile_torch(model)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    system_prompt = system_prompt or shared.opts.interrogate_vlm_system
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": b64(image)},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(devices.device, devices.dtype)
    output_ids = model.generate(
        **inputs,
        **get_kwargs(),
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response


def gemma(question: str, image: Image.Image, repo: str = None, system_prompt: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        if '3n' in repo:
            cls = transformers.Gemma3nForConditionalGeneration # pylint: disable=no-member
        else:
            cls = transformers.Gemma3ForConditionalGeneration
        model = cls.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            **quant_args,
        )
        if 'LLM' in shared.opts.cuda_compile:
            model = sd_models_compile.compile_torch(model)
        processor = transformers.AutoProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    system_prompt = system_prompt or shared.opts.interrogate_vlm_system

    system_content = []
    if system_prompt is not None and len(system_prompt) > 4:
        system_content.append({"type": "text", "text": system_prompt})

    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": b64(image)})
    if question is not None and len(question) > 4:
        user_content.append({"type": "text", "text": question})
    conversation = [
        { "role": "system", "content": system_content},
        { "role": "user", "content": user_content },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device=devices.device, dtype=devices.dtype)
    input_len = inputs["input_ids"].shape[-1]
    with devices.inference_context():
        generation = model.generate(
            **inputs,
            **get_kwargs(),
        )
        generation = generation[0][input_len:]
    response = processor.decode(generation, skip_special_tokens=True)
    return response


def paligemma(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        processor = transformers.PaliGemmaProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        model = None
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            repo,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    model_inputs = processor(text=question, images=image, return_tensors="pt").to(devices.device, devices.dtype)
    input_len = model_inputs["input_ids"].shape[-1]
    with devices.inference_context():
        generation = model.generate(
            **model_inputs,
            **get_kwargs(),
        )
    generation = generation[0][input_len:]
    response = processor.decode(generation, skip_special_tokens=True)
    return response


def ovis(question: str, image: Image.Image, repo: str = None):
    try:
        import flash_attn # pylint: disable=unused-import
    except Exception:
        shared.log.error(f'Interrogate: vlm="{repo}" flash-attn is not available')
        return ''
    global model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            multimodal_max_length=32768,
            trust_remote_code=True,
            cache_dir=shared.opts.hfcache_dir,
        )
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    max_partition = 9
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    question = f'<image>\n{question}'
    _prompt, input_ids, pixel_values = model.preprocess_inputs(question, [image], max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    with devices.inference_context():
        output_ids = model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True,
            **get_kwargs())
        response = text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'Output:\n{response}')
    return response


def smol(question: str, image: Image.Image, repo: str = None, system_prompt: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            repo,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
            _attn_implementation="eager",
            **quant_args,
            )
        processor = transformers.AutoProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        if 'LLM' in shared.opts.cuda_compile:
            model = sd_models_compile.compile_torch(model)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    system_prompt = system_prompt or shared.opts.interrogate_vlm_system
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": b64(image)},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(devices.device, devices.dtype)
    output_ids = model.generate(
        **inputs,
        **get_kwargs(),
    )
    response = processor.batch_decode(output_ids,skip_special_tokens=True)
    return response


def git(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.GitForCausalLM.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        processor = transformers.GitProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    git_dict = {}
    git_dict['pixel_values'] = pixel_values.to(devices.device, devices.dtype)
    if len(question) > 0:
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        git_dict['input_ids'] = input_ids.to(devices.device)
    with devices.inference_context():
        generated_ids = model.generate(**git_dict)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def blip(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.BlipForQuestionAnswering.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        processor = transformers.BlipProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    inputs = processor(image, question, return_tensors="pt")
    inputs = inputs.to(devices.device, devices.dtype)
    with devices.inference_context():
        outputs = model.generate(**inputs)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response


def vilt(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.ViltForQuestionAnswering.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        processor = transformers.ViltProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    inputs = processor(image, question, return_tensors="pt")
    inputs = inputs.to(devices.device)
    with devices.inference_context():
        outputs = model(**inputs)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    response = model.config.id2label[idx]
    return response


def pix(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.Pix2StructForConditionalGeneration.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        processor = transformers.Pix2StructProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    if len(question) > 0:
        inputs = processor(images=image, text=question, return_tensors="pt").to(devices.device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(devices.device)
    with devices.inference_context():
        outputs = model.generate(**inputs)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response


def moondream(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        model = None
        model = transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            revision="2025-06-21",
            trust_remote_code=True,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        processor = transformers.AutoTokenizer.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        loaded = repo
        model.eval()
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    encoded = model.encode_image(image)
    with devices.inference_context():
        if question == 'CAPTION':
            response = model.caption(image, length="short")['caption']
        elif question == 'DETAILED CAPTION':
            response = model.caption(image, length="normal")['caption']
        elif question == 'MORE DETAILED CAPTION':
            response = model.caption(image, length="long")['caption']
        else:
            response = model.answer_question(encoded, question, processor)['answer']
        # model.detect(image, "face")
        # model.point(image, "person")
        # model.detect_gaze(image)
    return response


def florence(question: str, image: Image.Image, repo: str = None, revision: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    _get_imports = transformers.dynamic_module_utils.get_imports
    def get_imports(f):
        R = _get_imports(f)
        if "flash_attn" in R:
            R.remove("flash_attn") # flash_attn is optional
        return R
    revision = None
    if '@' in repo:
        repo, revision = repo.split('@')
    if model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}" path="{shared.opts.hfcache_dir}"')
        transformers.dynamic_module_utils.get_imports = get_imports
        model = None
        model = transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            **quant_args,
        )
        processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=True, revision=revision, cache_dir=shared.opts.hfcache_dir)
        transformers.dynamic_module_utils.get_imports = _get_imports
        loaded = repo
        model.eval()
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    if question.startswith('<'):
        task = question.split('>', 1)[0] + '>'
    else:
        task = '<MORE_DETAILED_CAPTION>'
    inputs = processor(text=task, images=image, return_tensors="pt")
    input_ids = inputs['input_ids'].to(devices.device)
    pixel_values = inputs['pixel_values'].to(devices.device, devices.dtype)
    with devices.inference_context():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **get_kwargs()
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(generated_text, task="task", image_size=(image.width, image.height))
    return response


def sa2(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = None
        model = transformers.AutoModel.from_pretrained(
            repo,
            torch_dtype=devices.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        model = model.eval()
        processor = transformers.AutoTokenizer.from_pretrained(
            repo,
            trust_remote_code=True,
            use_fast=False,
        )
        loaded = repo
        devices.torch_gc()
    sd_models.move_model(model, devices.device)
    if question.startswith('<'):
        task = question.split('>', 1)[0] + '>'
    else:
        task = '<MORE_DETAILED_CAPTION>'
    input_dict = {
        'image': image,
        'text': f'<image>{task}',
        'past_text': '',
        'mask_prompts': None,
        'tokenizer': processor,
        }
    return_dict = model.predict_forward(**input_dict)
    response = return_dict["prediction"] # the text format answer
    return response


def interrogate(question:str='', system_prompt:str=None, prompt:str=None, image:Image.Image=None, model_name:str=None, quiet:bool=False):
    global quant_args # pylint: disable=global-statement
    if not quiet:
        shared.state.begin('Interrogate')
    t0 = time.time()
    quant_args = model_quant.create_config(module='LLM')
    model_name = model_name or shared.opts.interrogate_vlm_model
    if isinstance(image, list):
        image = image[0] if len(image) > 0 else None
    if isinstance(image, dict) and 'name' in image:
        image = Image.open(image['name'])
    if image is None:
        return ''
    if image.width > 768 or image.height > 768:
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if prompt is not None and len(prompt) > 0:
        question = prompt
    if len(question) < 2:
        question = "Describe the image."
    if shared.sd_loaded:
        from modules.sd_models import apply_balanced_offload # prevent circular import
        apply_balanced_offload(shared.sd_model)

    from modules import modelloader
    modelloader.hf_login()

    try:
        if model_name is None:
            shared.log.error(f'Interrogate: type=vlm model="{model_name}" no model selected')
            return ''
        vqa_model = vlm_models.get(model_name, None)
        if vqa_model is None:
            shared.log.error(f'Interrogate: type=vlm model="{model_name}" unknown')
            return ''
        if image is None:
            shared.log.error(f'Interrogate: type=vlm model="{model_name}" no input image')
            return ''

        if 'git' in vqa_model.lower():
            answer = git(question, image, vqa_model)
        elif 'vilt' in vqa_model.lower():
            answer = vilt(question, image, vqa_model)
        elif 'blip' in vqa_model.lower():
            answer = blip(question, image, vqa_model)
        elif 'pix' in vqa_model.lower():
            answer = pix(question, image, vqa_model)
        elif 'moondream2' in vqa_model.lower():
            answer = moondream(question, image, vqa_model)
        elif 'florence' in vqa_model.lower():
            answer = florence(question, image, vqa_model)
        elif 'qwen' in vqa_model.lower() or 'torii' in vqa_model.lower():
            answer = qwen(question, image, vqa_model, system_prompt)
        elif 'smol' in vqa_model.lower():
            answer = smol(question, image, vqa_model, system_prompt)
        elif 'joytag' in vqa_model.lower():
            from modules.interrogate import joytag
            answer = joytag.predict(image)
        elif 'joycaption' in vqa_model.lower():
            from modules.interrogate import joycaption
            answer = joycaption.predict(question, image, vqa_model)
        elif 'deepseek' in vqa_model.lower():
            from modules.interrogate import deepseek
            answer = deepseek.predict(question, image, vqa_model)
        elif 'paligemma' in vqa_model.lower():
            answer = paligemma(question, image, vqa_model)
        elif 'gemma' in vqa_model.lower():
            answer = gemma(question, image, vqa_model, system_prompt)
        elif 'ovis' in vqa_model.lower():
            answer = ovis(question, image, vqa_model)
        elif 'sa2' in vqa_model.lower():
            answer = sa2(question, image, vqa_model)
        else:
            answer = 'unknown model'
    except Exception as e:
        errors.display(e, 'VQA')
        answer = 'error'

    if shared.opts.interrogate_offload and model is not None:
        sd_models.move_model(model, devices.cpu, force=True)
    devices.torch_gc(force=True, reason='vqa')
    answer = clean(answer, question)
    t1 = time.time()
    if not quiet:
        shared.log.debug(f'Interrogate: type=vlm model="{model_name}" repo="{vqa_model}" args={get_kwargs()} time={t1-t0:.2f}')
        shared.state.end()
    return answer


def batch(model_name, system_prompt, batch_files, batch_folder, batch_str, question, prompt, write, append, recursive):
    class BatchWriter:
        def __init__(self, folder, mode='w'):
            self.folder = folder
            self.csv = None
            self.file = None
            self.mode = mode

        def add(self, file, prompt):
            txt_file = os.path.splitext(file)[0] + ".txt"
            if self.mode == 'a':
                prompt = '\n' + prompt
            with open(os.path.join(self.folder, txt_file), self.mode, encoding='utf-8') as f:
                f.write(prompt)

        def close(self):
            if self.file is not None:
                self.file.close()

    files = []
    if batch_files is not None:
        files += [f.name for f in batch_files]
    if batch_folder is not None:
        files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        from modules.files_cache import list_files
        files += list(list_files(batch_str, ext_filter=['.png', '.jpg', '.jpeg', '.webp', '.jxl'], recursive=recursive))
    if len(files) == 0:
        shared.log.warning('Interrogate batch: type=vlm no images')
        return ''
    shared.state.begin('Interrogate batch')
    prompts = []
    if write:
        mode = 'w' if not append else 'a'
        writer = BatchWriter(os.path.dirname(files[0]), mode=mode)
    orig_offload = shared.opts.interrogate_offload
    shared.opts.interrogate_offload = False
    import rich.progress as rp
    pbar = rp.Progress(rp.TextColumn('[cyan]Caption:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
    with pbar:
        task = pbar.add_task(total=len(files), description='starting...')
        for file in files:
            pbar.update(task, advance=1, description=file)
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file)
                prompt = interrogate(question, system_prompt, prompt, image, model_name, quiet=True)
                prompts.append(prompt)
                if write:
                    writer.add(file, prompt)
            except Exception as e:
                shared.log.error(f'Interrogate batch: {e}')
    if write:
        writer.close()
    shared.opts.interrogate_offload = orig_offload
    shared.state.end()
    return '\n\n'.join(prompts)
