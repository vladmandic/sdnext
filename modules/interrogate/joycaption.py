# based on <https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava>

from dataclasses import dataclass
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from modules import shared, devices, sd_models, model_quant


"""
Example prompts
Short description: Write a short description of the image.
Detailed descriptive: Please provide a detailed description of the image.
Descriptive: Write a descriptive caption for this image in a formal tone.
Descriptive (Informal): Write a descriptive caption for this image in a casual tone.
Training Prompt: Write a stable diffusion prompt for this image.
MidJourney: Write a MidJourney prompt for this image.
Booru tag list: Write a list of Booru tags for this image.
Booru-like tag list: Write a list of Booru-like tags for this image.
Art Critic: Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.
Product Listing: Write a caption for this image as though it were a product listing.
Social Media Post: Write a caption for this image as if it were being used for a social media post.
Extra Options:
- If there is a person/character in the image you must refer to them as {name}.
- Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).
- Include information about lighting.
- Include information about camera angle.
- Include information about whether there is a watermark or not.
- Include information about whether there are JPEG artifacts or not.
- If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.
- Do NOT include anything sexual; keep it PG.
- Do NOT mention the image's resolution.
- You MUST include information about the subjective aesthetic quality of the image from low to very high.
- Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.
- Do NOT mention any text that is in the image.
- Specify the depth of field and whether the background is in focus or blurred.
- If applicable, mention the likely use of artificial or natural lighting sources.
- Do NOT use any ambiguous language.
- Include whether the image is sfw, suggestive, or nsfw.
- ONLY describe the most important elements of the image.
"""

@dataclass
class JoyOptions():
    repo: str = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
    temp: float = 0.5
    top_k: float = 10
    top_p: float = 0.9
    max_new_tokens: int = 512
    sample: bool = True

    def __str__(self):
        return f'repo="{self.repo}" temp={self.temp} top_k={self.top_k} top_p={self.top_p} sample={self.sample} tokens={self.max_new_tokens}'


processor: AutoProcessor = None
llava_model: LlavaForConditionalGeneration = None
opts = JoyOptions()


@torch.no_grad()
def predict(question: str, image, vqa_model: str = None) -> str:
    global llava_model, processor # pylint: disable=global-statement
    opts.max_new_tokens = shared.opts.interrogate_vlm_max_length
    if vqa_model is not None and opts.repo != vqa_model:
        opts.repo = vqa_model
        llava_model = None
    if llava_model is None:
        shared.log.info(f'Interrogate: type=vlm model="JoyCaption" {str(opts)}')

        processor = AutoProcessor.from_pretrained(opts.repo)
        quant_args = model_quant.create_config(module='LLM')
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            opts.repo,
            torch_dtype=devices.dtype,
            device_map="auto",
            cache_dir=shared.opts.hfcache_dir,
            **quant_args,
        )
        llava_model.eval()

    if len(question) < 2:
        question = "Describe the image."
    question = question.replace('<', '').replace('>', '')
    convo = [
        { "role": "system", "content": "You are a helpful image captioner." },
        { "role": "user", "content": question },
    ]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(devices.device) # Process the inputs
    inputs['pixel_values'] = inputs['pixel_values'].to(devices.dtype)
    sd_models.move_model(llava_model, devices.device)
    with devices.inference_context():
        generate_ids = llava_model.generate( # Generate the captions
            **inputs,
            # input_ids=inputs['input_ids'],
            # pixel_values=inputs['pixel_values'],
            # attention_mask=inputs['attention_mask'],
            max_new_tokens=opts.max_new_tokens,
            suppress_tokens=None,
            use_cache=True,
            do_sample=opts.sample,
            temperature=opts.temp,
            top_k=opts.top_k,
            top_p=opts.top_p,
        )[0]
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:] # Trim off the prompt
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # Decode the caption
    sd_models.move_model(llava_model, devices.cpu, force=True)
    caption = caption.replace('\n\n', '\n').strip()
    return caption
