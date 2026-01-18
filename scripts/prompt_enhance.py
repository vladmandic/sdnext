from dataclasses import dataclass
import io
import os
import re
import time
import random
import base64
import torch
import transformers
import gradio as gr
from PIL import Image
from modules import scripts_manager, shared, devices, errors, processing, sd_models, sd_modules, timer, ui_symbols
from modules import ui_control_helpers


debug_enabled = os.environ.get('SD_LLM_DEBUG', None) is not None
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None


def b64(image):
    if image is None:
        return ''
    if isinstance(image, gr.Image): # should not happen
        return None
    with io.BytesIO() as stream:
        image.convert('RGB').save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def is_vision_model(model_name: str) -> bool:
    """Check if model supports vision/image input."""
    if not model_name:
        return False
    return model_name in Options.img2img


def is_thinking_model(model_name: str) -> bool:
    """Check if model supports thinking/reasoning mode."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    # Match VQA's detection patterns for consistency
    thinking_indicators = [
        'thinking',      # Qwen3-VL-*-Thinking models
        'moondream3',    # Moondream 3 supports thinking
        'moondream 3',
        'moondream2',    # Moondream 2 supports reasoning mode
        'moondream 2',
        'mimo',          # XiaomiMiMo models
    ]
    return any(indicator in model_lower for indicator in thinking_indicators)


def get_model_display_name(model_repo: str) -> str:
    """Generate display name with vision/reasoning symbols."""
    symbols = []
    if model_repo in Options.img2img:
        symbols.append(ui_symbols.vision)
    if is_thinking_model(model_repo):
        symbols.append(ui_symbols.reasoning)
    return f"{model_repo} {' '.join(symbols)}" if symbols else model_repo


def get_model_repo_from_display(display_name: str) -> str:
    """Strip symbols from display name to get repo."""
    if not display_name:
        return display_name
    result = display_name
    for symbol in [ui_symbols.vision, ui_symbols.reasoning]:
        result = result.replace(symbol, '')
    return result.strip()


def keep_think_block_open(text_prompt: str) -> str:
    """Remove closing </think> so model can continue reasoning with prefill."""
    think_open = "<think>"
    think_close = "</think>"
    last_open = text_prompt.rfind(think_open)
    if last_open == -1:
        return text_prompt
    close_index = text_prompt.find(think_close, last_open)
    if close_index == -1:
        return text_prompt
    end_close = close_index + len(think_close)
    while end_close < len(text_prompt) and text_prompt[end_close] in ' \t\r\n':
        end_close += 1
    return text_prompt[:close_index] + text_prompt[end_close:]


@dataclass
class Options:
    img2img = [
        'google/gemma-3-4b-it',
        'allura-org/Gemma-3-Glitter-4B',
        'Qwen/Qwen2.5-VL-3B-Instruct',
        'Qwen/Qwen3-VL-2B-Instruct',
        'Qwen/Qwen3-VL-2B-Thinking',
        'Qwen/Qwen3-VL-4B-Instruct',
        'Qwen/Qwen3-VL-4B-Thinking',
        'Qwen/Qwen3-VL-8B-Instruct',
        'Qwen/Qwen3-VL-8B-Thinking',
    ]
    models = {
        'google/gemma-3-1b-it': {},
        'google/gemma-3-4b-it': {},
        'google/gemma-3n-E2B-it': {},
        'google/gemma-3n-E4B-it': {},
        'Qwen/Qwen3-0.6B-FP8': {},
        'Qwen/Qwen3-1.7B-FP8': {},
        'Qwen/Qwen3-4B-FP8': {},
        'Qwen/Qwen3-0.6B': {},
        'Qwen/Qwen3-1.7B': {},
        'Qwen/Qwen3-4B': {},
        'Qwen/Qwen3-4B-Instruct-2507': {},
        'Qwen/Qwen2.5-0.5B-Instruct': {},
        'Qwen/Qwen2.5-1.5B-Instruct': {},
        'Qwen/Qwen2.5-3B-Instruct': {},
        'Qwen/Qwen2.5-VL-3B-Instruct': {},
        'Qwen/Qwen3-VL-2B-Instruct': {},
        'Qwen/Qwen3-VL-2B-Thinking': {},
        'Qwen/Qwen3-VL-4B-Instruct': {},
        'Qwen/Qwen3-VL-4B-Thinking': {},
        'Qwen/Qwen3-VL-8B-Instruct': {},
        'Qwen/Qwen3-VL-8B-Thinking': {},
        'microsoft/Phi-4-mini-instruct': {},
        'HuggingFaceTB/SmolLM2-135M-Instruct': {},
        'HuggingFaceTB/SmolLM2-360M-Instruct': {},
        'HuggingFaceTB/SmolLM2-1.7B-Instruct': {},
        'HuggingFaceTB/SmolLM3-3B': {},
        'meta-llama/Llama-3.2-1B-Instruct': {},
        'meta-llama/Llama-3.2-3B-Instruct': {},
        'cognitivecomputations/Dolphin3.0-Llama3.2-1B': {},
        'cognitivecomputations/Dolphin3.0-Llama3.2-3B': {},
        'nidum/Nidum-Gemma-3-4B-it-Uncensored': {},
        'allura-org/Gemma-3-Glitter-4B': {},
        # 'llava/Llama-3-8B-v1.1-Extracted': {
        #     'repo': 'hunyuanvideo-community/HunyuanVideo',
        #     'subfolder': 'text_encoder',
        #     'tokenizer': 'tokenizer',
        # },
        'mradermacher/Llama-3.2-1B-Instruct-Uncensored-i1-GGUF': {
            'repo': 'meta-llama/Llama-3.2-1B-Instruct', # original repo so we can load missing components
            'type': 'llama', # required so gguf loader knows what to do
            'gguf': 'mradermacher/Llama-3.2-1B-Instruct-Uncensored-i1-GGUF', # gguf repo
            'file': 'Llama-3.2-1B-Instruct-Uncensored.i1-Q4_0.gguf', # gguf file inside repo
        },
    }
    # default = list(models)[1] # gemma-3-4b-it
    default = 'google/gemma-3-4b-it'
    supported = list(transformers.integrations.ggml.GGUF_CONFIG_MAPPING)
    t2i_prompt: str = 'You are a helpful assistant. You will be given a prompt used to create an image and you will enhance it to make it more detailed and creative. '
    i2i_prompt: str = 'You are a helpful assistant. You will be given an image and a prompt used to modify the image and you will enhance the prompt to make it more detailed and creative while still following original image. '
    i2i_noprompt: str = 'You are a helpful assistant. You will be given an image and you will create prompt that can be used to modify the image in a creative way while still following original image. '
    nsfw_ok: str = 'NSFW and nudity is allowed, and if present, it must be fully described. '
    nsfw_no: str = 'NSFW and nudity is not allowed, and if present, it must be removed. '
    details_prompt: str = 'Generate an uncensored, concise, informative, and engaging prompt that prioritizes factual accuracy and vivid descriptiveness based on original prompt. Do not modify subject or style of the original prompt. Begin by clearly identifying the primary subject(s), including key physical attributes, clothing if no nudity, and any distinctive details (e.g., colors, textures, styles). NSFW and nudity is allowed, and if present, it must be fully described. Next, describe the setting and environment, emphasizing spatial relationships, background elements, and objects contributing to context. Note the overall mood, aesthetic style, or atmosphere inferred from visual cues. Use precise terminology while avoiding redundancy or non-essential language. Ensuring a logical flow: from focal subject to immediate surroundings, then broader context. Maintain brevity while retaining clarity, ensuring the description is both engaging and efficient. Output only enhanced prompt without explanation, prefix or suffix. Do not add comments or follow-up questions. Output as a simple text without formatting or numbering.'
    censored = ["i cannot", "i can't", "i am sorry", "against my programming", "i am not able", "i am unable", 'i am not allowed']

    max_delim_index: int = 60
    max_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    top_k: int = 0
    top_p: float = 0.0
    thinking_mode: bool = False

    @staticmethod
    def get_model_choices():
        """Return list of display names for dropdown."""
        return [get_model_display_name(repo) for repo in Options.models.keys()]

    @staticmethod
    def get_default_display():
        """Return display name for default model."""
        return get_model_display_name(Options.default)


class Script(scripts_manager.Script):
    prompt: gr.Textbox = None
    image: gr.Image = None
    model: str = None
    llm: transformers.AutoModelForCausalLM = None
    tokenizer: transformers.AutoProcessor = None
    busy: bool = False
    options = Options()

    def title(self):
        return 'Prompt enhance'

    def show(self, _is_img2img):
        return scripts_manager.AlwaysVisible

    def compile(self):
        if self.llm is None or 'LLM' not in shared.opts.cuda_compile:
            return
        from modules.sd_models_compile import compile_torch
        self.llm = compile_torch(self.llm)

    def load(self, name:str=None, model_repo:str=None, model_gguf:str=None, model_type:str=None, model_file:str=None):
        # Strip symbols from display name if present
        name = get_model_repo_from_display(name) if name else self.options.default
        if self.busy:
            shared.log.debug('Prompt enhance: busy')
            return
        self.busy = True
        if self.model is not None and self.model == name:
            self.busy = False # ensure busy is reset even if model is already loaded
            return

        from modules import modelloader, model_quant, ggml
        modelloader.hf_login()
        model_repo = model_repo or self.options.models.get(name, {}).get('repo', None) or name
        model_gguf = model_gguf or self.options.models.get(name, {}).get('gguf', None) or model_repo
        model_type = model_type or self.options.models.get(name, {}).get('type', None)
        model_file = model_file or self.options.models.get(name, {}).get('file', None)
        model_subfolder = self.options.models.get(name, {}).get('subfolder', None)
        model_tokenizer = self.options.models.get(name, {}).get('tokenizer', None)

        gguf_args = {}
        if model_type is not None and model_file is not None and len(model_type) > 2 and len(model_file) > 2:
            debug_log(f'Prompt enhance: gguf supported={self.options.supported}')
            if model_type not in self.options.supported:
                shared.log.error(f'Prompt enhance: name="{name}" repo="{model_repo}" fn="{model_file}" type={model_type} gguf not supported')
                shared.log.trace(f'Prompt enhance: gguf supported={self.options.supported}')
                self.busy = False
                return
            ggml.install_gguf()
            gguf_args['model_type'] = model_type
            gguf_args['gguf_file'] = model_file

        quant_args = model_quant.create_config(module='LLM') if not gguf_args else {}

        try:
            t0 = time.time()
            if self.llm is not None:
                self.llm = None
                shared.log.debug(f'Prompt enhance: name="{self.model}" unload')
            self.model = None
            load_args = { 'pretrained_model_name_or_path': model_repo if not gguf_args else model_gguf }
            if model_subfolder:
                load_args['subfolder'] = model_subfolder # Comma was incorrect here

            if 'Qwen3-VL' in model_repo or 'Qwen3VL' in model_repo:
                cls_name = transformers.Qwen3VLForConditionalGeneration
            elif 'Qwen2.5-VL' in model_repo or 'Qwen2_5_VL' in model_repo:
                cls_name = transformers.Qwen2_5_VLForConditionalGeneration
            elif 'Qwen2-VL' in model_repo or 'Qwen2VL' in model_repo:
                cls_name = transformers.Qwen2VLForConditionalGeneration
            else:
                cls_name = transformers.AutoModelForCausalLM

            self.llm = cls_name.from_pretrained(
                **load_args,
                trust_remote_code=True,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                # _attn_implementation="eager",
                **gguf_args,
                **quant_args,
            )
            self.llm.eval()
            if model_repo in self.options.img2img:
                cls = transformers.AutoProcessor # required to encode image
            else:
                cls = transformers.AutoTokenizer
            tokenizer_args = { 'pretrained_model_name_or_path': model_repo }
            if model_tokenizer:
                tokenizer_args['subfolder'] = model_tokenizer
            self.tokenizer = cls.from_pretrained(
                **tokenizer_args,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.tokenizer.is_processor = model_repo in self.options.img2img

            if debug_enabled:
                modules = sd_modules.get_model_stats(self.llm) + sd_modules.get_model_stats(self.tokenizer)
                for m in modules:
                    debug_log(f'Prompt enhance: {m}')
            self.model = name
            t1 = time.time()
            shared.log.info(f'Prompt enhance: cls={self.llm.__class__.__name__} name="{name}" repo="{model_repo}" fn="{model_file}" time={t1-t0:.2f} loaded')
            self.compile()
        except Exception as e:
            shared.log.error(f'Prompt enhance: load {e}')
            errors.display(e, 'Prompt enhance')
        devices.torch_gc()
        self.busy = False

    def censored(self, response):
        text = response.lower().replace("i'm", "i am")
        return any(c.lower() in text for c in self.options.censored)

    def unload(self):
        if self.llm is not None:
            model_name = self.model
            shared.log.debug(f'Prompt enhance: unloading model="{model_name}"')
            sd_models.move_model(self.llm, devices.cpu, force=True)
            self.model = None
            self.llm = None
            self.tokenizer = None
            devices.torch_gc(force=True, reason='prompt enhance unload')
            shared.log.debug(f'Prompt enhance: model="{model_name}" unloaded')
        else:
            shared.log.debug('Prompt enhance: no model loaded')

    def clean(self, response, keep_thinking=False, prefill_text='', keep_prefill=False):
        # Handle thinking tags FIRST (before generic tag removal)
        if '<think>' in response or '</think>' in response:
            if keep_thinking:
                # Format: handle partial tags (</think> without <think> means thinking was in prompt)
                if '</think>' in response and '<think>' not in response:
                    response = 'Reasoning:\n' + response.replace('</think>', '\n\nAnswer:\n')
                else:
                    response = response.replace('<think>', 'Reasoning:\n').replace('</think>', '\n\nAnswer:\n')
            else:
                # Strip all thinking content
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                response = response.replace('</think>', '')  # Handle orphaned closing tags

        # remove special characters
        response = response.replace('"', '').replace("'", "").replace('"', '').replace('"', '').replace('**', '')
        # remove repeating characters
        response = response.replace('\n\n', '\n').replace('  ', ' ').replace('...', '.')

        # remove comments between brackets (but not Reasoning:/Answer: which we may have added)
        response = re.sub(r'<.*?>', '', response)
        response = re.sub(r'\[.*?\]', '', response)
        response = re.sub(r'\/.*?\/', '', response)

        # remove llm commentary
        removed = ''
        if response.startswith('Prompt'):
            removed, response = response.split('Prompt', maxsplit=1)
        if 0 <= response.find(':') < self.options.max_delim_index:
            # Don't split on "Reasoning:" or "Answer:" if we're keeping thinking
            colon_pos = response.find(':')
            prefix_text = response[:colon_pos].strip()
            if not keep_thinking or (prefix_text not in ['Reasoning', 'Answer']):
                removed, response = response.split(':', maxsplit=1)
        if 0 <= response.find('---') < self.options.max_delim_index:
            response, removed = response.split('---', maxsplit=1)
        if len(removed) > 0:
            debug_log(f'Prompt enhance: max={self.options.max_delim_index} removed="{removed}"')

        # remove bullets and lists
        lines = [re.sub(r'^(\s*[-*]|\s*\d+)\s+', '', line).strip() for line in response.splitlines()]
        response = '\n'.join(lines)

        response = response.strip()

        # Handle prefill retention/removal
        prefill_text = (prefill_text or '').strip()
        if prefill_text:
            if keep_prefill:
                # Add prefill if it's missing from the cleaned response
                if not response.startswith(prefill_text):
                    sep = '' if (not response or response[0] in '.,!?;:') else ' '
                    response = f'{prefill_text}{sep}{response}'
            else:
                # Remove prefill if it's present in the cleaned response
                if response.startswith(prefill_text):
                    response = response[len(prefill_text):].strip()

        return response

    def post(self, response, prefix, suffix, networks):
        response = response.strip()
        prefix = prefix.strip()
        suffix = suffix.strip()
        if len(prefix) > 0:
            response = f'{prefix} {response}'
        if len(suffix) > 0:
            response = f'{response} {suffix}'
        if len(networks) > 0:
            response = f'{response} {" ".join(networks)}'
        return response

    def extract(self, prompt):
        pattern = r'(<.*?>)'
        matches = re.findall(pattern, prompt)
        filtered = re.sub(pattern, '', prompt)
        return filtered, matches

    def enhance(self, model: str=None, prompt:str=None, system:str=None, prefix:str=None, suffix:str=None, sample:bool=None, tokens:int=None, temperature:float=None, penalty:float=None, top_k:int=None, top_p:float=None, thinking:bool=False, seed:int=-1, image=None, nsfw:bool=None, use_vision:bool=True, prefill:str='', keep_prefill:bool=False, keep_thinking:bool=False):
        # Strip symbols from model name if present
        model = get_model_repo_from_display(model) if model else self.options.default
        prompt = prompt or (self.prompt.value if self.prompt else "") # Check if self.prompt is None
        # Handle vision toggle - if disabled or non-VL model, don't use image
        if use_vision and is_vision_model(model):
            image = image or self.image
        else:
            image = None
        prefix = prefix or ''
        suffix = suffix or ''
        tokens = tokens or self.options.max_tokens
        penalty = penalty or self.options.repetition_penalty
        temperature = temperature or self.options.temperature
        top_k = top_k if top_k is not None else self.options.top_k
        top_p = top_p if top_p is not None else self.options.top_p
        thinking = thinking or self.options.thinking_mode
        sample = sample if sample is not None else self.options.do_sample
        nsfw = nsfw if nsfw is not None else True # Default nsfw to True if not provided
        debug_log(f'Prompt enhance: model="{model}" model_class="{self.llm.__class__.__name__ if self.llm else "not loaded"}" nsfw={nsfw} thinking={thinking} prefill="{prefill[:30] if prefill else ""}" use_vision={use_vision} image={image is not None}')

        while self.busy:
            time.sleep(0.1)
        self.load(model)
        if seed is None or seed == -1:
            random.seed()
            seed = int(random.randrange(4294967294))
        torch.manual_seed(seed)
        if self.llm is None:
            shared.log.error('Prompt enhance: model not loaded')
            return prompt
        prompt_text, networks = self.extract(prompt) # Use prompt_text after extraction
        debug_log(f'Prompt enhance: networks={networks}')

        current_image = None
        # Only process images if vision is enabled and model supports it
        if use_vision and is_vision_model(model):
            try:
                if image is not None and isinstance(image, gr.Image):
                    current_image = image.value
                elif image is not None and isinstance(image, Image.Image): # if image is already a PIL image
                    current_image = image
                if current_image is not None and (current_image.width <= 64 or current_image.height <= 64):
                    current_image = None
                # Fallback to Kanvas/Control input if no image from Gradio component (e.g., when Kanvas is active)
                if current_image is None and ui_control_helpers.input_source is not None:
                    if isinstance(ui_control_helpers.input_source, list) and len(ui_control_helpers.input_source) > 0:
                        current_image = ui_control_helpers.input_source[0]
                    elif isinstance(ui_control_helpers.input_source, Image.Image):
                        current_image = ui_control_helpers.input_source
            except Exception:
                current_image = None
        debug_log(f'Prompt enhance: current_image={current_image is not None} size={f"{current_image.width}x{current_image.height}" if current_image else "N/A"}')

        # Check if vision was requested but no image is available
        if use_vision and is_vision_model(model) and current_image is None:
            shared.log.error(f'Prompt enhance: model="{model}" error="No input image provided"')
            return 'Error: No input image provided. Please upload or select an image.'

        # Resize large images to match VQA performance (Qwen3-VL performance is sensitive to resolution)
        # Create a copy to avoid modifying the original image used by img2img
        if current_image is not None and isinstance(current_image, Image.Image):
            original_size = (current_image.width, current_image.height)
            needs_resize = current_image.width > 768 or current_image.height > 768
            needs_rgb = current_image.mode != 'RGB'

            if needs_resize or needs_rgb:
                # Copy the image before any modifications to preserve the original
                current_image = current_image.copy()

                if needs_resize:
                    current_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    debug_log(f'Prompt enhance: Resized image from {original_size} to {(current_image.width, current_image.height)}')

                if needs_rgb:
                    current_image = current_image.convert('RGB')
                    debug_log('Prompt enhance: Converted image to RGB mode')

        has_system = system is not None and len(system) > 4

        if current_image is not None and isinstance(current_image, Image.Image):
            if (self.tokenizer is None) or (not self.tokenizer.is_processor):
                shared.log.error('Prompt enhance: image not supported by model')
                return prompt_text # Return original text part if image cannot be processed
            if prompt_text is not None and len(prompt_text) > 0:
                if not has_system:
                    system = self.options.i2i_prompt
                    system += self.options.nsfw_ok if nsfw else self.options.nsfw_no
                    system += self.options.details_prompt
                chat_template = [
                    { "role": "system", "content": [
                        {"type": "text", "text": system }
                    ] },
                    { "role": "user",   "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": b64(current_image)}
                    ] },
                ]
            else:
                if not has_system:
                    system = self.options.i2i_noprompt
                    system += self.options.nsfw_ok if nsfw else self.options.nsfw_no
                    system += self.options.details_prompt
                chat_template = [
                    { "role": "system", "content": [
                        {"type": "text", "text": system }
                    ] },
                    { "role": "user",   "content": [
                        {"type": "image", "image": b64(current_image)}
                    ] },
                ]
        else:
            if not has_system:
                system = self.options.t2i_prompt
                system += self.options.nsfw_ok if nsfw else self.options.nsfw_no
                system += self.options.details_prompt
            if not self.tokenizer.is_processor:
                chat_template = [
                    { "role": "system", "content": system },
                    { "role": "user",   "content": prompt_text },
                ]
            else:
                chat_template = [
                    { "role": "system", "content": [
                        {"type": "text", "text": system }
                    ] },
                    { "role": "user",   "content": [
                        {"type": "text", "text": prompt_text},
                    ] },
                ]

        # Prepare prefill (VQA approach: string concatenation, not assistant message)
        prefill_text = (prefill or '').strip()
        use_prefill = len(prefill_text) > 0
        is_thinking = is_thinking_model(model)

        debug_log(f'Prompt enhance: chat_template roles={[msg["role"] for msg in chat_template]} is_thinking={is_thinking} thinking={thinking} use_prefill={use_prefill}')
        t0 = time.time()
        self.busy = True
        try:
            # Generate text prompt using template (WITHOUT enable_thinking parameter)
            # Let template naturally generate <think> for thinking models
            try:
                text_prompt = self.tokenizer.apply_chat_template(
                    chat_template,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except TypeError:
                text_prompt = self.tokenizer.apply_chat_template(
                    chat_template,
                    tokenize=False,
                )

            # Manually handle thinking tags and prefill (VQA Qwen approach)
            if is_thinking:
                if not thinking:
                    # User wants to SKIP thinking
                    # Template opened the block with <think>, close it immediately
                    text_prompt += "</think>\n"
                    if use_prefill:
                        text_prompt += prefill_text
                    debug_log('Prompt enhance: forced thinking off, appended </think>')
                else:
                    # User wants thinking - prefill becomes part of thought process
                    if use_prefill:
                        text_prompt += prefill_text
                    debug_log('Prompt enhance: thinking enabled, prefill inside think block')
            else:
                # Standard model (no <think> block)
                if use_prefill:
                    text_prompt += prefill_text

            debug_log(f'Prompt enhance: final text_prompt (last 200 chars)="{text_prompt[-200:]}"')

            # Tokenize the final prompt
            # For VL models with images, pass the image to the processor (like VQA does)
            if self.tokenizer.is_processor and current_image is not None:
                inputs = self.tokenizer(text=[text_prompt], images=[current_image], padding=True, return_tensors="pt")
            elif self.tokenizer.is_processor:
                # VL processor without image - must use explicit text= parameter
                inputs = self.tokenizer(text=[text_prompt], images=None, padding=True, return_tensors="pt")
            else:
                inputs = self.tokenizer(text_prompt, return_tensors="pt")
            inputs = inputs.to(devices.device).to(devices.dtype)

            input_len = inputs['input_ids'].shape[1]
            debug_log(f'Prompt enhance: input_len={input_len} input_ids_shape={inputs["input_ids"].shape} sample={sample} temp={temperature} penalty={penalty} max_tokens={tokens}')
        except Exception as e:
            shared.log.error(f'Prompt enhance tokenize: {e}')
            errors.display(e, 'Prompt enhance')
            self.busy = False
            return prompt_text # Return original text part on error
        try:
            with devices.inference_context():
                sd_models.move_model(self.llm, devices.device)
                gen_kwargs = {
                    'do_sample': sample,
                    'temperature': float(temperature),
                    'max_new_tokens': int(tokens),
                    'repetition_penalty': float(penalty),
                }
                if top_k > 0:
                    gen_kwargs['top_k'] = int(top_k)
                if top_p > 0:
                    gen_kwargs['top_p'] = float(top_p)
                outputs = self.llm.generate(**inputs, **gen_kwargs)
                if shared.opts.diffusers_offload_mode != 'none':
                    sd_models.move_model(self.llm, devices.cpu, force=True)
                    devices.torch_gc(force=True, reason='prompt enhance offload')
            outputs_cropped = outputs[:, input_len:]
            response = self.tokenizer.batch_decode(
                outputs_cropped,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if debug_enabled:
                response_before_clean = response[0] if isinstance(response, list) else response
                debug_log(f'Prompt enhance: response_before_clean="{response_before_clean}"')
        except Exception as e:
            outputs = None
            shared.log.error(f'Prompt enhance generate: {e}')
            errors.display(e, 'Prompt enhance')
            self.busy = False
            response = f'Error: {str(e)}'
        t1 = time.time()

        if isinstance(response, list):
            response = response[0]
        is_censored =  self.censored(response)
        if not is_censored:
            response = self.clean(response, keep_thinking=keep_thinking, prefill_text=prefill_text, keep_prefill=keep_prefill)
            response = self.post(response, prefix, suffix, networks)
        shared.log.info(f'Prompt enhance: model="{model}" nsfw={nsfw} time={t1-t0:.2f} seed={seed} sample={sample} temperature={temperature} penalty={penalty} thinking={thinking} keep_thinking={keep_thinking} prefill="{prefill_text[:20] if prefill_text else ""}" keep_prefill={keep_prefill} tokens={tokens} inputs={input_len} outputs={outputs.shape[-1] if isinstance(outputs, torch.Tensor) else 0} prompt={len(prompt_text)} response={len(response)}')
        debug_log(f'Prompt enhance: prompt="{prompt_text}"')
        debug_log(f'Prompt enhance: response_after_clean="{response}"')
        self.busy = False
        if is_censored:
            shared.log.warning(f'Prompt enhance: censored response="{response}"')
            return prompt # Return original full prompt on censorship
        return response

    def apply(self, prompt, image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking):
        response = self.enhance(
            prompt=prompt,
            image=image,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            tokens=max_tokens,
            temperature=temperature,
            penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            thinking=thinking_mode,
            nsfw=nsfw_mode,
            use_vision=use_vision,
            prefill=prefill_text,
            keep_prefill=keep_prefill,
            keep_thinking=keep_thinking,
        )
        if apply_prompt:
            return [response, response]
        return [response, gr.update()]

    def get_custom(self, name):
        # Strip symbols from display name to get repo
        repo_name = get_model_repo_from_display(name)
        model_repo = self.options.models.get(repo_name, {}).get('repo', None) or repo_name
        model_gguf = self.options.models.get(repo_name, {}).get('gguf', None)
        model_type = self.options.models.get(repo_name, {}).get('type', None)
        model_file = self.options.models.get(repo_name, {}).get('file', None)
        return [model_repo, model_gguf, model_type, model_file]

    def update_vision_toggle(self, model_name):
        """Update vision toggle interactivity and value based on model selection."""
        repo_name = get_model_repo_from_display(model_name)
        is_vl = is_vision_model(repo_name)
        # When non-VL model: disable and uncheck. When VL model: enable and check.
        return gr.update(interactive=is_vl, value=is_vl)

    def ui(self, _is_img2img):
        with gr.Accordion('Prompt enhance', open=False, elem_id='prompt_enhance'):
            gr.HTML('<style>#prompt_enhance_use_vision:has(input:disabled) { opacity: 0.5; }</style>')
            with gr.Row():
                apply_btn = gr.Button(value='Enhance now', elem_id='prompt_enhance_apply', variant='primary')
            with gr.Row():
                apply_prompt = gr.Checkbox(label='Apply to prompt', value=False)
                apply_auto = gr.Checkbox(label='Auto enhance', value=False)
            with gr.Row():
                # Set initial state based on whether default model supports vision
                default_is_vl = is_vision_model(Options.default)
                use_vision = gr.Checkbox(label='Use vision', value=default_is_vl, interactive=default_is_vl, elem_id='prompt_enhance_use_vision')
            gr.HTML('<br>')
            with gr.Group():
                with gr.Row():
                    llm_model = gr.Dropdown(label='LLM model', choices=Options.get_model_choices(), value=Options.get_default_display(), interactive=True, allow_custom_value=True, elem_id='prompt_enhance_model')
                with gr.Row():
                    load_btn = gr.Button(value='Load model', elem_id='prompt_enhance_load', variant='secondary')
                    load_btn.click(fn=self.load, inputs=[llm_model], outputs=[])
                    unload_btn = gr.Button(value='Unload model', elem_id='prompt_enhance_unload', variant='secondary')
                    unload_btn.click(fn=self.unload, inputs=[], outputs=[])
                with gr.Accordion('Custom model', open=False, elem_id='prompt_enhance_custom'):
                    with gr.Row():
                        model_repo = gr.Textbox(label='Model repo', value=None, interactive=True, elem_id='prompt_enhance_model_repo', placeholder='Original model repo on huggingface')
                    with gr.Row():
                        model_gguf = gr.Textbox(label='Model gguf', value=None, interactive=True, elem_id='prompt_enhance_model_gguf', placeholder='Optional GGUF model repo on huggingface')
                    with gr.Row():
                        model_type = gr.Textbox(label='Model type', value=None, interactive=True, elem_id='prompt_enhance_model_type', placeholder='Optional GGUF model type')
                    with gr.Row():
                        model_file = gr.Textbox(label='Model file', value=None, interactive=True, elem_id='prompt_enhance_model_file', placeholder='Optional GGUF model file inside GGUF model repo')
                    with gr.Row():
                        custom_btn = gr.Button(value='Load custom model', elem_id='prompt_enhance_custom_load', variant='secondary')
                        custom_btn.click(fn=self.load, inputs=[model_repo, model_repo, model_gguf, model_type, model_file], outputs=[])
                        llm_model.change(fn=self.get_custom, inputs=[llm_model], outputs=[model_repo, model_gguf, model_type, model_file])
                        gr.HTML('<br>')
                with gr.Accordion('Options', open=False, elem_id='prompt_enhance_options'):
                    with gr.Row():
                        max_tokens = gr.Slider(label='Max tokens', value=self.options.max_tokens, minimum=10, maximum=1024, step=1, interactive=True)
                        do_sample = gr.Checkbox(label='Use samplers', value=self.options.do_sample, interactive=True)
                    with gr.Row():
                        temperature = gr.Slider(label='Temperature', value=self.options.temperature, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        repetition_penalty = gr.Slider(label='Repetition penalty', value=self.options.repetition_penalty, minimum=0.0, maximum=2.0, step=0.01, interactive=True)
                    with gr.Row():
                        top_k = gr.Slider(label='Top-K', value=self.options.top_k, minimum=0, maximum=100, step=1, interactive=True)
                        top_p = gr.Slider(label='Top-P', value=self.options.top_p, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    with gr.Row():
                        nsfw_mode = gr.Checkbox(label='NSFW allowed', value=True, interactive=True)
                        thinking_mode = gr.Checkbox(label='Thinking mode', value=False, interactive=True)
                    with gr.Row():
                        keep_thinking = gr.Checkbox(label='Keep Thinking Trace', value=False, interactive=True)
                        keep_prefill = gr.Checkbox(label='Keep Prefill', value=False, interactive=True)
                    with gr.Row():
                        prefill_text = gr.Textbox(label='Prefill text', value='', placeholder='Optional: pre-fill start of model response', interactive=True, lines=1)
                    gr.HTML('<br>')
                with gr.Accordion('Input', open=False, elem_id='prompt_enhance_system_prompt'): # Corrected elem_id reference
                    with gr.Row():
                        prompt_prefix = gr.Textbox(label='Prompt prefix', value='', placeholder='Text prepended to the enhanced result', interactive=True, lines=2, elem_id='prompt_enhance_prefix')
                    with gr.Row():
                        prompt_suffix = gr.Textbox(label='Prompt suffix', value='', placeholder='Text appended to the enhanced result', interactive=True, lines=2, elem_id='prompt_enhance_suffix')
                    with gr.Row():
                        prompt_system = gr.Textbox(label='System prompt', value='', placeholder='Leave empty to use built-in enhancement instructions', interactive=True, lines=4, elem_id='prompt_enhance_system')
                with gr.Accordion('Output', open=True, elem_id='prompt_enhance_output'): # Corrected elem_id reference
                    with gr.Row():
                        prompt_output = gr.Textbox(label='Enhanced prompt', value='', placeholder='Enhanced prompt will appear here', interactive=True, lines=4, max_lines=12, elem_id='prompt_enhance_result')
                    with gr.Row():
                        clear_btn = gr.Button(value='Clear', elem_id='prompt_enhance_clear', variant='secondary')
                        clear_btn.click(fn=lambda: '', inputs=[], outputs=[prompt_output])
                        copy_btn = gr.Button(value='Set prompt', elem_id='prompt_enhance_copy', variant='secondary')
                        copy_btn.click(fn=lambda x: x, inputs=[prompt_output], outputs=[self.prompt])
            if self.image is None:
                self.image = gr.Image(type='pil', interactive=False, visible=False, width=64, height=64) # dummy image
            # Update vision toggle interactivity when model changes
            llm_model.change(fn=self.update_vision_toggle, inputs=[llm_model], outputs=[use_vision], show_progress=False)
            apply_btn.click(fn=self.apply, inputs=[self.prompt, self.image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking], outputs=[prompt_output, self.prompt])
        return [self.prompt, self.image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking]

    def after_component(self, component, **kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt', 'video_prompt']:
            self.prompt = component
            self.prompt.use_original = True
        if getattr(component, 'elem_id', '') in ['img2img_image', 'control_input_select']:
            self.image = component
            self.image.use_original = True

    def before_process(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=unused-argument
        _self_prompt, self_image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking = args
        if not apply_auto and not p.enhance_prompt:
            return
        if shared.state.skipped or shared.state.interrupted:
            return
        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        jobid = shared.state.begin('LLM')
        p.prompt = self.enhance(
            prompt=p.prompt,
            seed=p.seed,
            image=self_image,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            tokens=max_tokens,
            temperature=temperature,
            penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            thinking=thinking_mode,
            nsfw=nsfw_mode,
            use_vision=use_vision,
            prefill=prefill_text,
            keep_prefill=keep_prefill,
            keep_thinking=keep_thinking,
        )
        timer.process.record('prompt')
        p.extra_generation_params['LLM'] = llm_model
        shared.state.end(jobid)
