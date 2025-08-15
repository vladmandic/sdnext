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
from modules import scripts_manager, shared, devices, errors, processing, sd_models, sd_modules, timer


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


@dataclass
class Options:
    img2img = [
        'google/gemma-3-4b-it',
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
    max_tokens: int = 50
    do_sample: bool = True
    temperature: float = 0.15
    repetition_penalty: float = 1.2
    thinking_mode: bool = False


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
        name = name or self.options.default
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
            self.llm = transformers.AutoModelForCausalLM.from_pretrained(
                **load_args,
                trust_remote_code=True,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                _attn_implementation="eager",
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
                    shared.log.trace(f'Prompt enhance: {m}')
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
            sd_models.move_model(self.llm, devices.cpu)
        self.model = None
        self.llm = None
        self.tokenizer = None
        devices.torch_gc()
        shared.log.debug('Prompt enhance: model unloaded')

    def clean(self, response):
        # remove special characters
        response = response.replace('"', '').replace("'", "").replace('“', '').replace('”', '').replace('**', '')
        # remove repeating characters
        response = response.replace('\n\n', '\n').replace('  ', ' ').replace('...', '.')

        # remove comments between brackets
        response = re.sub(r'<.*?>', '', response)
        response = re.sub(r'\[.*?\]', '', response) # Fixed regex for brackets
        response = re.sub(r'\/.*?\/', '', response) # Fixed regex for slashes

        # remove llm commentary
        removed = ''
        if response.startswith('Prompt'):
            removed, response = response.split('Prompt', maxsplit=1)
        if 0 <= response.find(':') < self.options.max_delim_index:
            removed, response = response.split(':', maxsplit=1)
        if 0 <= response.find('---') < self.options.max_delim_index:
            response, removed = response.split('---', maxsplit=1)
        if len(removed) > 0:
            debug_log(f'Prompt enhance: max={self.options.max_delim_index} removed="{removed}"')

        # remove bullets and lists
        lines = [re.sub(r'^(\s*[-*]|\s*\d+)\s+', '', line).strip() for line in response.splitlines()] # Fixed regex
        response = '\n'.join(lines)

        response = response.strip()
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

    def enhance(self, model: str=None, prompt:str=None, system:str=None, prefix:str=None, suffix:str=None, sample:bool=None, tokens:int=None, temperature:float=None, penalty:float=None, thinking:bool=False, seed:int=-1, image=None, nsfw:bool=None):
        model = model or self.options.default
        prompt = prompt or (self.prompt.value if self.prompt else "") # Check if self.prompt is None
        image = image or self.image
        prefix = prefix or ''
        suffix = suffix or ''
        tokens = tokens or self.options.max_tokens
        penalty = penalty or self.options.repetition_penalty
        temperature = temperature or self.options.temperature
        thinking = thinking or self.options.thinking_mode
        sample = sample if sample is not None else self.options.do_sample
        nsfw = nsfw if nsfw is not None else True # Default nsfw to True if not provided

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
        try:
            if image is not None and isinstance(image, gr.Image):
                current_image = image.value
            elif image is not None and isinstance(image, Image.Image): # if image is already a PIL image
                current_image = image
            if current_image is not None and (current_image.width <= 64 or current_image.height <= 64):
                current_image = None
        except Exception:
            current_image = None

        has_system = system is not None and len(system) > 4
        mode = 'custom' if has_system else ''

        if current_image is not None and isinstance(current_image, Image.Image):
            if (self.tokenizer is None) or (not self.tokenizer.is_processor):
                shared.log.error('Prompt enhance: image not supported by model')
                return prompt_text # Return original text part if image cannot be processed
            if prompt_text is not None and len(prompt_text) > 0:
                if not has_system:
                    mode = 'i2i-prompt'
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
                    mode = 'i2i-noprompt'
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
                mode = 't2i+tokenizer'
                chat_template = [
                    { "role": "system", "content": system },
                    { "role": "user",   "content": prompt_text },
                ]
            else:
                mode = 't2i+processor'
                chat_template = [
                    { "role": "system", "content": [
                        {"type": "text", "text": system }
                    ] },
                    { "role": "user",   "content": [
                        {"type": "text", "text": prompt_text},
                    ] },
                ]

        t0 = time.time()
        self.busy = True
        try:
            inputs = self.tokenizer.apply_chat_template(
                chat_template,
                add_generation_prompt=True,
                enable_thinking=thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(devices.device).to(devices.dtype)
            input_len = inputs['input_ids'].shape[1]
        except Exception as e:
            shared.log.error(f'Prompt enhance tokenize: {e}')
            errors.display(e, 'Prompt enhance')
            self.busy = False
            return prompt_text # Return original text part on error
        try:
            with devices.inference_context():
                sd_models.move_model(self.llm, devices.device)
                outputs = self.llm.generate(
                    **inputs,
                    do_sample=sample,
                    temperature=float(temperature),
                    max_new_tokens=int(input_len + tokens),
                    repetition_penalty=float(penalty),
                )
                if shared.opts.diffusers_offload_mode != 'none':
                    sd_models.move_model(self.llm, devices.cpu)
                    devices.torch_gc()
            if debug_enabled:
                raw_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                shared.log.trace(f'Prompt enhance: raw="{raw_response}"')
            outputs_cropped = outputs[:, input_len:]
            response = self.tokenizer.batch_decode(
                outputs_cropped,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
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
            response = self.clean(response)
            response = self.post(response, prefix, suffix, networks)
        shared.log.info(f'Prompt enhance: model="{model}" mode="{mode}" nsfw={nsfw} time={t1-t0:.2f} seed={seed} sample={sample} temperature={temperature} penalty={penalty} thinking={thinking} tokens={tokens} inputs={input_len} outputs={outputs.shape[-1] if isinstance(outputs, torch.Tensor) else 0} prompt={len(prompt_text)} response={len(response)}') # Added check for outputs
        if debug_enabled:
            shared.log.trace(f'Prompt enhance: prompt="{prompt_text}"')
            shared.log.trace(f'Prompt enhance: response="{response}"')
        self.busy = False
        if is_censored:
            shared.log.warning(f'Prompt enhance: censored response="{response}"')
            return prompt # Return original full prompt on censorship
        return response

    def apply(self, prompt, image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, thinking_mode, nsfw_mode): # Added nsfw_mode
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
            thinking=thinking_mode,
            nsfw=nsfw_mode # Pass nsfw_mode here
        )
        if apply_prompt:
            return [response, response]
        return [response, gr.update()]

    def get_custom(self, name):
        model_repo = self.options.models.get(name, {}).get('repo', None) or name
        model_gguf = self.options.models.get(name, {}).get('gguf', None)
        model_type = self.options.models.get(name, {}).get('type', None)
        model_file = self.options.models.get(name, {}).get('file', None)
        return [model_repo, model_gguf, model_type, model_file]

    def ui(self, _is_img2img):
        with gr.Accordion('Prompt enhance', open=False, elem_id='prompt_enhance'):
            with gr.Row():
                apply_btn = gr.Button(value='Enhance now', elem_id='prompt_enhance_apply', variant='primary')
            with gr.Row():
                apply_prompt = gr.Checkbox(label='Apply to prompt', value=False)
                apply_auto = gr.Checkbox(label='Auto enhance', value=False)
            gr.HTML('<br>')
            with gr.Group():
                with gr.Row():
                    llm_model = gr.Dropdown(label='LLM model', choices=list(self.options.models), value=self.options.default, interactive=True, allow_custom_value=True, elem_id='prompt_enhance_model')
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
                        do_sample = gr.Checkbox(label='Do sample', value=self.options.do_sample, interactive=True)
                    with gr.Row():
                        temperature = gr.Slider(label='Temperature', value=self.options.temperature, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        repetition_penalty = gr.Slider(label='Repetition penalty', value=self.options.repetition_penalty, minimum=0.0, maximum=2.0, step=0.01, interactive=True)
                    with gr.Row():
                        nsfw_mode = gr.Checkbox(label='NSFW allowed', value=True, interactive=True)
                        thinking_mode = gr.Checkbox(label='Thinking mode', value=False, interactive=True)
                    gr.HTML('<br>')
                with gr.Accordion('Input', open=False, elem_id='prompt_enhance_system_prompt'): # Corrected elem_id reference
                    with gr.Row():
                        prompt_prefix = gr.Textbox(label='Prompt prefix', value='', placeholder='Optional prompt prefix', interactive=True, lines=2, elem_id='prompt_enhance_prefix')
                    with gr.Row():
                        prompt_suffix = gr.Textbox(label='Prompt suffix', value='', placeholder='Optional prompt suffix', interactive=True, lines=2, elem_id='prompt_enhance_suffix')
                    with gr.Row():
                        prompt_system = gr.Textbox(label='System prompt', value='', interactive=True, lines=4, elem_id='prompt_enhance_system') # Default to empty as per diff
                with gr.Accordion('Output', open=True, elem_id='prompt_enhance_output'): # Corrected elem_id reference
                    with gr.Row():
                        prompt_output = gr.Textbox(label='Enhanced prompt', value='', interactive=True, lines=4)
                    with gr.Row():
                        clear_btn = gr.Button(value='Clear', elem_id='prompt_enhance_clear', variant='secondary')
                        clear_btn.click(fn=lambda: '', inputs=[], outputs=[prompt_output])
                        copy_btn = gr.Button(value='Set prompt', elem_id='prompt_enhance_copy', variant='secondary')
                        copy_btn.click(fn=lambda x: x, inputs=[prompt_output], outputs=[self.prompt])
            if self.image is None:
                self.image = gr.Image(type='pil', interactive=False, visible=False, width=64, height=64) # dummy image
            apply_btn.click(fn=self.apply, inputs=[self.prompt, self.image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, thinking_mode, nsfw_mode], outputs=[prompt_output, self.prompt])
        return [self.prompt, self.image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, thinking_mode, nsfw_mode]

    def after_component(self, component, **kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt', 'video_prompt']:
            self.prompt = component
            self.prompt.use_original = True
        if getattr(component, 'elem_id', '') in ['img2img_image', 'control_input_select']:
            self.image = component
            self.image.use_original = True

    def before_process(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=unused-argument
        _self_prompt, self_image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, max_tokens, do_sample, temperature, repetition_penalty, thinking_mode, nsfw_mode = args
        if not apply_auto and not p.enhance_prompt:
            return
        if shared.state.skipped or shared.state.interrupted:
            return
        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        shared.state.begin('LLM')
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
            thinking=thinking_mode,
            nsfw=nsfw_mode,
        )
        timer.process.record('prompt')
        p.extra_generation_params['LLM'] = llm_model
        shared.state.end()
