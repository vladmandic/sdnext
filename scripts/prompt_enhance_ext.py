import os
import re
import time
import random
import torch
import transformers
import gradio as gr
from PIL import Image
from modules import scripts_manager, shared, devices, errors, processing, sd_models, sd_modules, timer
from modules import ui_control_helpers
from modules.sd_offload_aux import register_aux, deregister_aux, move_aux_to_gpu, offload_aux
from modules.logger import log
from modules.caption.logits import LogitsParser
from modules.caption import helpers
from scripts.prompt_enhance.options import Options
from scripts.prompt_enhance.helpers import is_cloud_model, is_vision_model, is_thinking_model, get_model_repo_from_display
from scripts.prompt_enhance.template import set_template


debug_enabled = os.environ.get('SD_LLM_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None


class PromptEnhanceScript(scripts_manager.Script):
    prompt: gr.Textbox = None
    image: gr.Image = None
    model: str = None
    llm: transformers.AutoModelForCausalLM = None
    processor: transformers.AutoProcessor = None
    tokenizer: transformers.AutoTokenizer = None
    busy: bool = False
    server = None
    options = Options()
    video_capable = scripts_manager.AlwaysVisible

    def title(self):
        return 'Prompt enhance'

    def show(self, _is_img2img):
        return scripts_manager.AlwaysVisible

    def compile(self):
        if self.llm is None or 'LLM' not in shared.opts.cuda_compile:
            return
        from modules.sd_models_compile import compile_torch
        self.llm = compile_torch(self.llm, apply_to_components=False, op="LLM")

    def load(self, name:str | None=None, use_openai:bool=False, model_repo:str | None=None, model_gguf:str | None=None, model_type:str | None=None, model_file:str | None=None):
        # Strip symbols from display name if present
        name = get_model_repo_from_display(name) if name else self.options.default
        if self.busy:
            log.debug('Prompt enhance: busy')
            return model_repo
        if is_cloud_model(name):
            return model_repo
        if (self.model is not None) and (self.model == name):
            return model_repo

        model_repo = sd_models.path_to_repo(model_repo) if model_repo else None

        self.busy = True
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
                log.error(f'Prompt enhance: name="{name}" repo="{model_repo}" fn="{model_file}" type={model_type} gguf not supported')
                log.trace(f'Prompt enhance: gguf supported={self.options.supported}')
                self.busy = False
                return model_repo
            ggml.install_gguf()
            gguf_args['model_type'] = model_type
            gguf_args['gguf_file'] = model_file

        quant_args = model_quant.create_config(module='LLM', modules_to_not_convert=['conv1d', 'linear_attn.conv1d']) if not gguf_args else {}

        try:
            t0 = time.time()
            if self.llm is not None:
                deregister_aux('prompt_enhance')
                sd_models.move_model(self.llm, devices.cpu, force=True)
                self.llm = None
                self.tokenizer = None
                self.processor = None
                devices.torch_gc(force=True, reason='prompt-enhance:load')
                log.debug(f'Prompt enhance: unload="{self.model}"')
            self.model = None
            load_args = { 'pretrained_model_name_or_path': model_repo if not gguf_args else model_gguf }
            if model_subfolder:
                load_args['subfolder'] = model_subfolder # Comma was incorrect here

            model_config = transformers.AutoConfig.from_pretrained(load_args['pretrained_model_name_or_path'], trust_remote_code=True, cache_dir=shared.opts.hfcache_dir)
            model_type = getattr(model_config, 'model_type', '')
            cls_name = transformers.AutoModelForCausalLM
            custom_cls_name = self.options.models_cls.get(model_type, None)
            if custom_cls_name:
                custom_cls = getattr(transformers, custom_cls_name, None)
                if custom_cls:
                    cls_name = custom_cls

            log.info(f'Prompt enhance load: name="{name}" repo="{model_repo}" cls={cls_name.__name__}')

            if '-ct' in model_repo.lower():
                from installer import install
                install('compressed-tensors')
                quant_args = {}

            sd_models.set_caption_load_options()
            try:
                self.llm = cls_name.from_pretrained(
                    **load_args,
                    trust_remote_code=True,
                    torch_dtype=devices.dtype,
                    low_cpu_mem_usage=True,
                    cache_dir=shared.opts.hfcache_dir,
                    # _attn_implementation="eager",
                    **gguf_args,
                    **quant_args,
                )
            finally:
                sd_models.set_huggingface_options(quiet=True)

            self.llm.eval()
            register_aux('prompt_enhance', self.llm)
            tokenizer_args = { 'pretrained_model_name_or_path': model_repo }
            if model_tokenizer:
                tokenizer_args['subfolder'] = model_tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(**tokenizer_args, cache_dir=shared.opts.hfcache_dir)
            if model_repo in self.options.img2img:
                self.processor = transformers.AutoProcessor.from_pretrained(**tokenizer_args, cache_dir=shared.opts.hfcache_dir)

            if debug_enabled:
                modules = sd_modules.get_model_stats(self.llm) + sd_modules.get_model_stats(self.tokenizer)
                for m in modules:
                    debug_log(f'Prompt enhance: {m}')
            self.model = name
            t1 = time.time()
            log.debug(f'Prompt enhance: cls={self.llm.__class__.__name__} name="{name}" repo="{model_repo}" fn="{model_file}" processor="{self.processor.__class__.__name__ if self.processor else None}" tokenizer="{self.tokenizer.__class__.__name__ if self.tokenizer else None}" module={self.parent} time={t1-t0:.2f} loaded')
            self.compile()
        except Exception as e:
            log.error(f'Prompt enhance: load {e}')
            errors.display(e, 'Prompt enhance')

        devices.torch_gc()
        self.set_openai(enable=use_openai)
        self.busy = False
        return model_repo

    def censored(self, response):
        text = response.lower().replace("i'm", "i am")
        return any(c.lower() in text for c in self.options.censored)

    def unload(self):
        if self.llm is not None:
            model_name = self.model
            self.set_openai(enable=False)
            log.debug(f'Prompt enhance: unloading model="{model_name}"')
            deregister_aux('prompt_enhance')
            sd_models.move_model(self.llm, devices.cpu, force=True)
            self.model = None
            self.llm = None
            self.tokenizer = None
            self.processor = None
            devices.torch_gc(force=True, reason='prompt-enhance:unload')
            log.debug(f'Prompt enhance: model="{model_name}" unloaded')
        else:
            log.debug('Prompt enhance: no model loaded')

    def set_openai(self, enable: bool):
        from modules.openai.serve import OpenAIServer
        if enable and self.llm is not None and self.tokenizer is not None:
            self.server = OpenAIServer(
                model=self.llm,
                tokenizer=self.tokenizer,
                host="127.0.0.1",
                port=8000,
                server=shared.api.app,
            )
            self.server.start()
        elif self.server is not None:
            self.server.stop()
            self.server = None

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
        # remove repeating characters and short repeated tokens from model collapse
        response = response.replace('\n\n', '\n').replace('  ', ' ').replace('...', '.')
        response = re.sub(r'\b([A-Za-z]{1,3})(?:\s+\1){1,}\b', r'\1', response, flags=re.IGNORECASE)

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

        # Remove leading conversational filler that some LLMs prepend
        response = re.sub(
            r'^(?:\s*(?:wait|okay|ok|sure|alright|yes|yep|hello|hi|thanks|thank you|no problem|of course|got it|i love|i like|i appreciate|great|excellent|right)[^.!?]*[.!?]\s*)+',
            '',
            response,
            flags=re.IGNORECASE,
        )

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

    def get_image(self, image):
        current_image = None
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
        return current_image

    def enhance(self,
                model: str | None=None,
                prompt:str | None=None,
                system:str | None=None,
                prefix:str | None=None,
                suffix:str | None=None,
                sample:bool | None=None,
                min_tokens:int | None=None,
                max_tokens:int | None=None,
                temperature:float | None=None,
                penalty:float | None=None,
                top_k:int | None=None,
                top_p:float | None=None,
                thinking:bool=False,
                seed:int=-1,
                image=None,
                nsfw:bool | None=None,
                use_vision:bool=True,
                prefill:str='',
                keep_prefill:bool=False,
                keep_thinking:bool=False,
                custom_args:str | None=None,
                process_words:str='',
                semantic_threshold:float=0.0,
                embedding_similarity:float=0.0,
                use_openai:bool=False,
               ):
        # Strip symbols from model name if present
        model = get_model_repo_from_display(model) if model else self.options.default
        prompt = prompt or (self.prompt.value if self.prompt else "") # Check if self.prompt is None
        image = None
        if use_vision and is_vision_model(model): # handle vision toggle
            image = image or self.image
        if image is None:
            use_vision = False
        prefix = prefix or ''
        suffix = suffix or ''
        min_tokens = min_tokens or self.options.min_tokens
        max_tokens = max_tokens or self.options.max_tokens
        penalty = penalty or self.options.repetition_penalty
        temperature = temperature or self.options.temperature
        top_k = top_k if top_k is not None else self.options.top_k
        top_p = top_p if top_p is not None else self.options.top_p
        thinking = thinking or self.options.thinking_mode
        sample = sample if sample is not None else self.options.do_sample
        nsfw = nsfw if nsfw is not None else True # Default nsfw to True if not provided
        debug_log(f'Prompt enhance: model="{model}" model_class="{self.llm.__class__.__name__ if self.llm is not None else "not loaded"}" nsfw={nsfw} thinking={thinking} prefill="{prefill[:30] if prefill else ""}" use_vision={use_vision} image={image is not None}')

        while self.busy:
            time.sleep(0.1)

        if not is_cloud_model(model):
            self.load(model, use_openai=use_openai)

        if seed is None or seed == -1:
            random.seed()
            seed = int(random.randrange(4294967294))
        torch.manual_seed(seed)
        if (self.llm is None) and (not is_cloud_model(model)):
            log.error('Prompt enhance: model not loaded')
            return prompt
        prompt_text, networks = self.extract(prompt) # Use prompt_text after extraction
        debug_log(f'Prompt enhance: networks={networks}')

        current_image = None
        # Only process images if vision is enabled and model supports it
        if use_vision and is_vision_model(model):
            current_image = self.get_image(image)
        debug_log(f'Prompt enhance: image={current_image}')

        # Check if vision was requested but no image is available
        if use_vision and is_vision_model(model) and current_image is None:
            log.error(f'Prompt enhance: model="{model}" error="No input image provided"')
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

        chat_template = set_template(
            system=system,
            prompt=prompt_text,
            image=current_image,
            options=self.options,
            model=model,
            nsfw=nsfw,
            has_processor=self.processor is not None,
            module=self.parent,
        )

        # Prepare prefill (VQA approach: string concatenation, not assistant message)
        prefill_text = (prefill or '').strip()
        use_prefill = len(prefill_text) > 0
        is_thinking = is_thinking_model(model)

        debug_log(f'Prompt enhance: system="{system}"')
        debug_log(f'Prompt enhance: prompt="{prompt_text}"')
        debug_log(f'Prompt template: roles={[msg["role"] for msg in chat_template]} thinking={is_thinking}:{thinking} prefill={use_prefill}')
        t0 = time.time()
        self.busy = True

        if is_cloud_model(model):
            if 'gemini' in model:
                from modules.caption import gemini
                kwargs = {
                    'temperature': temperature,
                    'min_output_tokens': min_tokens,
                    'max_output_tokens': max_tokens,
                }
                model_name = model.replace('google/', '')
                response = gemini.predict(prompt_text, current_image, model_name, system, model, prefill_text, thinking, kwargs)
                t1 = time.time()
                log.info(f'Prompt enhance: model="{model}" nsfw={nsfw} time={t1-t0:.2f} prefill="{prefill_text[:20] if prefill_text else None}" response={len(response)}')
                debug_log(f'Prompt enhance: response="{response}"')
                self.busy = False
                return response

            else:
                return 'Model not recognized'

        try:
            # Qwen3.5 uses native enable_thinking parameter in the chat template
            is_qwen35 = 'qwen3.5' in model.lower()
            template_kwargs = {'enable_thinking': thinking} if is_qwen35 else {}

            # Generate text prompt using template
            apply_fn = self.processor if self.processor is not None else self.tokenizer
            try:
                text_prompt = apply_fn.apply_chat_template(
                    chat_template,
                    add_generation_prompt=True,
                    tokenize=False,
                    **template_kwargs,
                )
            except TypeError:
                text_prompt = apply_fn.apply_chat_template(
                    chat_template,
                    tokenize=False,
                )

            # Manual think handling - skip for Qwen3.5 (template handles it natively)
            if is_thinking and not is_qwen35:
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
                # Standard model or Qwen3.5 (no manual <think> manipulation needed)
                if use_prefill:
                    text_prompt += prefill_text

            # debug_log(f'Prompt enhance: template="{text_prompt}"')

            # Tokenize the final prompt
            # For VL models with images, pass the image to the processor (like VQA does)
            if self.processor is not None and current_image is not None:
                inputs = self.processor(text=[text_prompt], images=[current_image], padding=True, return_tensors="pt")
            elif self.processor is not None:
                # VL processor without image - must use explicit text= parameter
                inputs = self.processor(text=[text_prompt], images=None, padding=True, return_tensors="pt")
            else:
                inputs = self.tokenizer(text_prompt, return_tensors="pt")
            inputs = inputs.to(devices.device).to(devices.dtype)
            input_len = inputs['input_ids'].shape[1]
        except Exception as e:
            log.error(f'Prompt enhance tokenize: {e}')
            if debug_enabled:
                errors.display(e, 'Prompt enhance')
            self.busy = False
            return prompt_text # Return original text part on error

        try:
            with devices.llm_context():
                move_aux_to_gpu('prompt_enhance')
                gen_kwargs = {
                    'do_sample': sample,
                    'temperature': float(temperature),
                    'max_new_tokens': int(max_tokens),
                    'repetition_penalty': float(penalty),
                }
                if min_tokens > 0:
                    gen_kwargs['min_new_tokens'] = int(min_tokens)
                if top_k > 0:
                    gen_kwargs['top_k'] = int(top_k)
                if top_p > 0:
                    gen_kwargs['top_p'] = float(top_p)

                logits_processor = None
                if process_words is not None and len(process_words.strip()) > 0 and self.tokenizer is not None:
                    logits_processor = LogitsParser(self.tokenizer, process_words, semantic_threshold=semantic_threshold, embedding_similarity=embedding_similarity)
                    gen_kwargs['logits_processor'] = [logits_processor]

                custom = helpers.get_custom_args(self.llm, custom_args)
                for k, v in custom.items():
                    gen_kwargs[k] = v

                log.debug(f'Prompt enhance: cls={self.llm.__class__.__name__} model="{model}" tokens={input_len} args={gen_kwargs} custom={custom}')
                defaults = {k: v for k, v in helpers.get_default_args(self.llm).items() if k not in gen_kwargs}
                debug_log(f'Prompt enhance: defaults={defaults}')

                outputs = self.llm.generate(**inputs, **gen_kwargs)

                if logits_processor is not None:
                    log.debug(f'Prompt enhance: process={logits_processor.get_replacements()}')

            outputs_cropped = outputs[:, input_len:]
            decode_fn = self.processor if self.processor is not None else self.tokenizer
            response = decode_fn.batch_decode(
                outputs_cropped,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if debug_enabled:
                response_before_clean = response[0] if isinstance(response, list) else response
                debug_log(f'Prompt enhance: response_before_clean="{response_before_clean}"')
        except Exception as e:
            outputs = None
            log.error(f'Prompt enhance generate: {e}')
            errors.display(e, 'Prompt enhance')
            self.busy = False
            response = f'Error: {str(e)}'
        finally:
            offload_aux('prompt_enhance')
            devices.torch_gc(force=False, reason='prompt-enhance')
        t1 = time.time()

        if isinstance(response, list):
            response = response[0]
        is_censored =  self.censored(response)
        if not is_censored:
            response = self.clean(response, keep_thinking=keep_thinking, prefill_text=prefill_text, keep_prefill=keep_prefill)
            response = self.post(response, prefix, suffix, networks)
        log.info(f'Prompt enhance: model="{model}" nsfw={nsfw} time={t1-t0:.2f} seed={seed} thinking={thinking} keep={keep_thinking}:{keep_prefill} prefill="{prefill_text[:20] if prefill_text else None}" inputs={input_len} outputs={outputs.shape[-1] if isinstance(outputs, torch.Tensor) else 0} prompt={len(prompt_text)} response={len(response)}')
        debug_log(f'Prompt enhance: prompt="{prompt_text}"')
        debug_log(f'Prompt enhance: response_after_clean="{response}"')
        self.busy = False
        if is_censored:
            log.warning(f'Prompt enhance: censored response="{response}"')
            return prompt # Return original full prompt on censorship
        return response

    def apply(self, prompt, image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, min_tokens, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking, custom_args, process_words, semantic_threshold, embedding_similarity, use_openai):
        response = self.enhance(
            prompt=prompt,
            image=image,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
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
            custom_args=custom_args,
            process_words=process_words,
            semantic_threshold=semantic_threshold,
            embedding_similarity=embedding_similarity,
            use_openai=use_openai,
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
        if not is_vl:
            return gr.update(interactive=False, value=False)
        return gr.update(interactive=is_vl)

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
                use_vision = gr.Checkbox(label='Use vision', value=False, interactive=default_is_vl, elem_id='prompt_enhance_use_vision')
                use_openai = gr.Checkbox(label='OpenAI interface', value=False, elem_id='prompt_enhance_openai')
            gr.HTML('<br>')
            with gr.Group():
                with gr.Row():
                    llm_model = gr.Dropdown(label='LLM model', choices=Options.get_model_choices(), value=Options.get_default_display(), interactive=True, allow_custom_value=True, elem_id='prompt_enhance_model')
                with gr.Row():
                    load_btn = gr.Button(value='Load model', elem_id='prompt_enhance_load', variant='secondary')
                    load_btn.click(fn=self.load, inputs=[llm_model, use_openai], outputs=[])
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
                        custom_btn.click(fn=self.load, inputs=[model_repo, use_openai, model_repo, model_gguf, model_type, model_file], outputs=[llm_model])
                        llm_model.change(fn=self.get_custom, inputs=[llm_model], outputs=[model_repo, model_gguf, model_type, model_file])
                        gr.HTML('<br>')
                with gr.Accordion('Options', open=False, elem_id='prompt_enhance_options'):
                    with gr.Row():
                        min_tokens = gr.Slider(label='Min tokens', value=self.options.min_tokens, minimum=0, maximum=4096, step=1, interactive=True)
                        max_tokens = gr.Slider(label='Max tokens', value=self.options.max_tokens, minimum=10, maximum=4096, step=1, interactive=True)
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
                        custom_args = gr.Textbox(label='Custom arguments', value='', placeholder='Optional: custom arguments for the model as k=v, semicolon delimited', interactive=True, lines=1)
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
                with gr.Accordion('Process', open=False, elem_id='prompt_enhance_logits'): # Corrected elem_id reference
                    with gr.Row():
                        process_words = gr.Textbox(label='Words to process', value='', placeholder='list of words with optional substitutions', interactive=True, lines=3, elem_id='prompt_enhance_process_words')
                    with gr.Row():
                        semantic_threshold = gr.Slider(label='Semantic threshold', value=0.0, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id='prompt_enhance_semantic_threshold')
                        embedding_similarity = gr.Slider(label='Embedding similarity', value=0.0, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id='prompt_enhance_embedding_similarity')
                with gr.Accordion('Output', open=True, elem_id='prompt_enhance_output'): # Corrected elem_id reference
                    with gr.Row():
                        prompt_output = gr.Textbox(label='Enhanced prompt', value='', placeholder='Enhanced prompt will appear here', interactive=True, lines=4, max_lines=12, elem_id='prompt_enhance_result')
                    with gr.Row():
                        clear_btn = gr.Button(value='Clear', elem_id='prompt_enhance_clear', variant='secondary')
                        clear_btn.click(fn=lambda: '', inputs=[], outputs=[prompt_output])
                        copy_btn = gr.Button(value='Set prompt', elem_id='prompt_enhance_copy', variant='secondary')
                        if self.prompt: # not registered for api script runner
                            copy_btn.click(fn=lambda x: x, inputs=[prompt_output], outputs=[self.prompt])
            if self.image is None:
                self.image = gr.Image(type='pil', interactive=False, visible=False, width=64, height=64) # dummy image
            # Update vision toggle interactivity when model changes
            llm_model.change(fn=self.update_vision_toggle, inputs=[llm_model], outputs=[use_vision], show_progress=False)
            if self.prompt:
                apply_btn.click(fn=self.apply, inputs=[self.prompt, self.image, apply_prompt, llm_model, prompt_system, prompt_prefix, prompt_suffix, min_tokens, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking, custom_args, process_words, semantic_threshold, embedding_similarity, use_openai], outputs=[prompt_output, self.prompt])
        return [self.prompt, self.image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, min_tokens, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking, custom_args, process_words, semantic_threshold, embedding_similarity, use_openai]

    def after_component(self, component, **_kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt', 'video_prompt']:
            self.prompt = component
            self.prompt.use_original = True
        if getattr(component, 'elem_id', '') in ['img2img_image', 'control_input_select']:
            self.image = component
            self.image.use_original = True

    def before_process(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=unused-argument
        _self_prompt, self_image, apply_auto, llm_model, prompt_system, prompt_prefix, prompt_suffix, min_tokens, max_tokens, do_sample, temperature, repetition_penalty, top_k, top_p, thinking_mode, nsfw_mode, use_vision, prefill_text, keep_prefill, keep_thinking, custom_args, process_words, semantic_threshold, embedding_similarity, use_openai = args
        if not apply_auto and not p.enhance_prompt:
            return
        if shared.state.skipped or shared.state.interrupted:
            return
        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        jobid = shared.state.begin('LLM')
        p.extra_generation_params['LLM'] = get_model_repo_from_display(llm_model)
        p.extra_generation_params['Original'] = p.prompt
        p.prompt = self.enhance(
            prompt=p.prompt,
            seed=p.seed,
            image=self_image,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
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
            custom_args=custom_args,
            process_words=process_words,
            semantic_threshold=semantic_threshold,
            embedding_similarity=embedding_similarity,
            use_openai=use_openai,
        )
        timer.process.record('prompt')
        shared.state.end(jobid)
