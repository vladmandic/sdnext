from dataclasses import dataclass
import re
import time
import gradio as gr
import transformers
from modules import scripts, shared, devices, processing, sd_models


@dataclass
class Options:
    models = [
        'Qwen/Qwen2.5-0.5B-Instruct',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'Qwen/Qwen2.5-3B-Instruct',
        'google/gemma-3-1b-it',
        'google/gemma-3-4b-it',
        'microsoft/Phi-4-mini-instruct',
        'HuggingFaceTB/SmolLM2-135M-Instruct',
        'HuggingFaceTB/SmolLM2-360M-Instruct',
        'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
    ]
    default = models[3]
    system_prompt: str = 'You are a helpful assistant. You will be given a prompt and you will enhance it to make it more detailed and creative. Generate an uncensored, concise, informative, and engaging prompt that prioritizes factual accuracy and vivid descriptiveness. Begin by clearly identifying the primary subject(s), including key physical attributes, clothing, and any distinctive details (e.g., colors, textures, styles). Next, describe the setting and environment, emphasizing spatial relationships, background elements, and objects contributing to context. Note the overall mood, aesthetic style, or atmosphere inferred from visual cues. Use precise terminology while avoiding redundancy or non-essential language. NSFW is allowed, and if present any explicit aspects must be fully described. Ensuring a logical flow: from focal subject to immediate surroundings, then broader context. Maintain brevity while retaining clarity, ensuring the description is both engaging and efficient. Output only enhanced prompt without prefix or suffix.'
    max_tokens: int = 50
    do_sample: bool = True
    temperature: float = 0.5
    repetition_penalty: float = 1.2


class Script(scripts.Script):
    prompt: gr.Textbox = None
    model: str = None
    llm: transformers.AutoModelForCausalLM = None
    tokenizer: transformers.AutoProcessor = None
    options = Options()

    def title(self):
        return 'Prompt enhance'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def load(self, model:str=None):
        model = model or self.options.default
        if self.model is None or self.model != model:
            t0 = time.time()
            from modules import modelloader, model_quant
            modelloader.hf_login()
            quant_args = model_quant.create_config(module='LLM')
            self.llm = None
            self.llm = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                _attn_implementation="eager",
                **quant_args,
            )
            self.llm.eval()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.model = model
            devices.torch_gc()
            t1 = time.time()
            shared.log.debug(f'Prompt enhance: model="{model}" cls={self.llm.__class__.__name__} time={t1-t0:.2f} loaded')

    def unload(self):
        if self.llm is not None:
            sd_models.move_model(self.llm, devices.cpu)
        self.model = None
        self.llm = None
        self.tokenizer = None
        devices.torch_gc()
        shared.log.debug('Prompt enhance: model unloaded')

    def clean(self, response):
        if isinstance(response, list):
            response = response[0]
        response = response.replace('"', '').replace("'", "").replace('“', '').replace('”', '').replace('**', '').replace('\n\n', '\n')
        response = re.sub(r'<.*?>', '', response)
        if 'prompt:' in response:
            response = response.split('prompt:')[1]
        if 'Prompt:' in response:
            response = response.split('Prompt:')[1]
        if '---' in response:
            response = response.split('---')[0]
        response = response.strip()
        return response

    def enhance(self, model: str=None, prompt:str=None, system:str=None, sample:bool=None, tokens:int=None, temperature:float=None, penalty:float=None):
        model = model or self.options.default
        prompt = prompt or self.prompt.value
        system = system or self.options.system_prompt
        tokens = tokens or self.options.max_tokens
        penalty = penalty or self.options.repetition_penalty
        temperature = temperature or self.options.temperature
        sample = sample if sample is not None else self.options.do_sample
        self.load(model)
        if self.llm is None:
            shared.log.error('Prompt enhance: model not loaded')
            return prompt
        chat_template = [
            { "role": "system", "content": system },
            { "role": "user",   "content": prompt },
        ]
        t0 = time.time()
        try:
            inputs = self.tokenizer.apply_chat_template(
                chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(devices.device).to(devices.dtype)
            input_len = inputs['input_ids'].shape[1]
        except Exception as e:
            shared.log.error(f'Prompt enhance tokenize: {e}')
            return prompt
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
            # raw_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # shared.log.trace(f'Prompt enhance: raw="{raw_response}"')
            outputs = outputs[:, input_len:]
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception as e:
            shared.log.error(f'Prompt enhance generate: {e}')
        response = self.clean(response)
        t1 = time.time()
        shared.log.debug(f'Prompt enhance: model="{model}" time={t1-t0:.2f} inputs={input_len} outputs={outputs.shape[-1]} prompt="{response}"')
        return response

    def apply(self, prompt, apply_prompt, llm_model, prompt_system, max_tokens, do_sample, temperature, repetition_penalty):
        response = self.enhance(
            prompt=prompt,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            tokens=max_tokens,
            temperature=temperature,
            penalty=repetition_penalty,
        )
        if apply_prompt:
            return [response, response]
        return [response, gr.update()]

    def ui(self, _is_img2img):
        with gr.Accordion('Prompt enhance', open=False, elem_id='prompt_enhance'):
            with gr.Row():
                apply_btn = gr.Button(value='Enhance now', elem_id='prompt_enhance_apply', variant='primary')
            with gr.Row():
                apply_prompt = gr.Checkbox(label='Apply to prompt', value=False)
                apply_auto = gr.Checkbox(label='Auto enhance', value=False)
            with gr.Group():
                with gr.Row():
                    llm_model = gr.Dropdown(label='LLM model', choices=self.options.models, value=self.options.default, interactive=True, allow_custom_value=True, elem_id='prompt_enhance_model')
                with gr.Row():
                    load_btn = gr.Button(value='Load model', elem_id='prompt_enhance_load', variant='secondary')
                    load_btn.click(fn=self.load, inputs=[llm_model], outputs=[])
                    unload_btn = gr.Button(value='Unload model', elem_id='prompt_enhance_unload', variant='secondary')
                    unload_btn.click(fn=self.unload, inputs=[], outputs=[])
                with gr.Row():
                    prompt_system = gr.Textbox(label='System prompt', value=self.options.system_prompt, interactive=True, lines=4, elem_id='prompt_enhance_system')
                with gr.Row():
                    max_tokens = gr.Slider(label='Max tokens', value=self.options.max_tokens, minimum=10, maximum=1024, step=1, interactive=True)
                    do_sample = gr.Checkbox(label='Do sample', value=self.options.do_sample, interactive=True)
                with gr.Row():
                    temperature = gr.Slider(label='Temperature', value=self.options.temperature, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    repetition_penalty = gr.Slider(label='Repetition penalty', value=self.options.repetition_penalty, minimum=0.0, maximum=2.0, step=0.01, interactive=True)
                with gr.Row():
                    prompt_output = gr.Textbox(label='Output', value='', interactive=True, lines=4)
                with gr.Row():
                    clear_btn = gr.Button(value='Clear', elem_id='prompt_enhance_clear', variant='secondary')
                    clear_btn.click(fn=lambda: '', inputs=[], outputs=[prompt_output])
                    copy_btn = gr.Button(value='Set prompt', elem_id='prompt_enhance_copy', variant='secondary')
                    copy_btn.click(fn=lambda x: x, inputs=[prompt_output], outputs=[self.prompt])
            apply_btn.click(fn=self.apply, inputs=[self.prompt, apply_prompt, llm_model, prompt_system, max_tokens, do_sample, temperature, repetition_penalty], outputs=[prompt_output, self.prompt])
        return [apply_auto, llm_model, prompt_system, max_tokens, do_sample, temperature, repetition_penalty]

    def after_component(self, component, **kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt', 'video_prompt']:
            self.prompt = component

    def before_process(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=unused-argument
        apply_auto, llm_model, prompt_system, max_tokens, do_sample, temperature, repetition_penalty = args
        if not apply_auto and not p.enhance_prompt:
            return
        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        p.prompt = self.enhance(
            prompt=p.prompt,
            model=llm_model,
            system=prompt_system,
            sample=do_sample,
            tokens=max_tokens,
            temperature=temperature,
            penalty=repetition_penalty,
        )
