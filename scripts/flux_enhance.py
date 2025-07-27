# repo: https://huggingface.co/gokaygokay/Flux-Prompt-Enhance

import time
import random
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
from modules import shared, scripts_manager, devices, processing


repo_id = "gokaygokay/Flux-Prompt-Enhance"
num_return_sequences = 5
load_lock = threading.Lock()


class Script(scripts_manager.Script):
    prompts = [['']]
    tokenizer: AutoTokenizer = None
    model: AutoModelForSeq2SeqLM = None
    prefix: str = "enhance prompt: "
    button: gr.Button = None
    auto_apply: gr.Checkbox = None
    max_length: gr.Slider = None
    temperature: gr.Slider = None
    repetition_penalty: gr.Slider = None
    table: gr.DataFrame = None
    prompt: gr.Textbox = None

    def title(self):
        return 'Flux Prompt enhance (Legacy)'

    def show(self, is_img2img):
        return True

    def load(self):
        with load_lock:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained('gokaygokay/Flux-Prompt-Enhance', cache_dir=shared.opts.hfcache_dir)
            if self.model is None:
                shared.log.info(f'Prompt enhance: model="{repo_id}"')
                self.model = AutoModelForSeq2SeqLM.from_pretrained('gokaygokay/Flux-Prompt-Enhance', cache_dir=shared.opts.hfcache_dir).to(device=devices.cpu, dtype=devices.dtype)

    def enhance(self, prompt, auto_apply: bool = False, temperature: float = 0.7, repetition_penalty: float = 1.2, max_length: int = 128):
        self.load()
        t0 = time.time()
        input_text = self.prefix + prompt
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(devices.device)
        self.model = self.model.to(devices.device)
        kwargs = {
            'max_length': int(max_length),
            'num_return_sequences': int(num_return_sequences),
            'do_sample': True,
            'temperature': float(temperature),
            'repetition_penalty': float(repetition_penalty),
        }
        try:
            outputs = self.model.generate(input_ids, **kwargs)
        except Exception as e:
            shared.log.error(f'Prompt enhance: error="{e}"')
            return [['']]
        self.model = self.model.to(devices.cpu)
        prompts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prompts = [[p] for p in prompts]
        t1 = time.time()
        shared.log.info(f'Prompt enhance: temperature={temperature} repetition={repetition_penalty} length={max_length} sequences={num_return_sequences} apply={auto_apply} time={t1-t0:.2f}s')
        return prompts

    def select(self, cell: gr.SelectData, _table):
        prompt = cell.value if hasattr(cell, 'value') else cell
        shared.log.info(f'Prompt enhance: prompt="{prompt}"')
        return prompt

    def ui(self, _is_img2img):
        with gr.Row():
            self.button = gr.Button(value='Enhance prompt')
            self.auto_apply = gr.Checkbox(label='Auto apply', value=False)
        with gr.Row():
            self.max_length = gr.Slider(label='Length', minimum=64, maximum=512, step=1, value=128)
            self.temperature = gr.Slider(label='Temperature', minimum=0.1, maximum=2.0, step=0.05, value=0.7)
            self.repetition_penalty = gr.Slider(label='Penalty', minimum=0.1, maximum=2.0, step=0.05, value=1.2)
        with gr.Row():
            self.table = gr.DataFrame(self.prompts, label='', show_label=False, interactive=False, wrap=True, datatype="str", col_count=1, headers=['Prompts'])

        if self.prompt is not None:
            self.button.click(fn=self.enhance, inputs=[self.prompt, self.auto_apply, self.temperature, self.repetition_penalty, self.max_length], outputs=[self.table])
            self.table.select(fn=self.select, inputs=[self.table], outputs=[self.prompt])
        return [self.auto_apply, self.temperature, self.repetition_penalty, self.max_length]

    def run(self, p: processing.StableDiffusionProcessing, auto_apply, temperature, repetition_penalty, max_length): # pylint: disable=arguments-differ
        if auto_apply:
            p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
            p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
            shared.prompt_styles.apply_styles_to_extra(p)
            p.styles = []
            shared.log.debug(f'Prompt enhance: source="{p.prompt}"')
            prompts = self.enhance(p.prompt, auto_apply, temperature, repetition_penalty, max_length)
            p.prompt = random.choice(prompts)[0]
            shared.log.debug(f'Prompt enhance: prompt="{p.prompt}"')

    def after_component(self, component, **kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt', 'video_prompt']:
            self.prompt = component
            self.prompt.use_original = True
