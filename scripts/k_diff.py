import inspect
import gradio as gr
from modules import scripts, processing, shared, sd_models, ui_common


class Script(scripts.Script):
    supported_models = ['sd', 'sdxl']
    orig_pipe = None

    def title(self):
        return 'K-Diffusion Samplers'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/crowsonkb/k-diffusion">&nbsp K-Diffusion Samplers</a><br>')
        with gr.Row():
            sampler = gr.Dropdown(label="Sampler", choices=[])
            _refresh = ui_common.create_refresh_button(sampler, self.samplers, lambda: {"choices": self.samplers()})
        return [sampler]

    def samplers(self):
        samplers = []
        from modules import sd_samplers_kdiffusion
        for s in dir(sd_samplers_kdiffusion.k_sampling):
            if s.startswith('sample_'):
                samplers.append(s.replace('sample_', ''))
        return samplers

    def callback(self, d):
        _step = d['i']

    def run(self, p: processing.StableDiffusionProcessing, sampler: str): # pylint: disable=arguments-differ
        if sampler is None or len(sampler) == 0:
            return None
        if shared.sd_model_type not in self.supported_models:
            shared.log.warning(f'K-Diffusion: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={self.supported_models}')
            return None
        import diffusers
        cls = None
        if shared.sd_model_type == "sd":
            cls = diffusers.pipelines.StableDiffusionKDiffusionPipeline
        if shared.sd_model_type == "sdxl":
            cls = diffusers.pipelines.StableDiffusionXLKDiffusionPipeline
        if cls is None:
            return None
        from modules import sd_samplers_kdiffusion

        sampler_fn = getattr(sd_samplers_kdiffusion.k_sampling, f'sample_{sampler}', None)
        if sampler_fn is None:
            shared.log.warning(f'K-Diffusion: sampler={sampler} not found')
            return None

        self.orig_pipe = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(cls, shared.sd_model)
        shared.sd_model.sampler = sampler_fn

        params = inspect.signature(shared.sd_model.sampler).parameters.values()
        params = {param.name: param.default for param in params if param.default != inspect.Parameter.empty}
        # if 'callback' in list(params):
        #     params['callback'] = self.callback
        # if 'disable' in list(params):
        #     params['disable'] = False
        shared.log.info(f'K-diffusion apply: class={shared.sd_model.__class__.__name__} sampler={sampler} params={params}')
        p.extra_generation_params["Sampler"] = sampler

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, sampler): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        if shared.sd_model_type == "sdxl" or shared.sd_model_type == "sd":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        return processed
