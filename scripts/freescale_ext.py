import gradio as gr
from modules import scripts_manager, processing, shared, sd_models


registered = False


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.orig_slice = None
        self.orig_tile = None
        self.is_img2img = False

    def title(self):
        return 'FreeScale: Tuning-Free Scale Fusion'

    def show(self, is_img2img):
        self.is_img2img = is_img2img
        return shared.native

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/ali-vilab/FreeScale">&nbsp FreeScale: Tuning-Free Scale Fusion</a><br>')
        with gr.Row():
            cosine_scale = gr.Slider(minimum=0.1, maximum=5.0, value=2.0, label='Cosine scale')
            override_sampler = gr.Checkbox(value=True, label='Override sampler')
        with gr.Row(visible=self.is_img2img):
            cosine_scale_bg = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, label='Cosine Background')
            dilate_tau = gr.Slider(minimum=1, maximum=100, value=35, label='Dilate tau')
        with gr.Row():
            s1_enable = gr.Checkbox(value=True, label='1st Stage', interactive=False)
            s1_scale = gr.Slider(minimum=1, maximum=8.0, value=1.0, label='Scale')
            s1_restart = gr.Slider(minimum=0, maximum=1.0, value=0.75, label='Restart step')
        with gr.Row():
            s2_enable = gr.Checkbox(value=True, label='2nd Stage')
            s2_scale = gr.Slider(minimum=1, maximum=8.0, value=2.0, label='2nd Scale')
            s2_restart = gr.Slider(minimum=0, maximum=1.0, value=0.75, label='2nd Restart step')
        with gr.Row():
            s3_enable = gr.Checkbox(value=False, label='3rd Stage')
            s3_scale = gr.Slider(minimum=1, maximum=8.0, value=3.0, label='3rd Scale')
            s3_restart = gr.Slider(minimum=0, maximum=1.0, value=0.75, label='3rd Restart step')
        with gr.Row():
            s4_enable = gr.Checkbox(value=False, label='4th Stage')
            s4_scale = gr.Slider(minimum=1, maximum=8.0, value=4.0, label='4th Scale')
            s4_restart = gr.Slider(minimum=0, maximum=1.0, value=0.75, label='4th Restart step')
        return [cosine_scale, override_sampler, cosine_scale_bg, dilate_tau, s1_enable, s1_scale, s1_restart, s2_enable, s2_scale, s2_restart, s3_enable, s3_scale, s3_restart, s4_enable, s4_scale, s4_restart]

    def run(self, p: processing.StableDiffusionProcessing, cosine_scale, override_sampler, cosine_scale_bg, dilate_tau, s1_enable, s1_scale, s1_restart, s2_enable, s2_scale, s2_restart, s3_enable, s3_scale, s3_restart, s4_enable, s4_scale, s4_restart): # pylint: disable=arguments-differ
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'FreeScale: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None

        if self.is_img2img:
            if p.init_images is None or len(p.init_images) == 0:
                shared.log.warning('FreeScale: missing input image')
                return None

        from scripts.freescale import StableDiffusionXLFreeScale, StableDiffusionXLFreeScaleImg2Img
        self.orig_pipe = shared.sd_model
        self.orig_slice = shared.opts.diffusers_vae_slicing
        self.orig_tile = shared.opts.diffusers_vae_tiling

        def scale(x):
            if (p.width == 0 or p.height == 0) and p.init_images is not None:
                p.width, p.height = p.init_images[0].width, p.init_images[0].height
            resolution = [int(8 * p.width * x // 8), int(8 * p.height * x // 8)]
            return resolution

        scales = []
        resolutions_list = []
        restart_steps = []
        if s1_enable:
            scales.append(s1_scale)
            resolutions_list.append(scale(s1_scale))
            restart_steps.append(int(p.steps * s1_restart))
        if s2_enable and s2_scale > s1_scale:
            scales.append(s2_scale)
            resolutions_list.append(scale(s2_scale))
            restart_steps.append(int(p.steps * s2_restart))
        if s3_enable and s3_scale > s2_scale:
            scales.append(s3_scale)
            resolutions_list.append(scale(s3_scale))
            restart_steps.append(int(p.steps * s3_restart))
        if s4_enable and s4_scale > s3_scale:
            scales.append(s4_scale)
            resolutions_list.append(scale(s4_scale))
            restart_steps.append(int(p.steps * s4_restart))

        p.task_args['resolutions_list'] = resolutions_list
        p.task_args['cosine_scale'] = cosine_scale
        p.task_args['restart_steps'] = [min(max(1, step), p.steps-1) for step in restart_steps]
        if self.is_img2img:
            p.task_args['cosine_scale_bg'] = cosine_scale_bg
            p.task_args['dilate_tau'] = dilate_tau
            p.task_args['img_path'] = p.init_images[0]
            p.init_images = None
        if override_sampler:
            p.sampler_name = 'Euler a'

        if p.width < 1024 or p.height < 1024:
            shared.log.error(f'FreeScale: width={p.width} height={p.height} minimum=1024')
            return None

        if not self.is_img2img:
            shared.sd_model = sd_models.switch_pipe(StableDiffusionXLFreeScale, shared.sd_model)
        else:
            shared.sd_model = sd_models.switch_pipe(StableDiffusionXLFreeScaleImg2Img, shared.sd_model)
        shared.sd_model.enable_vae_slicing()
        shared.sd_model.enable_vae_tiling()

        shared.log.info(f'FreeScale: mode={"txt" if not self.is_img2img else "img"} cosine={cosine_scale} bg={cosine_scale_bg} tau={dilate_tau} scales={scales} resolutions={resolutions_list} steps={restart_steps} sampler={p.sampler_name}')
        resolutions = ','.join([f'{x[0]}x{x[1]}' for x in resolutions_list])
        steps = ','.join([str(x) for x in restart_steps])
        p.extra_generation_params["FreeScale"] = f'cosine {cosine_scale} resolutions {resolutions} steps {steps}'

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        # restore pipeline
        if shared.sd_model_type == "sdxl":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        if not self.orig_slice:
            shared.sd_model.disable_vae_slicing()
        if not self.orig_tile:
            shared.sd_model.disable_vae_tiling()
        return processed
