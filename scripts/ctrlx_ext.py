# https://github.com/genforce/ctrl-x

import gradio as gr
from diffusers import StableDiffusionXLPipeline
from modules import shared, scripts_manager, processing, processing_helpers, sd_models, devices


class Script(scripts_manager.Script):
    def title(self):
        return 'Ctrl-X: Controlling Structure and Appearance'

    def show(self, is_img2img):
        return True

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/genforce/ctrl-x">&nbsp Ctrl-X: Controlling Structure and Appearance</a><br>')
        with gr.Accordion(label='Structure', open=True):
            with gr.Row():
                struct_prompt = gr.Textbox(label='Prompt', value='')
            with gr.Row():
                struct_strength = gr.Slider(label='Strength', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                struct_guidance = gr.Slider(label='Guidance', value=5.0, minimum=0.0, maximum=14.0, step=0.05)
            with gr.Row():
                struct_image = gr.Image(label='Image', type='pil')
        with gr.Accordion(label='Appearance', open=True):
            with gr.Row():
                appear_prompt = gr.Textbox(label='Prompt', value='')
            with gr.Row():
                appear_strength = gr.Slider(label='Strength', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                appear_guidance = gr.Slider(label='Guidance', value=5.0, minimum=0.0, maximum=14.0, step=0.05)
            with gr.Row():
                appear_image = gr.Image(label='Image', type='pil')
        return struct_prompt, struct_strength, struct_guidance, struct_image, appear_prompt, appear_strength, appear_guidance, appear_image

    def restore(self):
        del shared.sd_model.restore_pipeline
        shared.sd_model = sd_models.switch_pipe(StableDiffusionXLPipeline, shared.sd_model, force=True)

    def run(self, p: processing.StableDiffusionProcessing, struct_prompt, struct_strength, struct_guidance, struct_image, appear_prompt, appear_strength, appear_guidance, appear_image): # pylint: disable=arguments-differ
        c = shared.sd_model.__class__.__name__ if shared.sd_loaded else ''
        if shared.sd_model_type != 'sdxl':
            shared.log.warning(f'Ctrl-X: pipeline={c} required=StableDiffusionXLPipeline')
            return None

        import yaml
        from scripts.ctrlx import CtrlXStableDiffusionXLPipeline # pylint: disable=no-name-in-module
        from scripts.ctrlx.sdxl import get_control_config, register_control # pylint: disable=no-name-in-module
        from scripts.ctrlx.utils import get_self_recurrence_schedule # pylint: disable=no-name-in-module

        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'fixed'
        shared.sd_model = sd_models.switch_pipe(CtrlXStableDiffusionXLPipeline, shared.sd_model)
        shared.sd_model.restore_pipeline = self.restore

        # calculate ctrx+x schedule
        if p.sampler_name not in ['DDIM', 'Euler', 'Euler a', 'DPM++ 1S', 'DDPM', 'Euler SGM', 'LCM', 'TCD']:
            shared.log.warning(f'Ctrl-X: sampler={p.sampler_name} override="Euler a" supported=[Euler, Euler a, Euler SGM, DDIM, DDPM, , LCM, TCD]')
            p.sampler_name = 'Euler a'
        processing_helpers.update_sampler(p, shared.sd_model)
        shared.sd_model.scheduler.set_timesteps(p.steps, device=devices.device)
        timesteps = shared.sd_model.scheduler.timesteps
        control_config = get_control_config(structure_schedule=struct_strength, appearance_schedule=appear_strength)
        config = yaml.safe_load(control_config)
        register_control(
            model=shared.sd_model,
            timesteps=timesteps,
            control_schedule=config['control_schedule'],
            control_target=config['control_target'],
        )

        # set args
        if struct_image is not None:
            p.task_args['structure_prompt'] = struct_prompt
            p.task_args['structure_image'] = struct_image
            p.task_args['structure_guidance_scale'] = struct_guidance
        if appear_image is not None:
            p.task_args['appearance_prompt'] = appear_prompt
            p.task_args['appearance_image'] = appear_image
            p.task_args['appearance_guidance_scale'] = appear_guidance
        elif hasattr(p, 'init_images') and p.init_images is not None and len(p.init_images) > 0:
            p.task_args['appearance_image'] = p.init_images[0]
            p.init_images = None
        p.task_args['control_schedule'] = config['control_schedule']
        p.task_args['self_recurrence_schedule'] = get_self_recurrence_schedule(config['self_recurrence_schedule'], p.steps)
        is_struct = p.task_args.get('structure_image') is not None
        is_appear = p.task_args.get('appearance_image') is not None
        shared.log.info(f'Ctrl-X: structure={struct_strength if is_struct else None} appearance={appear_strength if is_appear else None}')
        shared.log.debug(f'Ctrl-X: config={control_config} args={p.task_args}')

        # process
        processed: processing.Processed = processing.process_images(p)

        # restore and return
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.sd_model = sd_models.switch_pipe(StableDiffusionXLPipeline, shared.sd_model, force=True)
        return processed
