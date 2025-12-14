import diffusers
import gradio as gr
from sdnext_core import MODELDATA
from modules import scripts_manager, processing, shared, devices, sd_models


class Script(scripts_manager.Script):
    def title(self):
        return 'LEdits: Limitless Image Editing'

    def show(self, is_img2img):
        return is_img2img

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://leditsplusplus-project.static.hf.space/index.html">&nbsp LEdits++: Limitless Image Editing</a><br>')
        with gr.Row():
            edit_start = gr.Slider(label='Edit start', minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            edit_stop = gr.Slider(label='Edit stop', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            intersect_mask = gr.Checkbox(label='Smooth mask', value=True)
        with gr.Row():
            prompt1 = gr.Textbox(show_label=False, placeholder='Positive prompt')
            scale1 = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            threshold1 = gr.Slider(label='Threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.9)
        with gr.Row():
            prompt2 = gr.Textbox(show_label=False, placeholder='Negative prompt')
            scale2 = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            threshold2 = gr.Slider(label='Threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.9)
        return [edit_start, edit_stop, intersect_mask, prompt1, scale1, threshold1, prompt2, scale2, threshold2]

    def run(self, p: processing.StableDiffusionProcessing, edit_start, edit_stop, intersect_mask, prompt1, scale1, threshold1, prompt2, scale2, threshold2): # pylint: disable=arguments-differ, unused-argument
        image = getattr(p, 'init_images', None)
        if len(prompt1) == 0 and len(prompt2) == 0:
            shared.log.error('LEdits: no prompts')
            return None
        if image is None or len(image) == 0:
            shared.log.error('LEdits: no init_images')
            return None
        if MODELDATA.sd_model_type != 'sd' and MODELDATA.sd_model_type != 'sdxl':
            shared.log.error(f'LEdits: invalid model type: {MODELDATA.sd_model_type}')
            return None

        orig_pipeline = MODELDATA.sd_model
        orig_offload = shared.opts.diffusers_model_cpu_offload
        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['diffusers_model_cpu_offload'] = False
        shared.opts.data['prompt_attention'] = 'fixed'
        # shared.sd_model.maybe_free_model_hooks() # ledits is not compatible with offloading
        # shared.sd_model.has_accelerate = False
        sd_models.move_model(MODELDATA.sd_model, devices.device, force=True)
        if MODELDATA.sd_model_type == 'sd':
            MODELDATA.sd_model = sd_models.switch_pipe(diffusers.LEditsPPPipelineStableDiffusion, MODELDATA.sd_model)
        elif MODELDATA.sd_model_type == 'sdxl':
            MODELDATA.sd_model = sd_models.switch_pipe(diffusers.LEditsPPPipelineStableDiffusionXL, MODELDATA.sd_model)
        if str(devices.dtype) == 'torch.float16':
            MODELDATA.sd_model.vae.config.force_upcast = False # not compatible

        MODELDATA.sd_model.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(MODELDATA.sd_model.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2) # ledits is very picky
        p.sampler_name = 'Default'
        p.init() # run init early to take care of resizing

        invert_args = {
            'image': p.init_images[0],
            'source_prompt': p.prompt,
            'source_guidance_scale': p.cfg_scale,
            'num_inversion_steps': p.steps,
            'skip': 1.0 - p.denoising_strength, # invert start
            'generator': None, # not supported
        }
        shared.log.info(f'LEdits invert: {invert_args}')
        _output = MODELDATA.sd_model.invert(**invert_args)
        p.task_args = {
            'editing_prompt': [],
            'reverse_editing_direction': [],
            'edit_guidance_scale': [],
            'edit_threshold': [],
            'edit_warmup_steps': int(edit_start * p.steps),
            'edit_cooldown_steps': int((1.0 - edit_stop) * p.steps) if edit_stop < 1.0 else None,
            'use_intersect_mask': intersect_mask, # smoothing?
            'generator': None,
            'guidance_rescale': 0.0, # bug in pipeline if guidance rescale is enabled
        }
        if len(prompt1) > 0:
            p.task_args['editing_prompt'].append(prompt1)
            p.task_args['reverse_editing_direction'].append(False)
            p.task_args['edit_guidance_scale'].append(10.0 * scale1)
            p.task_args['edit_threshold'].append(threshold1)
        if len(prompt2) > 0:
            p.task_args['editing_prompt'].append(prompt2)
            p.task_args['reverse_editing_direction'].append(True)
            p.task_args['edit_guidance_scale'].append(10.0 * scale2)
            p.task_args['edit_threshold'].append(threshold2)

        shared.log.info(f'LEdits: {p.task_args}')
        processed = processing.process_images(p)

        # restore pipeline
        MODELDATA.sd_model = orig_pipeline
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.opts.data['diffusers_model_cpu_offload'] = orig_offload
        return processed
