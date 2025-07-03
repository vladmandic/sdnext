# https://huggingface.co/ByteDance/InfiniteYou
# https://github.com/bytedance/InfiniteYou
# flux base model + 11.8gb controlnet module + 338mb image module + 428 insightface module

import gradio as gr
from PIL import Image
from modules import scripts_manager, processing, shared, sd_models, devices


prefix = 'InfiniteYou'
model_versions = ['aes_stage2', 'sim_stage1']
orig_pipeline, orig_prompt_attention = None, None


def verify_insightface():
    from installer import installed, install, reload
    if not installed('insightface', reload=False, quiet=True):
        install('git+https://github.com/deepinsight/insightface@554a05561cb71cfebb4e012dfea48807f845a0c2#subdirectory=python-package', 'insightface') # insightface==0.7.3 with patches
        install('albumentations==1.4.3', ignore=False, reinstall=True)
        install('pydantic==1.10.21', ignore=False, reinstall=True, force=True)
        reload('pydantic')


def load_infiniteyou(model: str):
    from scripts.infiniteyou import InfUFluxPipeline
    shared.sd_model = InfUFluxPipeline(
        pipe=shared.sd_model,
        model_version=model,
    )
    sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline)
    sd_models.set_diffuser_options(shared.sd_model)


class Script(scripts_manager.Script):
    def title(self):
        return f'{prefix}: Flexible Photo Recrafting'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML(f'<a href="https://github.com/bytedance/InfiniteYou">&nbsp {prefix}: Flexible Photo Recrafting</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='IY model', choices=model_versions, value=model_versions[0])
            restore = gr.Checkbox(label='Restore pipeline on end', value=False)
        with gr.Row():
            scale = gr.Slider(label='IY scale', value=1.0, minimum=0.0, maximum=2.0, step=0.05)
        with gr.Row():
            start = gr.Slider(label='IY start', value=0.0, minimum=0.0, maximum=1.0, step=0.05)
            end = gr.Slider(label='IY end', value=1.0, minimum=0.0, maximum=1.0, step=0.05)
        with gr.Row():
            id_guidance = gr.Slider(label='Identity guidance', value=3.5, minimum=0.0, maximum=14.0, step=0.05)
        with gr.Row():
            id_image = gr.Image(label='Identity image', type='pil')
        with gr.Row():
            control_guidance = gr.Slider(label='Control guidance', value=1.0, minimum=0.0, maximum=14.0, step=0.05)
        with gr.Row():
            control_image = gr.Image(label='Control image', type='pil')
        return [model, id_image, control_image, scale, start, end, id_guidance, control_guidance, restore]

    def run(self, p: processing.StableDiffusionProcessing,
            model: str = None,
            id_image: Image.Image = None,
            control_image: Image.Image = None,
            scale: float = 1.0,
            start: float = 0.0,
            end: float = 1.0,
            id_guidance: float = 3.5,
            control_guidance: float = 1.0,
            restore: bool = False,
        ): # pylint: disable=arguments-differ, unused-argument

        if model is None or model not in model_versions:
            return None
        if id_image is None:
            shared.log.error(f'{prefix}: no init_images')
            return None
        if shared.sd_model_type != 'f1':
            shared.log.error(f'{prefix}: invalid model type: {shared.sd_model_type}')
            return None
        if scale <= 0:
            return None

        global orig_pipeline, orig_prompt_attention # pylint: disable=global-statement
        orig_pipeline = shared.sd_model
        if shared.sd_model.__class__.__name__ != 'InfUFluxPipeline':
            verify_insightface()
            load_infiniteyou(model)
            devices.torch_gc()
            shared.log.info(f'{prefix}: cls={shared.sd_model.__class__.__name__} loaded')

        processing.fix_seed(p)
        p.task_args['id_image'] = id_image
        p.task_args['control_image'] = control_image
        p.task_args['infusenet_conditioning_scale'] = p.task_args.get('infusenet_conditioning_scale', scale)
        p.task_args['infusenet_guidance_start'] = p.task_args.get('infusenet_guidance_start', start)
        p.task_args['infusenet_guidance_end'] = p.task_args.get('infusenet_guidance_end', end)
        p.task_args['seed'] = p.seed
        p.task_args['negative_prompt'] = None
        p.task_args['guidance_scale'] = id_guidance
        p.task_args['controlnet_guidance_scale'] = control_guidance
        p.extra_generation_params['IY model'] = model
        p.extra_generation_params['IY guidance'] = f'{scale:.1f}/{start:.1f}/{end:.1f}'
        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'fixed'
        shared.log.debug(f'{prefix}: args={p.task_args}')

        processed = processing.process_images(p)
        return processed

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args, **kwargs): # pylint: disable=unused-argument
        # restore pipeline
        global orig_pipeline, orig_prompt_attention # pylint: disable=global-statement
        restore = args[-1]
        if orig_prompt_attention is not None:
            shared.opts.data['prompt_attention'] = orig_prompt_attention
            orig_prompt_attention = None
        if restore and orig_pipeline is not None:
            shared.log.info(f'{prefix}: restoring pipeline')
            shared.sd_model = orig_pipeline
            orig_pipeline = None
