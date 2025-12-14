import gradio as gr
import diffusers
from sdnext_core import MODELDATA
from modules import scripts_manager, processing, shared, sd_models, devices


class Script(scripts_manager.Script):
    def title(self):
        return 'Kohya HiRes Fix'

    def show(self, is_img2img):
        return not is_img2img

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/huggingface/diffusers/pull/7633">&nbsp Kohya HiRes Fix</a><br>')
        with gr.Row():
            enabled = gr.Checkbox(label="Enabled", value=True)
        with gr.Row():
            scale_factor = gr.Slider(value=0.5, minimum=0, maximum=1, step=0.05, label="Scale factor")
            timestep = gr.Number(value=600, minimum=0, maximum=1000, label="Timestep")
            block_num = gr.Number(value=1, minimum=0, maximum=10, label="Block")
        return [enabled, scale_factor, timestep, block_num]

    def run(self, p: processing.StableDiffusionProcessing, enabled, scale_factor, timestep, block_num): # pylint: disable=arguments-differ
        if not enabled:
            return None
        if MODELDATA.sd_model_type != 'sd':
            shared.log.warning(f'Kohya Hires Fix: pipeline={MODELDATA.sd_model_type} required=sd')
            return None
        old_pipe = MODELDATA.sd_model
        high_res_fix = [{'timestep': timestep, 'scale_factor': scale_factor, 'block_num': block_num}]
        MODELDATA.sd_model = diffusers.StableDiffusionPipeline.from_pipe(MODELDATA.sd_model, **{ 'custom_pipeline': 'kohya_hires_fix', 'high_res_fix': high_res_fix })
        sd_models.copy_diffuser_options(MODELDATA.sd_model, old_pipe)
        sd_models.move_model(MODELDATA.sd_model, devices.device) # move pipeline to device
        sd_models.set_diffuser_options(MODELDATA.sd_model, vae=None, op='model')
        shared.log.debug(f'Kohya Hires Fix: pipeline={MODELDATA.sd_model.__class__.__name__} args={high_res_fix}')
        processed = processing.process_images(p)
        MODELDATA.sd_model = old_pipe
        return processed
