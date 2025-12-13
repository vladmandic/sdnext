import gradio as gr
from modules import scripts_manager, processing, shared, sd_models, devices
from installer import install


class Script(scripts_manager.Script):
    def title(self):
        return 'T-Gate: Accelerate via Gating Attention'

    def show(self, is_img2img):
        return not is_img2img

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/HaozheLiu-ST/T-GATE">&nbsp T-Gate: Accelerate via Gating Attention</a><br>')
        with gr.Row():
            enabled = gr.Checkbox(label="Enabled", value=True)
        with gr.Row():
            gate_step = gr.Slider(minimum=1, maximum=50, step=1, label="Gate step", elem_id="t_gate_steps", value=10)
        return [enabled, gate_step]

    def run(self, p: processing.StableDiffusionProcessing, enabled, gate_step): # pylint: disable=arguments-differ
        p.gate_step = min(gate_step, p.steps) if enabled else -1
        if not enabled:
            return None
        install('tgate')
        import tgate
        if MODELDATA.sd_model_type == 'sd':
            cls = tgate.TgateSDLoader
        elif MODELDATA.sd_model_type == 'sdxl':
            cls = tgate.TgateSDXLLoader
        else:
            shared.log.warning(f'T-Gate: pipeline={MODELDATA.sd_model_type} required=sd or sdxl')
            return None
        old_pipe = MODELDATA.sd_model
        MODELDATA.sd_model = cls(MODELDATA.sd_model, gate_step=p.gate_step)
        sd_models.copy_diffuser_options(MODELDATA.sd_model, old_pipe)
        sd_models.move_model(MODELDATA.sd_model, devices.device) # move pipeline to device
        sd_models.set_diffuser_options(MODELDATA.sd_model, vae=None, op='model')
        shared.log.debug(f'T-Gate: pipeline={MODELDATA.sd_model.__class__.__name__} steps={p.gate_step}')
        processed = processing.process_images(p)
        MODELDATA.sd_model = old_pipe
        del MODELDATA.sd_model.tgate
        return processed
