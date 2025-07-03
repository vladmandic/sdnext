import gradio as gr
from modules import shared, scripts_manager, sd_models


class Script(scripts_manager.Script):

    def title(self):
        return 'LayerDiffuse: Transparent Image'

    def show(self, is_img2img):
        return True if shared.native else False

    def apply(self):
        from scripts import layerdiffuse
        if not shared.sd_loaded:
            shared.log.error('LayerDiffuse: model not loaded')
            return self.is_active()
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            shared.log.error(f'LayerDiffuse: incorrect base model: class={shared.sd_model.__class__.__name__} type={shared.sd_model_type}')
            return self.is_active()
        if hasattr(shared.sd_model, 'layerdiffusion'):
            shared.log.warning('LayerDiffuse: already applied')
            return self.is_active()
        layerdiffuse.apply_layerdiffuse()
        return self.is_active()

    def reload(self):
        sd_models.reload_model_weights(force=True)
        return self.is_active()

    def is_active(self):
        if not shared.sd_loaded:
            return '<div style="color: darkred">LayerDiffuse: model not loaded</div><br>'
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            return '<div style="color: darkred">LayerDiffuse: incorrect base model</div><br>'
        if hasattr(shared.sd_model, 'layerdiffusion'):
            return '<div style="color: darkgreen">LayerDiffuse: active</div><br>'
        return '<div style="color: darkgray">LayerDiffuse: inactive</div><br>'

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML("""
                    <a href="https://github.com/rootonchair/diffuser_layerdiffuse">&nbsp LayerDiffuse: Transparent Image</a><br><br>
                    <div>- Click Apply to model to apply LayerDiffuse to current model</div>
                    <div>- Click Reload model to remove LayerDiffuse from current model</div><br>
                    """)
        with gr.Row():
            active = gr.HTML('')
        with gr.Row():
            check_btn = gr.Button('Check status', variant='primary')
            apply_btn = gr.Button('Apply to model', variant='primary')
            reload_btn = gr.Button('Reload model', variant='primary')
            check_btn.click(fn=self.is_active, inputs=[], outputs=[active])
            apply_btn.click(fn=self.apply, inputs=[], outputs=[active])
            reload_btn.click(fn=self.reload, inputs=[], outputs=[active])
        return []
