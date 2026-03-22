import gradio as gr
import installer
from modules import scripts_manager, shared


class Script(scripts_manager.Script):
    def title(self):
        return "ROCm: Advanced Config"

    def show(self, _is_img2img):
        if shared.cmd_opts.use_rocm or installer.torch_info.get('type') == 'rocm':
            return scripts_manager.AlwaysVisible # script should be visible only if rocm is detected or forced
        return False

    def ui(self, _is_img2img):
        # all ui elements go here
        # logic should go into scripts/rocm/rocm_mgr.py and be called from here
        if not shared.cmd_opts.use_rocm and not installer.torch_info.get('type') == 'rocm': # skip ui creation if not rocm
            return []
        from scripts.rocm import rocm_mgr # pylint: disable=no-name-in-module
        rocm_mgr.load() # load config before ui creation so we can populate fields with current values
        with gr.Accordion('ROCM: Advanced Config', open=False, elem_id='rocm_config'): # set all ui in this accordion
            with gr.Row():
                gr.HTML('Advanced configuration for ROCm users')
            with gr.Row():
                btn_info = gr.Button("Refresh") # refresh info and populate with additional fields
                btn_apply = gr.Button("Apply") # apply changes and save config
                btn_reset = gr.Button("Reset") # reset config and save config
            with gr.Row():
                info = gr.JSON(installer.gpu_info, label="ROCm Info") # just an example how to display info we already have
            with gr.Row():
                db_path = gr.Textbox(label="MiOpen SystemDB Path", value=rocm_mgr.DATA.get('MIOPEN_SYSTEM_DB_PATH', ''), lines=1)

        args = [ # list all params here and note that it *must* match with scripts/rocm/rocm_mgr.apply() args
            db_path,
        ]

        btn_info.click(fn=rocm_mgr.info, inputs=[], outputs=[info])
        btn_apply.click(fn=rocm_mgr.apply, inputs=args, outputs=[])
        btn_reset.click(fn=rocm_mgr.reset, inputs=[], outputs=[])

        return args
