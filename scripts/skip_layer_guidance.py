import sys
import gradio as gr
from modules import scripts_manager, processing, shared


registered = False


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.register()

    def title(self):
        return 'SLG: Skip Layer Guidance'

    def show(self, is_img2img):
        return shared.native

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            layers = gr.Textbox(label='Skip guidance layers', value='7,8,9')
        with gr.Row():
            scale = gr.Slider(label='Guidance strength', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Row():
            start = gr.Slider(label='Guidance start', minimum=0.0, maximum=1.0, step=0.01, value=0.01)
            stop = gr.Slider(label='Guidance stop', minimum=0.0, maximum=1.0, step=0.01, value=0.2)
        return [layers, scale, start, stop]

    def register(self): # register xyz grid elements
        global registered # pylint: disable=global-statement
        if registered:
            return
        registered = True
        def apply_task_args(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                try:
                    val = str(x).replace('"', '')
                    val = [int(layer.strip()) for layer in val.split(',')]
                except Exception:
                    return
                if len(val) > 0:
                    shared.log.debug(f'SLG: {field}={val}')
                    p.task_args[field] = val
            return fun

        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k][0]
        options = [
            xyz_classes.AxisOption("[SLG] Layers", str, apply_task_args("skip_guidance_layers")),
        ]
        for option in options:
            if option not in xyz_classes.axis_options:
                xyz_classes.axis_options.append(option)


    def run(self, p: processing.StableDiffusionProcessing, layers: str = '', scale: float = 1.0, start: float = 1.0, stop: float = 1.0): # pylint: disable=arguments-differ, unused-argument
        if shared.sd_model_type != 'sd3':
            return
        p.task_args['skip_layer_guidance_scale'] = float(scale)
        p.task_args['skip_layer_guidance_start'] = float(start)
        p.task_args['skip_layer_guidance_stop'] = float(stop)
        parsed = []
        try:
            parsed = [int(layer.strip()) for layer in layers.split(',')]
        except Exception:
            return
        if len(parsed) == 0:
            return
        p.task_args['skip_guidance_layers'] = parsed
        shared.log.info(f'SLG: layers={parsed} scale={scale} start={start} stop={stop}')
