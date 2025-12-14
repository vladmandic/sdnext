import gradio as gr
from sdnext_core import MODELDATA
from modules import scripts_manager, processing, shared, sd_models


registered = False


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.register()

    def title(self):
        return 'APG: Adaptive Projected Guidance'

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://arxiv.org/abs/2410.02416">&nbsp APG: Adaptive Projected Guidance</a><br>')
        with gr.Row():
            eta = gr.Slider(label="ETA", value=1.0, minimum=0, maximum=2.0, step=0.05)
            momentum = gr.Slider(label="Momentum", value=-0.50, minimum=-1.0, maximum=1.0, step=0.05)
            threshold = gr.Slider(label="Threshold", value=0.0, minimum=0.0, maximum=10.0, step=0.05)
        return [eta, momentum, threshold]

    def register(self): # register xyz grid elements
        global registered # pylint: disable=global-statement
        if registered:
            return
        registered = True
        def apply_field(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                setattr(p, field, x)
                self.run(p)
            return fun

        import sys
        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k]
        if xyz_classes and len(xyz_classes) > 0:
            xyz_classes = xyz_classes[0]
            options = [
                xyz_classes.AxisOption("[APG] ETA", float, apply_field("apg_eta")),
                xyz_classes.AxisOption("[APG] Momentum", float, apply_field("apg_momentum")),
                xyz_classes.AxisOption("[APG] Threshold", float, apply_field("apg_threshold")),
            ]
            for option in options:
                if option not in xyz_classes.axis_options:
                    xyz_classes.axis_options.append(option)

    def run(self, p: processing.StableDiffusionProcessing, eta = 0.0, momentum = 0.0, threshold = 0.0): # pylint: disable=arguments-differ
        supported_model_list = ['sd', 'sdxl', 'sc']
        if MODELDATA.sd_model_type not in supported_model_list:
            shared.log.warning(f'APG: class={MODELDATA.sd_model.__class__.__name__} model={MODELDATA.sd_model_type} required={supported_model_list}')
            return None
        from modules import apg
        apg.eta = getattr(p, 'apg_eta', eta) # use values set by xyz grid or via ui
        apg.momentum = getattr(p, 'apg_momentum', momentum)
        apg.threshold = getattr(p, 'apg_threshold', threshold)
        apg.buffer = apg.MomentumBuffer(apg.momentum) # recreate buffer
        # pipelines with call to apg.normalized_guidance instead of default
        if MODELDATA.sd_model_type == "sd":
            self.orig_pipe = MODELDATA.sd_model
            MODELDATA.sd_model = sd_models.switch_pipe(apg.StableDiffusionPipelineAPG, MODELDATA.sd_model)
        if MODELDATA.sd_model_type == "sdxl":
            self.orig_pipe = MODELDATA.sd_model
            MODELDATA.sd_model = sd_models.switch_pipe(apg.StableDiffusionXLPipelineAPG, MODELDATA.sd_model)
        elif MODELDATA.sd_model_type == "sc":
            self.orig_pipe = MODELDATA.sd_model.prior_pipe
            MODELDATA.sd_model.prior_pipe = sd_models.switch_pipe(apg.StableCascadePriorPipelineAPG, MODELDATA.sd_model.prior_pipe)
        shared.log.info(f'APG apply: guidance={p.cfg_scale} momentum={apg.momentum} eta={apg.eta} threshold={apg.threshold} class={MODELDATA.sd_model.__class__.__name__}')
        p.extra_generation_params["APG"] = f'ETA={apg.eta} Momentum={apg.momentum} Threshold={apg.threshold}'
        # processed = processing.process_images(p)
        return None

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, eta, momentum, threshold): # pylint: disable=arguments-differ, unused-argument
        from modules import apg
        if self.orig_pipe is None:
            return processed
        # restore pipeline
        if MODELDATA.sd_model_type == "sdxl" or MODELDATA.sd_model_type == "sd":
            MODELDATA.sd_model = self.orig_pipe
        elif MODELDATA.sd_model_type == "sc":
            MODELDATA.sd_model.prior_pipe = self.orig_pipe
        apg.buffer = None
        self.orig_pipe = None
        return processed
