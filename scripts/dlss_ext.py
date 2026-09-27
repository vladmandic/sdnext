import gradio as gr
from modules import processing, scripts_manager, scripts_postprocessing
from modules.dlss import NR_PROFILES, VSR_PROFILES, FG_PROFILES
from scripts.dlss.dlss_run import dlss_run # pylint: disable=no-name-in-module


registered = False


def create_ui(parent):
    with gr.Accordion('nVidia DLSS', open=False, elem_id=f'{parent}_dlss_accordion'):
        with gr.Row():
            dlss_enabled = gr.Dropdown(label='DLSS Operations', choices=['NeuralRender', 'SuperRes', 'FrameGen'], value=[], multiselect=True, elem_id=f'{parent}_dlss_ops')

        with gr.Accordion('DLSS NeuralRender', open=False, elem_id=f'{parent}_dlss_nr'):
            with gr.Row():
                nr_profile = gr.Dropdown(label='NR profile', choices=list(NR_PROFILES), value='Standard', elem_id=f'{parent}_dlss_nr_profile')
                nr_motion = gr.Dropdown(label='NR motion vector', choices=['flow', 'zero'], value='flow', elem_id=f'{parent}_dlss_nr_motion')
            with gr.Row():
                nr_scale = gr.Slider(label='NR scale', minimum=0.1, maximum=4.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_scale')
                nr_intensity = gr.Slider(label='NR intensity', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_intensity')
            with gr.Row():
                nr_detail = gr.Slider(label='NR detail', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_detail')
                nr_colour = gr.Slider(label='NR colour', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_colour')
            with gr.Row():
                nr_blend = gr.Slider(label='NR blend', minimum=0.0, maximum=1.0, step=0.01, value=0.73974, elem_id=f'{parent}_dlss_nr_blend')
                nr_radius = gr.Slider(label='NR radius', minimum=1.0, maximum=16.0, step=0.5, value=4.0, elem_id=f'{parent}_dlss_nr_radius')
            with gr.Row():
                nr_threshold = gr.Slider(label='NR scene threshold', minimum=0.0, maximum=1.0, step=0.05, value=0.3, elem_id=f'{parent}_dlss_nr_threshold')
                nr_normalized = gr.Slider(label='NR normalized style', minimum=0.0, maximum=1.0, step=0.01, value=0.0, elem_id=f'{parent}_dlss_nr_normalized')
            with gr.Row():
                nr_local_tone = gr.Slider(label='NR local tone', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_local_tone')
                nr_local_structure = gr.Slider(label='NR local structure', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_nr_local_structure')
            with gr.Row():
                nr_skin_structure = gr.Slider(label='NR skin structure', minimum=0.0, maximum=2.0, step=0.05, value=0.0, elem_id=f'{parent}_dlss_nr_skin_structure')
                nr_mask_structure = gr.Slider(label='NR mask structure', minimum=0.0, maximum=2.0, step=0.05, value=0.0, elem_id=f'{parent}_dlss_nr_mask_structure')

        with gr.Accordion('DLSS SuperRes', open=False, elem_id=f'{parent}_dlss_ss'):
            with gr.Row():
                ss_profile = gr.Dropdown(label='SS profile', choices=list(VSR_PROFILES), value='Ultra', elem_id=f'{parent}_dlss_ss_profile')
                ss_scale = gr.Slider(label='SS scale', minimum=1.0, maximum=4.0, step=0.1, value=2.0, elem_id=f'{parent}_dlss_ss_scale')
            with gr.Row():
                ss_detail = gr.Slider(label='SS detail strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_ss_detail')
                ss_colour = gr.Slider(label='SS colour strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id=f'{parent}_dlss_ss_colour')
            with gr.Row():
                ss_radius = gr.Slider(label='SS detail radius', minimum=1.0, maximum=16.0, step=0.5, value=4.0, elem_id=f'{parent}_dlss_ss_radius')
                ss_threshold = gr.Slider(label='SS scene threshold', minimum=0.0, maximum=1.0, step=0.05, value=0.4, elem_id=f'{parent}_dlss_ss_threshold')

        with gr.Accordion('DLSS FrameGen', open=False, elem_id=f'{parent}_dlss_fg'):
            with gr.Row():
                fg_profile = gr.Dropdown(label='FG profile', choices=list(FG_PROFILES), value='None', elem_id=f'{parent}_dlss_fg_profile', visible=False)
                fg_factor = gr.Slider(label='FG factor', minimum=2, maximum=4, step=1, value=2, elem_id=f'{parent}_dlss_fg_factor')
            with gr.Row():
                fg_mode = gr.Dropdown(label='FG mode', choices=['fps', 'slowmo'], value='fps', elem_id=f'{parent}_dlss_fg_mode')
                fg_threshold = gr.Slider(label='FG scene threshold', minimum=0.0, maximum=1.0, step=0.05, value=0.4, elem_id=f'{parent}_dlss_fg_threshold')

        with gr.Accordion('DLSS Advanced', open=False, elem_id=f'{parent}_dlss_advanced'):
            dlss_graph = gr.Checkbox(label='DLSS use CUDA graph', value=False, elem_id=f'{parent}_dlss_graph')

    return [
        dlss_enabled, dlss_graph,
        nr_profile, nr_motion, nr_scale, nr_intensity, nr_blend, nr_detail, nr_colour, nr_radius, nr_threshold, nr_normalized, nr_local_tone, nr_local_structure, nr_skin_structure, nr_mask_structure,
        ss_profile, ss_scale, ss_detail, ss_colour, ss_radius, ss_threshold,
        fg_profile, fg_mode, fg_factor, fg_threshold,
    ]


class DLSSScript(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.video_capable = True
        self.register()

    def title(self):
        return 'nVidia DLSS'

    def show(self, _is_img2img):
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return create_ui(self.parent)

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
                xyz_classes.AxisOption("[DLSS] NR profile", str, apply_field("nr_profile"), choices=lambda: list(NR_PROFILES)),
                xyz_classes.AxisOption("[DLSS] SS profile", str, apply_field("ss_profile"), choices=lambda: list(VSR_PROFILES)),
                xyz_classes.AxisOption("[DLSS] FG profile", str, apply_field("fg_profile"), choices=lambda: list(FG_PROFILES)),
            ]
            for option in options:
                if option not in xyz_classes.axis_options:
                    xyz_classes.axis_options.append(option)

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, *args, **kwargs):
        if p.xyz:
            pp = dlss_run(p, pp, *args, **kwargs)

    def postprocess(self, p: processing.StableDiffusionProcessing, pp: processing.Processed, *args, **kwargs): # pylint: disable=arguments-differ,unused-argument
        if p.xyz: # do not postprocessing when running in xyz mode
            return pp
        _pp = dlss_run(p, pp, *args, **kwargs)
        # postprocess triggers after initial images have already been saved
        if _pp is not None and hasattr(_pp, 'images') and _pp.images is not None:
            pp = _pp
            orig_infos = pp.infotexts if hasattr(pp, 'infotexts') else []
            out_images, out_infos = processing.process_samples(p, pp.images)
            pp.images = out_images
            pp.infotexts = out_infos
            if hasattr(pp, 'originals') and pp.originals is not None and len(pp.originals) > 0:
                pp.infotexts = orig_infos + pp.infotexts
                pp.images = pp.originals + pp.images
        return pp


class DLSSPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "nVidia DLSS"
    order = 30000

    def ui(self):
        dlss_enabled, dlss_graph, nr_profile, nr_motion, nr_scale, nr_intensity, nr_blend, nr_detail, nr_colour, nr_radius, nr_threshold, nr_normalized, nr_local_tone, nr_local_structure, nr_skin_structure, nr_mask_structure, ss_profile, ss_scale, ss_detail, ss_colour, ss_radius, ss_threshold, fg_profile, fg_mode, fg_factor, fg_threshold = create_ui('postprocess')
        return {
            "dlss_enabled": dlss_enabled,
            "dlss_graph": dlss_graph,
            "nr_profile": nr_profile,
            "nr_motion": nr_motion,
            "nr_scale": nr_scale,
            "nr_intensity": nr_intensity,
            "nr_blend": nr_blend,
            "nr_detail": nr_detail,
            "nr_colour": nr_colour,
            "nr_radius": nr_radius,
            "nr_threshold": nr_threshold,
            "nr_normalized": nr_normalized,
            "nr_local_tone": nr_local_tone,
            "nr_local_structure": nr_local_structure,
            "nr_skin_structure": nr_skin_structure,
            "nr_mask_structure": nr_mask_structure,
            "ss_profile": ss_profile,
            "ss_scale": ss_scale,
            "ss_detail": ss_detail,
            "ss_colour": ss_colour,
            "ss_radius": ss_radius,
            "ss_threshold": ss_threshold,
            "fg_profile": fg_profile,
            "fg_mode": fg_mode,
            "fg_factor": fg_factor,
            "fg_threshold": fg_threshold,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, *args, **kwargs):
        dlss_enabled = kwargs.get("dlss_enabled", False)
        if not dlss_enabled:
            return
        if pp.image is None and pp.video is None:
            return
        result = dlss_run(None, pp, *args, **kwargs)
        if result is None or not hasattr(result, "images") or len(result.images) == 0:
            return
        pp.image = result.images[0]
        if isinstance(dlss_enabled, list) and pp.image is not None:
            pp.info["DLSS"] = ' '.join(dlss_enabled)
