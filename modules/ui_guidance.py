from functools import partial
import gradio as gr
from modules import shared
from modules import ui_symbols, ui_components
from modules.modular_guiders import guiders


_stored_args = {}


def get_modular_args():
    return _stored_args


def create_guidance_inputs(tab):
    with gr.Accordion(open=False, label='Guidance', elem_id=f"{tab}_guidance", elem_classes=["small-accordion"]):
        with gr.Group():

            with gr.Row(elem_id=f"{tab}_guider_row", elem_classes=['flexbox'], visible=shared.opts.model_modular_enable):
                cfg_name = gr.Dropdown(choices=guiders.keys(), value='Default', label='Guider', elem_id=f"{tab}_guider")
                cfg_name_btn = ui_components.ToolButton(value=ui_symbols.info, elem_id=f"{tab}_guider_docs")
                cfg_name_btn.click(fn=None, _js='getGuidanceDocs', inputs=[cfg_name], outputs=[])

            base_group = gr.Group()
            with base_group:
                with gr.Row():
                    cfg_scale = gr.Slider(minimum=-1.0, maximum=30.0, step=0.1, label='Guidance scale', value=-1.0, elem_id=f"{tab}_guidance_scale")
                    cfg_image = gr.Slider(minimum=-1.0, maximum=30.0, step=0.1, label='Guidance image', value=-1.0, elem_id=f"{tab}_guidance_image")
                with gr.Row():
                    cfg_rescale = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, label='Guidance rescale', value=-1.0, elem_id=f"{tab}_guidance_rescale")
                with gr.Row():
                    cfg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance start', value=0.0, elem_id=f"{tab}_guidance_start")
                    cfg_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='Guidance stop', value=1.0, elem_id=f"{tab}_guidance_stop")
                args_base = [cfg_name, cfg_scale, cfg_image, cfg_rescale, cfg_start, cfg_stop]

            auto_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with auto_group:
                guidance_auto_dropout = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='AutoGuidance dropout', value=1.0)
                guidance_auto_layers = gr.Textbox(label='AutoGuidance layers', value='', placeholder='layer indices, e.g. 7,8,9 or ranges, e.g. 7-9')
                args_auto = [guidance_auto_dropout, guidance_auto_layers]

            zero_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with zero_group:
                guidance_zero_init_steps = gr.Slider(minimum=0, maximum=10, step=1, label='ZeroStar init steps', value=1)
                args_zero = [guidance_zero_init_steps]

            pag_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with pag_group:
                guidance_pag_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.05, label='PAG scale', value=7.5)
                guidance_pag_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='PAG start', value=0.01)
                guidance_pag_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='PAG stop', value=0.2)
                guidance_pag_layers = gr.Textbox(label='PAG layers', value='', placeholder='layer indices, e.g. 7,8,9 or ranges, e.g. 7-9')
                args_pag = [guidance_pag_scale, guidance_pag_start, guidance_pag_stop, guidance_pag_layers]

            apg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with apg_group:
                guidance_apg_momentum = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, label='APG momentum', value=-1.0)
                guidance_apg_rescale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='APG rescale', value=15.0)
                args_apg = [guidance_apg_momentum, guidance_apg_rescale]

            slg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with slg_group:
                guidance_slg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='SLG scale', value=2.8)
                guidance_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SLG start', value=0.01)
                guidance_slg_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SLG stop', value=0.2)
                guidance_slg_layers = gr.Textbox(label='SLG layers', value='', placeholder='layer indices, e.g. 7,8,9 or ranges, e.g. 7-9')
                args_slg = [guidance_slg_scale, guidance_slg_start, guidance_slg_stop, guidance_slg_layers]

            seg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with seg_group:
                guidance_seg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='SEG scale', value=3.0)
                guidance_seg_blur_sigma = gr.Number(label='SEG blur sigma', value=9999999.0)
                guidance_seg_blur_threshold_inf = gr.Number(label='SEG blur threshold inf', value=9999.0)
                guidance_seg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SEG start', value=0.0)
                guidance_seg_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SEG stop', value=1.0)
                guidance_seg_layers = gr.Textbox(label='SEG layers', value='', placeholder='layer indices, e.g. 7,8,9 or ranges, e.g. 7-9')
                args_seg = [guidance_seg_scale, guidance_seg_blur_sigma, guidance_seg_blur_threshold_inf, guidance_seg_start, guidance_seg_stop, guidance_seg_layers]

            tcfg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with tcfg_group:
                args_tcfg = []

            fdg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with fdg_group:
                guidance_fdg_scales = gr.Textbox(label='FDG scales', value='10.0, 5.0', placeholder='descending scales, e.g. 10.0,5.0')
                guidance_fdg_weights = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='FDG weights', value=1.0)
                guidance_fdg_rescale_space = gr.Dropdown(choices=['data', 'freq'], value='data', label='FDG rescale space')
                args_fdg = [guidance_fdg_scales, guidance_fdg_weights, guidance_fdg_rescale_space]

            def adv_visibility(guidance_name):
                return [
                    gr.update(visible=guidance_name != 'Default' and guidance_name != 'None'),
                    gr.update(visible=guidance_name.startswith('Auto')),
                    gr.update(visible=guidance_name.startswith('Zero')),
                    gr.update(visible=guidance_name.startswith('PAG')),
                    gr.update(visible=guidance_name.startswith('APG')),
                    gr.update(visible=guidance_name.startswith('SLG')),
                    gr.update(visible=guidance_name.startswith('SEG')),
                    gr.update(visible=guidance_name.startswith('TCFG')),
                    gr.update(visible=guidance_name.startswith('FDG')),
                ]
            cfg_name.change(fn=adv_visibility, inputs=[cfg_name], outputs=[base_group, auto_group, zero_group, pag_group, apg_group, slg_group, seg_group, tcfg_group, fdg_group])

            legacy_group = gr.Row(visible=not shared.opts.model_modular_enable)
            with legacy_group:
                cfg_true = gr.Slider(minimum=-1.0, maximum=30.0, step=0.05, label='Attention guidance', value=-1.0, elem_id=f"{tab}_cfg_true")
                cfg_adaptive = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Adaptive scaling', value=0.5, elem_id=f"{tab}_cfg_adaptive")
                args_legacy = [cfg_true, cfg_adaptive]

    modular_args = args_auto + args_zero + args_pag + args_apg + args_slg + args_seg + args_tcfg + args_fdg
    standard_args = args_base + args_legacy

    def update_stored(component, name):
        _stored_args[name] = component
    for component in modular_args:
        label = getattr(component, 'label', None)
        value = getattr(component, 'value', None)
        name = label.lower().replace(' ', '_') if label is not None else None
        _stored_args[name] = value
        component.change(fn=partial(update_stored, name=name), inputs=[component], outputs=[])

    return standard_args
