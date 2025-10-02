import gradio as gr
from modules import shared
from modules import ui_symbols, ui_components


guiders = ['Default', 'CFG', 'Zero', 'PAG', 'APG', 'SLG', 'SEG', 'TCFG', 'FDG']


def create_guidance_inputs(tab):
    with gr.Accordion(open=False, label='Guidance', elem_id=f"{tab}_guidance", elem_classes=["small-accordion"]):
        with gr.Group():

            with gr.Row(elem_id=f"{tab}_guider_row", elem_classes=['flexbox'], visible=shared.opts.model_modular_enable):
                guidance_name = gr.Dropdown(choices=guiders, value='Default', label='Guider', elem_id=f"{tab}_guider")
                guidance_btn = ui_components.ToolButton(value=ui_symbols.book, elem_id=f"{tab}_guider_docs")
                guidance_btn.click(fn=None, _js='getGuidanceDocs', inputs=[guidance_name], outputs=[])
            with gr.Row(visible=shared.opts.model_modular_enable):
                guidance_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Guidance scale', value=6.0, elem_id=f"{tab}_guidance_scale")
                guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.0, elem_id=f"{tab}_guidance_rescale")
            with gr.Row(visible=shared.opts.model_modular_enable):
                guidance_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance start', value=0.0, elem_id=f"{tab}_guidance_start")
                guidance_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='Guidance stop', value=1.0, elem_id=f"{tab}_guidance_stop")
            guidance_args = [guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop]

            lsc_group = gr.Accordion(open=False, label='Layer skip guidance', elem_classes=["small-accordion"], visible=shared.opts.model_modular_enable)
            with lsc_group:
                with gr.Row():
                    guidance_lsc_enabled = gr.Checkbox(label='Enable LayerSkipConfig', value=False)
                    guidance_lsc_label = gr.Label(value='LSC: LayerSkipConfig', elem_id=f"{tab}_lsc_label", visible=False)
                    guidance_lsc_btn = ui_components.ToolButton(value=ui_symbols.book, elem_id=f"{tab}_lsc_docs", elem_classes=["guidance-docs"])
                    guidance_lsc_btn.click(fn=None, _js='getGuidanceDocs', inputs=[guidance_lsc_label], outputs=[])
                with gr.Row():
                    guidance_lsc_indices = gr.Textbox(label='LSC layer indices', value='1, 2, 3', placeholder='Comma-separated layer indices to skip')
                with gr.Row():
                    guidance_lsc_fqn = gr.Textbox(label='LSC fully qualified name', value='transformer_blocks', placeholder='Fully qualified name of the layer stack')
                with gr.Row():
                    guidance_lsc_skip_attention = gr.Checkbox(label='LSC skip attention blocks', value=True)
                    guidance_lsc_skip_ff = gr.Checkbox(label='LSC skip feed-forward blocks', value=True)
                    guidance_lsc_skip_attention_scores = gr.Checkbox(label='LSC skip attention scores', value=False)
                with gr.Row():
                    guidance_lsc_dropout = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='LSC dropout rate', value=1.0)
                lsc_args = [guidance_lsc_enabled, guidance_lsc_indices, guidance_lsc_fqn, guidance_lsc_skip_attention, guidance_lsc_skip_ff, guidance_lsc_skip_attention_scores, guidance_lsc_dropout]

            auto_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with auto_group:
                guidance_auto_dropout = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='AutoGuidance dropout', value=0.1)
                guidance_auto_layers = gr.Textbox(label='AutoGuidance layers', value='7, 8, 9', placeholder='Comma-separated layer indices, e.g. 7,8,9')
                guidance_auto_config = gr.Dropdown(choices=[None, 'config1', 'config2'], value=None, label='AutoGuidance config')
                guidance_auto_args = [guidance_auto_dropout, guidance_auto_layers, guidance_auto_config]

            zero_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with zero_group:
                guidance_zero_init_steps = gr.Slider(minimum=0, maximum=10, step=1, label='ZeroStar init steps', value=1)
                guidance_zero_args = [guidance_zero_init_steps]

            pag_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with pag_group:
                guidance_pag_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.05, label='PAG scale', value=2.8)
                guidance_pag_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='PAG start', value=0.01)
                guidance_pag_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='PAG stop', value=0.2)
                guidance_pag_layers = gr.Textbox(label='PAG layers', value='7, 8, 9', placeholder='Comma-separated layer indices, e.g. 7,8,9')
                guidance_pag_config = gr.Dropdown(choices=[None, 'config1', 'config2'], value=None, label='PAG config')
                guidance_pag_args = [guidance_pag_scale, guidance_pag_start, guidance_pag_stop, guidance_pag_layers, guidance_pag_config]

            apg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with apg_group:
                guidance_apg_momentum = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, label='APG momentum', value=-1.0)
                guidance_apg_rescale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='APG rescale', value=15.0)
                guidance_apg_args = [guidance_apg_momentum, guidance_apg_rescale]

            slg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with slg_group:
                guidance_slg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='SLG scale', value=2.8)
                guidance_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SLG start', value=0.01)
                guidance_slg_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SLG stop', value=0.2)
                guidance_slg_layers = gr.Textbox(label='SLG layers', value='7, 8, 9', placeholder='Comma-separated layer indices, e.g. 7,8,9')
                guidance_slg_config = gr.Dropdown(choices=[None, 'config1', 'config2'], value=None, label='SLG config')
                guidance_slg_args = [guidance_slg_scale, guidance_slg_start, guidance_slg_stop, guidance_slg_layers, guidance_slg_config]

            seg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with seg_group:
                guidance_seg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='SEG scale', value=3.0)
                guidance_seg_blur_sigma = gr.Number(label='SEG blur sigma', value=9999999.0)
                guidance_seg_blur_threshold_inf = gr.Number(label='SEG blur threshold inf', value=9999.0)
                guidance_seg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SEG start', value=0.0)
                guidance_seg_stop = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='SEG stop', value=1.0)
                guidance_seg_layers = gr.Textbox(label='SEG layers', value='7, 8, 9', placeholder='Comma-separated layer indices, e.g. 7,8,9')
                guidance_seg_config = gr.Dropdown(choices=[None, 'config1', 'config2'], value=None, label='SEG config')
                guidance_seg_args = [guidance_seg_scale, guidance_seg_blur_sigma, guidance_seg_blur_threshold_inf, guidance_seg_start, guidance_seg_stop, guidance_seg_layers, guidance_seg_config]

            tcfg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with tcfg_group:
                pass

            fdg_group = gr.Accordion(open=True, label='Advanced guidance params', elem_classes=["small-accordion"], visible=False)
            with fdg_group:
                guidance_fdg_scales = gr.Textbox(label='FDG scales', value='10.0, 5.0', placeholder='Comma-separated scales, e.g. 10.0,5.0')
                guidance_fdg_weights = gr.Textbox(label='FDG weights', value='1.0', placeholder='Single float or comma-separated weights, e.g. 1.0 or 1.0,0.5')
                guidance_fdg_rescale_space = gr.Dropdown(choices=['data', 'freq'], value='data', label='FDG rescale space')
                guidance_fdg_args = [guidance_fdg_scales, guidance_fdg_weights, guidance_fdg_rescale_space]

            def adv_visibility(guidance_name):
                return [
                    gr.update(visible=guidance_name.startswith('Auto')),
                    gr.update(visible=guidance_name.startswith('Zero')),
                    gr.update(visible=guidance_name.startswith('PAG')),
                    gr.update(visible=guidance_name.startswith('APG')),
                    gr.update(visible=guidance_name.startswith('SLG')),
                    gr.update(visible=guidance_name.startswith('SEG')),
                    gr.update(visible=guidance_name.startswith('TCFG')),
                    gr.update(visible=guidance_name.startswith('FDG')),
                ]
            guidance_name.change(fn=adv_visibility, inputs=[guidance_name], outputs=[auto_group, zero_group, pag_group, apg_group, slg_group, seg_group, tcfg_group, fdg_group])

            gr.HTML(value='<br><h2>Fallback guidance</h2>', visible=shared.opts.model_modular_enable, elem_id=f"{tab}_guidance_note")
            with gr.Row(elem_id=f"{tab}_cfg_row", elem_classes=['flexbox']):
                cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Guidance scale', value=6.0, elem_id=f"{tab}_cfg_scale")
                cfg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='Guidance end', value=1.0, elem_id=f"{tab}_cfg_end")
            with gr.Row():
                image_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Refine guidance', value=6.0, elem_id=f"{tab}_image_cfg_scale")
                diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Rescale guidance', value=0.0, elem_id=f"{tab}_image_cfg_rescale")
            with gr.Row():
                diffusers_pag_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.05, label='Attention guidance', value=0.0, elem_id=f"{tab}_pag_scale")
                diffusers_pag_adaptive = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Adaptive scaling', value=0.5, elem_id=f"{tab}_pag_adaptive")

    _modular_args = guidance_args + lsc_args + guidance_auto_args + guidance_zero_args + guidance_pag_args + guidance_apg_args + guidance_slg_args + guidance_seg_args + guidance_fdg_args
    standard_args = [cfg_scale, image_cfg_scale, diffusers_guidance_rescale, diffusers_pag_scale, diffusers_pag_adaptive, cfg_end]
    return guidance_args + standard_args
