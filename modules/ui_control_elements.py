import gradio as gr
import matplotlib.pyplot as plt
from modules.control import unit
from modules.control import processors # patrickvonplaten controlnet_aux
from modules.control.units import controlnet # lllyasviel ControlNet
from modules.control.units import xs # vislearn ControlNet-XS
from modules.control.units import lite # vislearn ControlNet-XS
from modules.control.units import t2iadapter # TencentARC T2I-Adapter
from modules.control.units import reference # reference pipeline
from modules import shared, ui_components, ui_symbols, ui_common, masking # pylint: disable=ungrouped-imports
from modules import ui_control_helpers as helpers


def create_ui_elements(units, result_txt, preview_process):
    max_units = shared.opts.control_max_units
    with gr.Accordion('Control elements', open=False, elem_id="control_elements"):
        with gr.Tabs(elem_id='control-tabs') as _tabs_control_type:

            with gr.Tab('ControlNet') as _tab_controlnet:
                gr.HTML('<a href="https://github.com/lllyasviel/ControlNet">ControlNet</a>')
                with gr.Row():
                    extra_controls = [
                        gr.Checkbox(label="Guess mode", value=False, scale=3),
                    ]
                    num_controlnet_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                controlnet_ui_units = [] # list of hidable accordions
                for i in range(max_units):
                    enabled = True if i==0 else False
                    with gr.Accordion(f'ControlNet unit {i+1}', visible= i < num_controlnet_units.value, elem_classes='control-unit') as unit_ui:
                        with gr.Row():
                            with gr.Group(elem_id=f'controlnet_unit-{i}-controls', elem_classes='controlnet-controls'):
                                enabled_cb = gr.Checkbox(enabled, label='Active', container=False, show_label=True, elem_id=f'control_unit-{i}-enabled')
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'], elem_id=f'controlnet_unit-{i}-upload')
                                image_reuse= ui_components.ToolButton(value=ui_symbols.reuse, elem_id=f'controlnet_unit-{i}-reuse')
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset, elem_id=f'controlnet_unit-{i}-reset')
                                preview_btn = ui_components.ToolButton(value=ui_symbols.preview, elem_id=f'controlnet_unit-{i}-preview')
                            process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None', elem_id=f'control_unit-{i}-process_name')
                            model_id = gr.Dropdown(label="ControlNet", choices=controlnet.list_models(), value='None', elem_id=f'control_unit-{i}-model_name')
                            ui_common.create_refresh_button(model_id, controlnet.list_models, lambda: {"choices": controlnet.list_models(refresh=True)}, f'controlnet_models_{i}_refresh')
                            control_mode = gr.Dropdown(label="CN Mode", choices=['default'], value='default', visible=False, elem_id=f'control_unit-{i}-mode')
                            model_strength = gr.Slider(label="CN Strength", minimum=0.01, maximum=2.0, step=0.01, value=1.0, elem_id=f'control_unit-{i}-strength')
                            control_start = gr.Slider(label="CN Start", minimum=0.0, maximum=1.0, step=0.05, value=0, elem_id=f'control_unit-{i}-start')
                            control_end = gr.Slider(label="CN End", minimum=0.0, maximum=1.0, step=0.05, value=1.0, elem_id=f'control_unit-{i}-end')
                            control_tile = gr.Dropdown(label="CN Tiles", choices=[x.strip() for x in shared.opts.control_tiles.split(',') if 'x' in x], value='1x1', visible=False, elem_id=f'control_unit-{i}-tile')
                            image_preview = gr.Image(label="Input", type="pil", height=128, width=128, visible=False, interactive=True, show_label=False, show_download_button=False, container=False, elem_id=f'controlnet_unit-{i}-override')
                    controlnet_ui_units.append(unit_ui)
                    units.append(unit.Unit(
                        unit_type = 'controlnet',
                        index = i,
                        enabled = enabled,
                        result_txt = result_txt,
                        enabled_cb = enabled_cb,
                        reset_btn = reset_btn,
                        process_id = process_id,
                        model_id = model_id,
                        model_strength = model_strength,
                        preview_process = preview_process,
                        preview_btn = preview_btn,
                        image_upload = image_upload,
                        image_reuse = image_reuse,
                        image_preview = image_preview,
                        control_start = control_start,
                        control_end = control_end,
                        control_mode = control_mode,
                        control_tile = control_tile,
                        extra_controls = extra_controls,
                        )
                    )
                    if i == 0:
                        units[-1].enabled = True # enable first unit in group
                num_controlnet_units.change(fn=helpers.display_units, inputs=[num_controlnet_units], outputs=controlnet_ui_units)

            with gr.Tab('T2I Adapter') as _tab_t2iadapter:
                gr.HTML('<a href="https://github.com/TencentARC/T2I-Adapter">T2I-Adapter</a>')
                with gr.Row():
                    extra_controls = [
                        gr.Slider(label="Control factor", minimum=0.0, maximum=1.0, step=0.05, value=1.0, scale=3),
                    ]
                    num_adapter_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                adapter_ui_units = [] # list of hidable accordions
                for i in range(max_units):
                    enabled = True if i==0 else False
                    with gr.Accordion(f'T2I-Adapter unit {i+1}', visible= i < num_adapter_units.value, elem_classes='control-unit') as unit_ui:
                        with gr.Row():
                            enabled_cb = gr.Checkbox(enabled, label='Active', container=False, show_label=True, elem_id=f'control_unit-{i}-enabled')
                            process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None', elem_id=f'control_unit-{i}-process_name')
                            model_id = gr.Dropdown(label="Adapter", choices=t2iadapter.list_models(), value='None', elem_id=f'control_unit-{i}-model_name')
                            ui_common.create_refresh_button(model_id, t2iadapter.list_models, lambda: {"choices": t2iadapter.list_models(refresh=True)}, f'adapter_models_{i}_refresh')
                            model_strength = gr.Slider(label="T2I Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0, elem_id=f'control_unit-{i}-strength')
                            reset_btn = ui_components.ToolButton(value=ui_symbols.reset, elem_id=f'adapter_unit-{i}-reset')
                            image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'], elem_id=f'adapter_unit-{i}-upload')
                            image_reuse= ui_components.ToolButton(value=ui_symbols.reuse, elem_id=f'adapter_unit-{i}-reuse')
                            btn_preview= ui_components.ToolButton(value=ui_symbols.preview, elem_id=f'adapter_unit-{i}-preview')
                            image_preview = gr.Image(label="Input", show_label=False, type="pil", interactive=False, height=128, width=128, visible=False, elem_id=f'adapter_unit-{i}-override')
                    adapter_ui_units.append(unit_ui)
                    units.append(unit.Unit(
                        unit_type = 't2i adapter',
                        index = i,
                        enabled = enabled,
                        result_txt = result_txt,
                        enabled_cb = enabled_cb,
                        reset_btn = reset_btn,
                        process_id = process_id,
                        model_id = model_id,
                        model_strength = model_strength,
                        preview_process = preview_process,
                        preview_btn = btn_preview,
                        image_upload = image_upload,
                        image_reuse = image_reuse,
                        image_preview = image_preview,
                        extra_controls = extra_controls,
                        )
                    )
                    if i == 0:
                        units[-1].enabled = True # enable first unit in group
                num_adapter_units.change(fn=helpers.display_units, inputs=[num_adapter_units], outputs=adapter_ui_units)

            with gr.Tab('XS') as _tab_controlnetxs:
                gr.HTML('<a href="https://vislearn.github.io/ControlNet-XS/">ControlNet XS</a>')
                with gr.Row():
                    extra_controls = [
                        gr.Slider(label="Time embedding mix", minimum=0.0, maximum=1.0, step=0.05, value=0.0, scale=3)
                    ]
                    num_controlnet_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                controlnetxs_ui_units = [] # list of hidable accordions
                for i in range(max_units):
                    enabled = True if i==0 else False
                    with gr.Accordion(f'ControlNet-XS unit {i+1}', visible= i < num_controlnet_units.value, elem_classes='control-unit') as unit_ui:
                        with gr.Row():
                            enabled_cb = gr.Checkbox(enabled, label='Active', container=False, show_label=True, elem_id=f'control_unit-{i}-enabled')
                            process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None', elem_id=f'control_unit-{i}-process_name')
                            model_id = gr.Dropdown(label="ControlNet-XS", choices=xs.list_models(), value='None', elem_id=f'control_unit-{i}-model_name')
                            ui_common.create_refresh_button(model_id, xs.list_models, lambda: {"choices": xs.list_models(refresh=True)}, f'xs_models_{i}_refresh')
                            model_strength = gr.Slider(label="CN Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0, elem_id=f'control_unit-{i}-strength')
                            control_start = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.05, value=0, elem_id=f'control_unit-{i}-start')
                            control_end = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.05, value=1.0, elem_id=f'control_unit-{i}-end')
                            reset_btn = ui_components.ToolButton(value=ui_symbols.reset, elem_id=f'controlnetxs_unit-{i}-reset')
                            image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'], elem_id=f'controlnetxs_unit-{i}-upload')
                            image_reuse= ui_components.ToolButton(value=ui_symbols.reuse, elem_id=f'controlnetxs_unit-{i}-reuse')
                            btn_preview= ui_components.ToolButton(value=ui_symbols.preview, elem_id=f'controlnetxs_unit-{i}-preview')
                            image_preview = gr.Image(label="Input", show_label=False, type="pil", interactive=False, height=128, width=128, visible=False, elem_id=f'controlnetxs_unit-{i}-override')
                    controlnetxs_ui_units.append(unit_ui)
                    units.append(unit.Unit(
                        unit_type = 'xs',
                        index = i,
                        enabled = enabled,
                        result_txt = result_txt,
                        enabled_cb = enabled_cb,
                        reset_btn = reset_btn,
                        process_id = process_id,
                        model_id = model_id,
                        model_strength = model_strength,
                        preview_process = preview_process,
                        preview_btn = btn_preview,
                        image_upload = image_upload,
                        image_reuse = image_reuse,
                        image_preview = image_preview,
                        control_start = control_start,
                        control_end = control_end,
                        extra_controls = extra_controls,
                        )
                    )
                    if i == 0:
                        units[-1].enabled = True # enable first unit in group
                num_controlnet_units.change(fn=helpers.display_units, inputs=[num_controlnet_units], outputs=controlnetxs_ui_units)

            with gr.Tab('Lite') as _tab_lite:
                gr.HTML('<a href="https://huggingface.co/kohya-ss/controlnet-lllite">Control LLLite</a>')
                with gr.Row():
                    extra_controls = [
                    ]
                    num_lite_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                lite_ui_units = [] # list of hidable accordions
                for i in range(max_units):
                    enabled = True if i==0 else False
                    with gr.Accordion(f'Control-LLLite unit {i+1}', visible= i < num_lite_units.value, elem_classes='control-unit') as unit_ui:
                        with gr.Row():
                            enabled_cb = gr.Checkbox(enabled, label='Active', container=False, show_label=True, elem_id=f'control_unit-{i}-enabled')
                            process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None', elem_id=f'control_unit-{i}-process_name')
                            model_id = gr.Dropdown(label="Model", choices=lite.list_models(), value='None', elem_id=f'control_unit-{i}-model_name')
                            ui_common.create_refresh_button(model_id, lite.list_models, lambda: {"choices": lite.list_models(refresh=True)}, f'lite_models_{i}_refresh')
                            model_strength = gr.Slider(label="CN Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0, elem_id=f'control_unit-{i}-strength')
                            reset_btn = ui_components.ToolButton(value=ui_symbols.reset, elem_id=f'lite_unit-{i}-reset')
                            image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'], elem_id=f'lite_unit-{i}-upload')
                            image_reuse= ui_components.ToolButton(value=ui_symbols.reuse, elem_id=f'lite_unit-{i}-reuse')
                            image_preview = gr.Image(label="Input", show_label=False, type="pil", interactive=False, height=128, width=128, visible=False, elem_id=f'lite_unit-{i}-override')
                            btn_preview= ui_components.ToolButton(value=ui_symbols.preview, elem_id=f'lite_unit-{i}-preview')
                    lite_ui_units.append(unit_ui)
                    units.append(unit.Unit(
                        unit_type = 'lite',
                        index = i,
                        enabled = enabled,
                        result_txt = result_txt,
                        enabled_cb = enabled_cb,
                        reset_btn = reset_btn,
                        process_id = process_id,
                        model_id = model_id,
                        model_strength = model_strength,
                        preview_process = preview_process,
                        preview_btn = btn_preview,
                        image_upload = image_upload,
                        image_reuse = image_reuse,
                        image_preview = image_preview,
                        extra_controls = extra_controls,
                        )
                    )
                    if i == 0:
                        units[-1].enabled = True # enable first unit in group
                num_lite_units.change(fn=helpers.display_units, inputs=[num_lite_units], outputs=lite_ui_units)

            with gr.Tab('Reference') as _tab_reference:
                gr.HTML('<a href="https://github.com/Mikubill/sd-webui-controlnet/discussions/1236">ControlNet reference-only control</a>')
                with gr.Row():
                    extra_controls = [
                        gr.Radio(label="Reference context", choices=['Attention', 'Adain', 'Attention Adain'], value='Attention', interactive=True),
                        gr.Slider(label="Style fidelity", minimum=0.0, maximum=1.0, step=0.05, value=0.5, interactive=True), # prompt vs control importance
                        gr.Slider(label="Reference query weight", minimum=0.0, maximum=1.0, step=0.05, value=1.0, interactive=True),
                        gr.Slider(label="Reference adain weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True),
                    ]
                for i in range(1): # can only have one reference unit
                    enabled = True if i==0 else False
                    with gr.Accordion(f'Reference unit {i+1}', visible=True, elem_classes='control-unit') as unit_ui:
                        with gr.Row():
                            enabled_cb = gr.Checkbox(enabled, label='Active', container=False, show_label=True, elem_id=f'control_unit-{i}-enabled')
                            model_id = gr.Dropdown(label="Reference", choices=reference.list_models(), value='Reference', visible=False, elem_id=f'control_unit-{i}-model_name')
                            model_strength = gr.Slider(label="CN Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0, visible=False, elem_id=f'control_unit-{i}-strength')
                            reset_btn = ui_components.ToolButton(value=ui_symbols.reset, elem_id=f'reference_unit-{i}-reset')
                            image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'], elem_id=f'reference_unit-{i}-upload')
                            image_reuse= ui_components.ToolButton(value=ui_symbols.reuse, elem_id=f'reference_unit-{i}-reuse')
                            image_preview = gr.Image(label="Input", show_label=False, type="pil", interactive=False, height=128, width=128, visible=False, elem_id=f'reference_unit-{i}-override')
                            btn_preview= ui_components.ToolButton(value=ui_symbols.preview, elem_id=f'reference_unit-{i}-preview')
                    units.append(unit.Unit(
                        unit_type = 'reference',
                        index = i,
                        enabled = enabled,
                        result_txt = result_txt,
                        enabled_cb = enabled_cb,
                        reset_btn = reset_btn,
                        process_id = process_id,
                        model_id = model_id,
                        model_strength = model_strength,
                        preview_process = preview_process,
                        preview_btn = btn_preview,
                        image_upload = image_upload,
                        image_reuse = image_reuse,
                        image_preview = image_preview,
                        extra_controls = extra_controls,
                        )
                    )
                    if i == 0:
                        units[-1].enabled = True # enable first unit in group

        with gr.Accordion('Control settings', open=False, elem_classes=['control-settings']) as _tab_settings:
            with gr.Group(elem_classes=['processor-group']):
                settings = []
                with gr.Accordion('Global', open=True, elem_classes=['processor-settings']):
                    control_max_units = gr.Slider(label="Maximum units", minimum=1, maximum=10, step=1, value=shared.opts.control_max_units, elem_id='control_max_units')
                    def set_control_max_units(value):
                        shared.opts.control_max_units = value
                    control_max_units.change(fn=set_control_max_units, inputs=[control_max_units], outputs=[])
                    control_tiles = gr.Textbox(label="Tiling options", value=shared.opts.control_tiles, elem_id='control_tiles')
                    def set_control_tiles(value):
                        shared.opts.control_tiles = value
                    control_tiles.change(fn=set_control_tiles, inputs=[control_tiles], outputs=[])
                    control_hires = gr.Checkbox(label="Hires use control", value=shared.opts.control_hires, elem_id='control_hires')
                    def set_control_hires(value):
                        shared.opts.control_hires = value
                    control_hires.change(fn=set_control_hires, inputs=[control_hires], outputs=[])
                    control_aspect_ratio = gr.Checkbox(label="Keep aspect ratio", value=shared.opts.control_aspect_ratio, elem_id='control_aspect_ratio')
                    def set_control_aspect_ratio(value):
                        shared.opts.control_aspect_ratio = value
                    control_aspect_ratio.change(fn=set_control_aspect_ratio, inputs=[control_aspect_ratio], outputs=[])
                    control_move_processor = gr.Checkbox(label="Offload processor", value=shared.opts.control_move_processor, elem_id='control_move_processor')
                    def set_control_move_processor(value):
                        shared.opts.control_move_processor = value
                    control_move_processor.change(fn=set_control_move_processor, inputs=[control_move_processor], outputs=[])
                    control_unload_processor = gr.Checkbox(label="Unload processor", value=shared.opts.control_unload_processor, elem_id='control_unload_processor')
                    def set_control_unload_processor(value):
                        shared.opts.control_unload_processor = value
                    control_unload_processor.change(fn=set_control_unload_processor, inputs=[control_unload_processor], outputs=[])

                with gr.Accordion('HED', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Scribble", value=False))
                with gr.Accordion('Midas depth', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Slider(label="Background threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                    settings.append(gr.Checkbox(label="Depth and normal", value=False))
                with gr.Accordion('MLSD', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Slider(label="Score threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                    settings.append(gr.Slider(label="Distance threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                with gr.Accordion('OpenBody', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Body", value=True))
                    settings.append(gr.Checkbox(label="Hands", value=False))
                    settings.append(gr.Checkbox(label="Face", value=False))
                with gr.Accordion('PidiNet', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Scribble", value=False))
                    settings.append(gr.Checkbox(label="Apply filter", value=False))
                with gr.Accordion('LineArt', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Coarse", value=False))
                with gr.Accordion('Leres Depth', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Boost", value=False))
                    settings.append(gr.Slider(label="Near threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.0))
                    settings.append(gr.Slider(label="Depth threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.0))
                with gr.Accordion('MediaPipe Face', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Slider(label="Max faces", minimum=1, maximum=10, step=1, value=1))
                    settings.append(gr.Slider(label="Face confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.5))
                with gr.Accordion('Canny', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Slider(label="Low threshold", minimum=0, maximum=1000, step=1, value=100))
                    settings.append(gr.Slider(label="High threshold", minimum=0, maximum=1000, step=1, value=200))
                with gr.Accordion('DWPose', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Radio(label="Pose Model", choices=['Tiny', 'Medium', 'Large'], value='Tiny'))
                    settings.append(gr.Slider(label="Pose confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.3))
                with gr.Accordion('SegmentAnything', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Radio(label="Segment Model", choices=['Base', 'Large'], value='Base'))
                with gr.Accordion('Edge', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Parameter free", value=True))
                    settings.append(gr.Radio(label="Edge mode", choices=['edge', 'gradient'], value='edge'))
                with gr.Accordion('Zoe Depth', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Checkbox(label="Gamma corrected", value=False))
                with gr.Accordion('Marigold Depth', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Dropdown(label="Color map", choices=['None'] + plt.colormaps(), value='None'))
                    settings.append(gr.Slider(label="Denoising steps", minimum=1, maximum=99, step=1, value=10))
                    settings.append(gr.Slider(label="Ensemble size", minimum=1, maximum=99, step=1, value=10))
                with gr.Accordion('Depth Anything', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Dropdown(label="Depth map", choices=['none'] + masking.COLORMAP, value='inferno'))
                with gr.Accordion('Depth Pro', open=True, elem_classes=['processor-settings']):
                    settings.append(gr.Dropdown(label="Depth map", choices=['none'] + masking.COLORMAP, value='inferno'))
                for setting in settings:
                    setting.change(fn=processors.update_settings, inputs=settings, outputs=[])
