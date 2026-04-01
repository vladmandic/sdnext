import gradio as gr
from modules import scripts_manager, shared, ui_common, postprocessing, call_queue, generation_parameters_copypaste
from modules.logger import log


def submit_info(image):
    from modules.extras import run_pnginfo
    from modules.ui_common import infotext_to_html
    _, geninfo, info = run_pnginfo(image)
    if hasattr(scripts_manager, 'scripts_postproc'):
        scripts_manager.scripts_postproc.image_changed()
    return infotext_to_html(geninfo), info, geninfo


def submit_process(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, save_output, *script_inputs):
    from modules.ui_common import infotext_to_html
    result_images, geninfo, _js_info = postprocessing.run_postprocessing(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, *script_inputs, save_output=save_output)
    return result_images, geninfo, infotext_to_html(geninfo)


def create_ui():
    log.debug('UI initialize: tab=process')
    tab_index = gr.State(value=0) # pylint: disable=abstract-class-instantiated
    with gr.Row(equal_height=False, variant='compact', elem_classes="extras", elem_id="extras_tab"):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.Tab('Process Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    with gr.Row():
                        extras_image = gr.Image(label="Source", interactive=True, type="pil", elem_id="extras_image")
                with gr.Tab('Process Batch', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch process", interactive=True, elem_id="extras_image_batch")
                with gr.Tab('Process Folder', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")
            with gr.Row():
                save_output = gr.Checkbox(label='Save output', value=True, elem_id="extras_save_output")

            script_inputs = scripts_manager.scripts_postproc.setup_ui()
        with gr.Column():
            id_part = 'extras'
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", variant='secondary')
                interrupt.click(fn=shared.state.interrupt, inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", variant='secondary')
                skip.click(fn=shared.state.skip, inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=shared.state.pause, _js='checkPaused', inputs=[], outputs=[])
            result_images, generation_info, _html_info, html_info_formatted, _html_log = ui_common.create_output_panel("extras")
            gr.HTML('File metadata')
            exif_info = gr.HTML(elem_id="pnginfo_html_info")
            with gr.Row(elem_id='copy_buttons_process'):
                copy_process_buttons = generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "control", "caption"])

        for tabname, button in copy_process_buttons.items():
            generation_parameters_copypaste.register_paste_params_button(generation_parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=extras_image))
        generation_parameters_copypaste.add_paste_fields("extras", extras_image, None)

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])
    extras_image.change(fn=submit_info, inputs=[extras_image], outputs=[html_info_formatted, exif_info, generation_info])
    submit.click(
        _js="submit_postprocessing",
        fn=call_queue.wrap_gradio_gpu_call(submit_process, extra_outputs=[None, ''], name='Postprocess'),
        inputs=[
            tab_index,
            extras_image,
            image_batch,
            extras_batch_input_dir,
            extras_batch_output_dir,
            show_extras_results,
            save_output,
            *script_inputs,
        ],
        outputs=[
            result_images,
            generation_info,
            html_info_formatted,
        ]
    )
