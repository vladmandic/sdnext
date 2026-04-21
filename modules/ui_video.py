import os
import gradio as gr
from modules import shared, timer, images, ui_common, ui_sections, generation_parameters_copypaste
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def create_ui():
    log.debug('UI initialize: tab=video')
    with gr.Blocks(analytics_enabled=False) as _video_interface:
        prompt, styles, negative, generate_btn, _reprocess, paste, networks_button, _token_counter, _token_button, _token_counter_negative, _token_button_negative = ui_sections.create_toprow(
            is_img2img=False,
            id_part="video",
            negative_visible=True,
            reprocess_visible=False,
        )
        prompt_image = gr.File(label="", elem_id="video_prompt_image", file_count="single", type="binary", visible=False)
        prompt_image.change(fn=images.image_data, inputs=[prompt_image], outputs=[prompt, prompt_image])

        with gr.Row(variant='compact', elem_id="video_extra_networks", elem_classes=["extra_networks_root"], visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, networks_button, 'video', skip_indexing=shared.opts.extra_network_skip_indexing)
            ui_extra_networks.setup_ui(extra_networks_ui)
            timer.startup.record('ui-networks')

        with gr.Row(elem_id="video_interface", equal_height=False):
            with gr.Tabs(elem_classes=['video-tabs'], elem_id='video-tabs'):
                overrides = ui_common.create_override_inputs('video')
                with gr.Tab('Output', id='video-outputs-tab') as _video_outputs_tab:
                    from modules.video_models import video_ui
                    mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf = video_ui.create_ui_outputs()
                with gr.Tab('Generic', id='video-core-tab') as video_core_tab:
                    from modules.video_models import video_ui
                    engine, model, steps, sampler_index, width, height, frames, seed = video_ui.create_ui(
                        prompt, negative, styles, overrides,
                        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
                    )
                with gr.Tab('FramePack', id='framepack-tab') as framepack_tab:
                    from modules.framepack import framepack_ui
                    framepack_ui.create_ui(
                        prompt, negative, styles, overrides,
                        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
                    )
                with gr.Tab('LTX', id='ltx-tab') as ltx_tab:
                    from modules.ltx import ltx_ui
                    ltx_ui.create_ui(
                        prompt, negative, styles, overrides,
                        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
                    )

        paste_fields = [
            (prompt, "Prompt"), # cannot add more fields as they are not defined yet
            (negative, "Negative prompt"),
            (width, "Width"),
            (height, "Height"),
            (frames, "Frames"),
            (seed, "Seed"),
            (styles, "Styles"),
            (steps, "Steps"),
            (sampler_index, "Sampler"),
            (engine, "Engine"),
            (model, "Model"),
        ]
        generation_parameters_copypaste.add_paste_fields("video", None, paste_fields)
        bindings = generation_parameters_copypaste.ParamBinding(paste_button=paste, tabname="video", source_text_component=prompt, source_image_component=None)
        generation_parameters_copypaste.register_paste_params_button(bindings)

        current_tab = gr.Textbox(visible=False, value='video')
        video_core_tab.select(fn=lambda: 'video', inputs=[], outputs=[current_tab])
        framepack_tab.select(fn=lambda: 'framepack', inputs=[], outputs=[current_tab])
        ltx_tab.select(fn=lambda: 'ltx', inputs=[], outputs=[current_tab])

        generate_btn.click(fn=None, _js='submit_video_wrapper', inputs=[current_tab], outputs=[])

        # from framepack_api import create_api # pylint: disable=wrong-import-order
