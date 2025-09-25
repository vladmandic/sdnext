import os
import gradio as gr
from modules import shared, ui_sections, ui_symbols, ui_common
from modules.ui_components import ToolButton
from modules.video_models.video_utils import get_codecs
from modules.video_models.models_def import models
from modules.ltx import ltx_process


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_model(model_name):
    if model_name is None or model_name == 'None':
        shared.log.info('LTX model unload')
        from modules import sd_models
        sd_models.unload_model_weights()
        return
    else:
        model_instance = [m for m in models['LTX Video'] if m.name == model_name][0]
        from modules.video_models import video_load
        video_load.load_model(model_instance)


def create_ui(prompt, negative, styles, overrides):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="ltx_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="ltx_generate_btn", variant='primary', visible=False)
            with gr.Row():
                ltx_models = [m.name for m in models['LTX Video']]
                model = gr.Dropdown(label='LTX model', choices=ltx_models, value=ltx_models[0])
                model.change(fn=load_model, inputs=[model], outputs=[], show_progress=True)
            with gr.Accordion(open=True, label="LTX size", elem_id='ltx_generate_accordion'):
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('ltx', default_width=832, default_height=480)
                with gr.Row():
                    frames = gr.Slider(label='LTX frames', minimum=1, maximum=513, step=1, value=17, elem_id="ltx_frames")
                    seed = gr.Number(label='LTX seed', value=-1, elem_id="ltx_seed", container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id="ltx_seed_random")
            with gr.Accordion(open=False, label="Condition", elem_id='ltx_condition_accordion'):
                condition_strength = gr.Slider(label='LTX condition strength', minimum=0.1, maximum=1.0, step=0.05, value=0.8, elem_id="ltx_condition_image_strength")
                with gr.Tabs():
                    with gr.Tab('Image', id='ltx_condition_image_tab'):
                        condition_image = gr.Image(sources='upload', type="pil", label="Image", width=256, height=256, interactive=True, tool="editor", image_mode='RGB', elem_id="ltx_condition_image")
                    with gr.Tab('Video', id='ltx_condition_video_tab'):
                        condition_video = gr.Video(label='Video', type='filepath', elem_id="ltx_condition_video", width=256, height=256, source='upload')
                        with gr.Row():
                            condition_video_frames = gr.Slider(label='LTX frames number', minimum=-1, maximum=1024, step=1, value=-1, elem_id="ltx_condition_video_frames")
                            condition_video_skip = gr.Slider(label='LTX frames skip', minimum=0, maximum=1024, step=1, value=0, elem_id="ltx_condition_video_sip")
                    with gr.Tab('Gallery', id='ltx_condition_batch_tab'):
                        condition_files = gr.Files(label="Image Batch", interactive=True, elem_id="ltx_condition_batch")
            with gr.Accordion(open=False, label="Upsample", elem_id='ltx_upsample_accordion'):
                with gr.Row():
                    upsample_enable = gr.Checkbox(label='LTX enable upsampling', value=False, elem_id="ltx_upsample_enable")
                    upsample_ratio = gr.Slider(label='LTX upsample ratio', minimum=1.0, maximum=4.0, step=0.1, value=2.0, elem_id="ltx_upsample_ratio", interactive=False)
            with gr.Accordion(open=False, label="Refine", elem_id='ltx_refine_accordion'):
                with gr.Row():
                    refine_enable = gr.Checkbox(label='LTX enable refine', value=False, elem_id="ltx_refine_enable")
                    refine_strength = gr.Slider(label='LTX refine strength', minimum=0.1, maximum=1.0, step=0.05, value=0.4, elem_id="ltx_refine_strength")
            with gr.Accordion(label="Video", open=False):
                with gr.Row():
                    mp4_fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=24, step=1)
                    mp4_interpolate = gr.Slider(label="LTX interpolation", minimum=0, maximum=10, value=0, step=1)
                with gr.Row():
                    mp4_codec = gr.Dropdown(label="LTX codec", choices=['none', 'libx264'], value='libx264', type='value')
                    ui_common.create_refresh_button(mp4_codec, get_codecs, elem_id="framepack_mp4_codec_refresh")
                    mp4_ext = gr.Textbox(label="LTX format", value='mp4', elem_id="framepack_mp4_ext")
                    mp4_opt = gr.Textbox(label="LTX options", value='crf:16', elem_id="framepack_mp4_ext")
                with gr.Row():
                    mp4_video = gr.Checkbox(label='LTX save video', value=True, elem_id="framepack_mp4_video")
                    mp4_frames = gr.Checkbox(label='LTX save frames', value=False, elem_id="framepack_mp4_frames")
                    mp4_sf = gr.Checkbox(label='LTX save safetensors', value=False, elem_id="framepack_mp4_sf")
            with gr.Accordion(open=False, label="Advanced", elem_id='ltx_parameters_accordion'):
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "ltx", default_steps=50)
                with gr.Row():
                    decode_timestep = gr.Slider(label='LTX decode timestep', minimum=0.01, maximum=1.0, step=0.01, value=0.05, elem_id="ltx_decode_timestep")
                    image_cond_noise_scale = gr.Slider(label='Noise scale', minimum=0.01, maximum=1.0, step=0.01, value=0.025, elem_id="ltx_image_cond_noise_scale")

        with gr.Column(elem_id='ltx-output-column', scale=2) as _column_output:
            with gr.Row():
                video = gr.Video(label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512, autoplay=False)
                # video = gr.Gallery(value=[], label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512)
            with gr.Row():
                text = gr.HTML('', elem_id='ltx_generation_info', show_label=False)

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    video_inputs = [
        prompt, negative, styles,
        width, height, frames,
        steps, sampler_index, seed,
        upsample_enable, upsample_ratio,
        refine_enable, refine_strength,
        condition_strength, condition_image, condition_files, condition_video, condition_video_frames, condition_video_skip,
        decode_timestep, image_cond_noise_scale,
        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
        overrides,
    ]
    video_outputs = [
        video,
        text,
    ]

    video_dict = dict(
        fn=ltx_process.run_ltx,
        _js="submit_ltx",
        inputs=state_inputs + video_inputs,
        outputs=video_outputs,
        show_progress=False,
    )
    generate.click(**video_dict)
