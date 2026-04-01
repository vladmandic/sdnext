import os
import gradio as gr
from modules import ui_sections
from modules.logger import log
from modules.video_models.models_def import models
from modules.ltx import ltx_process


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def create_ui(prompt, negative, styles, overrides, init_image, init_strength, last_image, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, width, height, frames, seed):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="ltx_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="ltx_generate_btn", variant='primary', visible=False)
            with gr.Row():
                ltx_models = [m.name for m in models['LTX Video']] if 'LTX Video' in models else ['None']
                model = gr.Dropdown(label='LTX model', choices=ltx_models, value=ltx_models[0], elem_id="ltx_model")
            with gr.Accordion(open=False, label="Condition", elem_id='ltx_condition_accordion'):
                with gr.Tabs():
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
            with gr.Accordion(open=False, label="Advanced", elem_id='ltx_parameters_accordion'):
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "ltx", default_steps=50)
                with gr.Row():
                    decode_timestep = gr.Slider(label='LTX decode timestep', minimum=0.01, maximum=1.0, step=0.01, value=0.05, elem_id="ltx_decode_timestep")
                    image_cond_noise_scale = gr.Slider(label='Noise scale', minimum=0.01, maximum=1.0, step=0.01, value=0.025, elem_id="ltx_image_cond_noise_scale")
            with gr.Accordion(open=False, label="Audio", elem_id='ltx_audio_accordion'):
                with gr.Row():
                    audio_enable = gr.Checkbox(label='LTX enable audio', value=False, elem_id="ltx_audio_enable")

        with gr.Column(elem_id='ltx-output-column', scale=2) as _column_output:
            with gr.Row():
                video = gr.Video(label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512, autoplay=False)
                # video = gr.Gallery(value=[], label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512)
            with gr.Row():
                text = gr.HTML('', elem_id='ltx_generation_info', show_label=False)

    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    video_inputs = [
        model,
        prompt, negative, styles,
        width, height, frames,
        steps, sampler_index, seed,
        upsample_enable, upsample_ratio,
        refine_enable, refine_strength,
        init_strength, init_image, last_image, condition_files, condition_video, condition_video_frames, condition_video_skip,
        decode_timestep, image_cond_noise_scale,
        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
        audio_enable,
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
        show_progress='hidden',
    )
    generate.click(**video_dict)
