import os
import gradio as gr
from modules import ui_sections, ui_symbols
from modules.ui_components import ToolButton
from modules.logger import log
from modules.video_models.models_def import models
from modules.ltx import ltx_process, ltx_capabilities


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def _model_change(model_name: str):
    caps = ltx_capabilities.get_caps(model_name)
    if caps is None:
        return (
            gr.update(visible=False),   # input_media_accordion
            gr.update(visible=False),   # tab_start_video
            gr.update(visible=False),   # tab_start_batch
            gr.update(value='Image'),   # start_source_mode (reset to default tab)
            gr.update(visible=False),   # upsample_accordion
            gr.update(visible=False),   # refine_accordion
            gr.update(value=False),     # upsample_enable (reset)
            gr.update(value=False),     # refine_enable (reset)
            gr.update(interactive=True),  # refine_strength
            gr.update(),                # guidance_scale
            gr.update(interactive=True),  # steps (reset; previous selection may have locked it)
            gr.update(),                # sampler_shift
            gr.update(),                # dynamic_shift
            gr.update(interactive=False),  # decode_timestep
            gr.update(interactive=False),  # image_cond_noise_scale
            gr.update(visible=False),   # audio_accordion
        )
    # 2.x refine runs fixed canonical sigmas regardless of the slider value; refine_strength
    # only feeds 0.9.x LTXConditionPipeline's denoise_strength. Disable elsewhere so the slider
    # reflects what the pipeline actually consumes.
    refine_strength_interactive = caps.family == '0.9'
    # 2.x Distilled enforces the 8-value DISTILLED_SIGMA_VALUES schedule and drops num_inference_steps
    # (see ltx_process._latent_pass and the no-latent-path stage 1). Disable the slider so the
    # displayed value matches what runs.
    steps_locked = caps.family == '2.x' and caps.is_distilled
    # Default Refine on for any 2.x variant; auto_refine_upsample at ltx_process.py:183 couples
    # the stages once Refine is on. Condition variants currently follow the same canonical path.
    refine_default = caps.supports_two_stage_refine
    return (
        gr.update(visible=caps.supports_input_media),   # input_media_accordion
        gr.update(visible=caps.supports_multi_condition),  # tab_start_video
        gr.update(visible=caps.supports_multi_condition),  # tab_start_batch
        gr.update(value='Image'),                       # start_source_mode (reset; Image is always available)
        gr.update(visible=True),                        # upsample_accordion
        gr.update(visible=True),                        # refine_accordion
        gr.update(value=False),                         # upsample_enable
        gr.update(value=refine_default),                # refine_enable
        gr.update(interactive=refine_strength_interactive),  # refine_strength
        gr.update(value=caps.default_cfg),              # guidance_scale
        gr.update(value=caps.default_steps, interactive=not steps_locked),  # steps
        gr.update(value=caps.default_sampler_shift),    # sampler_shift
        gr.update(value=caps.default_dynamic_shift),    # dynamic_shift
        gr.update(interactive=caps.supports_decode_timestep),  # decode_timestep
        gr.update(interactive=caps.supports_image_cond_noise_scale),  # image_cond_noise_scale
        gr.update(visible=caps.supports_audio),         # audio_accordion
    )


def create_ui(prompt, negative, styles, overrides, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf):
    # Gradio sets <input disabled> when interactive=False but neither Standard UI nor Modern UI
    # dims the surrounding control on its own. Inline :has() rule matches the prompt_enhance
    # pattern (scripts/prompt_enhance.py:947) so the dim works in both UI modes.
    gr.HTML(
        '<style>'
        '#ltx_steps:has(input:disabled),'
        '#ltx_refine_strength:has(input:disabled),'
        '#ltx_decode_timestep:has(input:disabled),'
        '#ltx_image_cond_noise_scale:has(input:disabled)'
        '{opacity:0.5;cursor:not-allowed;}'
        '</style>'
    )
    with gr.Row():
        with gr.Column(variant='compact', elem_id="ltx_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="ltx_generate_btn", variant='primary', visible=False)
            with gr.Row():
                ltx_models = [m.name for m in models['LTX Video']] if 'LTX Video' in models else ['None']
                model = gr.Dropdown(label='LTX model', choices=ltx_models, value=ltx_models[0], elem_id="ltx_model")
            with gr.Accordion(open=False, label='Size', elem_id='ltx_size_accordion'):
                width, height = ui_sections.create_resolution_inputs('ltx', default_width=832, default_height=480)
                with gr.Row():
                    frames = gr.Slider(label='Frames', minimum=1, maximum=1024, step=1, value=121, elem_id='ltx_frames')
                    seed = gr.Number(label='Initial seed', value=-1, elem_id='ltx_seed', container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id='ltx_seed_random')
                    random_seed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[seed])
            input_media_accordion = gr.Accordion(open=False, label="Input media", elem_id='ltx_input_media_accordion', visible=False)
            with input_media_accordion:
                gr.HTML('<b>Start frames</b>', elem_id='ltx_start_frames_label')
                # Three input methods for the opening frames: single image, video segment, or
                # image sequence. Tabs make the mutual exclusion explicit; tab.select updates
                # start_source_mode so the backend honors only the active tab's input.
                with gr.Tabs(elem_id='ltx_start_source_tabs'):
                    with gr.Tab('Image', id='ltx_start_image_tab') as tab_start_image:
                        ltx_init_image = gr.Image(label='Image', elem_id='ltx_init_image', type='pil', image_mode='RGB', width=256, height=256)
                    with gr.Tab('Video', id='ltx_start_video_tab') as tab_start_video:
                        condition_video = gr.Video(label='Video', type='filepath', elem_id="ltx_condition_video", width=256, height=256, source='upload')
                        with gr.Row():
                            condition_video_frames = gr.Slider(label='LTX frames number', minimum=-1, maximum=1024, step=1, value=-1, elem_id="ltx_condition_video_frames")
                            condition_video_skip = gr.Slider(label='LTX frames skip', minimum=0, maximum=1024, step=1, value=0, elem_id="ltx_condition_video_skip")
                    with gr.Tab('Image Batch', id='ltx_start_batch_tab') as tab_start_batch:
                        condition_files = gr.Files(label="Image Batch", interactive=True, elem_id="ltx_condition_batch")
                start_source_mode = gr.Textbox(value='Image', visible=False, elem_id='ltx_start_source_mode')
                tab_start_image.select(fn=lambda: 'Image', outputs=[start_source_mode])
                tab_start_video.select(fn=lambda: 'Video', outputs=[start_source_mode])
                tab_start_batch.select(fn=lambda: 'Image Batch', outputs=[start_source_mode])
                ltx_condition_strength = gr.Slider(label='LTX input strength', minimum=0.0, maximum=1.0, step=0.05, value=1.0, elem_id='ltx_condition_strength')
                gr.HTML('<b>End frame</b>', elem_id='ltx_end_frame_label')
                with gr.Row():
                    last_image = gr.Image(label='Last image', elem_id='ltx_last_image', type='pil', image_mode='RGB', width=256, height=256)
            upsample_accordion = gr.Accordion(open=False, label="Upsample", elem_id='ltx_upsample_accordion')
            with upsample_accordion:
                with gr.Row():
                    upsample_enable = gr.Checkbox(label='LTX enable upsampling', value=False, elem_id="ltx_upsample_enable")
                    upsample_ratio = gr.Slider(label='LTX upsample ratio', minimum=1.0, maximum=4.0, step=0.1, value=2.0, elem_id="ltx_upsample_ratio")
            refine_accordion = gr.Accordion(open=False, label="Refine", elem_id='ltx_refine_accordion')
            with refine_accordion:
                with gr.Row():
                    refine_enable = gr.Checkbox(label='LTX enable refine', value=False, elem_id="ltx_refine_enable")
                    refine_strength = gr.Slider(label='LTX refine strength', minimum=0.1, maximum=1.0, step=0.05, value=0.4, elem_id="ltx_refine_strength")
            parameters_accordion = gr.Accordion(open=False, label="Advanced", elem_id='ltx_parameters_accordion')
            with parameters_accordion:
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "ltx", default_steps=40)
                with gr.Row():
                    guidance_scale = gr.Slider(label='LTX guidance scale', minimum=0.0, maximum=14.0, step=0.1, value=4.0, elem_id="ltx_guidance_scale")
                with gr.Row():
                    sampler_shift = gr.Slider(label='LTX sampler shift', minimum=-1.0, maximum=20.0, step=0.1, value=-1.0, elem_id="ltx_sampler_shift")
                    dynamic_shift = gr.Checkbox(label='LTX dynamic shift', value=False, elem_id="ltx_dynamic_shift")
                with gr.Row():
                    decode_timestep = gr.Slider(label='LTX decode timestep', minimum=0.0, maximum=1.0, step=0.01, value=0.05, elem_id="ltx_decode_timestep")
                    image_cond_noise_scale = gr.Slider(label='LTX image cond noise scale', minimum=0.0, maximum=1.0, step=0.005, value=0.025, elem_id="ltx_image_cond_noise_scale")
            audio_accordion = gr.Accordion(open=False, label="Audio", elem_id='ltx_audio_accordion', visible=False)
            with audio_accordion:
                with gr.Row():
                    audio_enable = gr.Checkbox(label='LTX save audio', value=True, elem_id="ltx_audio_enable")

        with gr.Column(elem_id='ltx-output-column', scale=2) as _column_output:
            with gr.Row():
                video = gr.Video(label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512, autoplay=False)
            with gr.Row():
                text = gr.HTML('', elem_id='ltx_generation_info', show_label=False)

    model.change(
        fn=_model_change,
        inputs=[model],
        outputs=[
            input_media_accordion,
            tab_start_video,
            tab_start_batch,
            start_source_mode,
            upsample_accordion,
            refine_accordion,
            upsample_enable,
            refine_enable,
            refine_strength,
            guidance_scale,
            steps,
            sampler_shift,
            dynamic_shift,
            decode_timestep,
            image_cond_noise_scale,
            audio_accordion,
        ],
    )

    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    video_inputs = [
        model,
        prompt, negative, styles,
        width, height, frames,
        steps, sampler_index,
        guidance_scale, sampler_shift, dynamic_shift,
        seed,
        upsample_enable, upsample_ratio,
        refine_enable, refine_strength,
        ltx_condition_strength, ltx_init_image, last_image, condition_files, condition_video, condition_video_frames, condition_video_skip,
        start_source_mode,
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
