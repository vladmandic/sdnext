import os
import gradio as gr
from modules import shared, ui_sections, ui_symbols
from modules.ui_components import ToolButton
from modules.ltx import ltx_process


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def create_ui(prompt, negative, styles, overrides):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="ltx_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="ltx_generate_btn", variant='primary', visible=False)
            with gr.Accordion(open=True, label="Size", elem_id='ltx_generate_accordion'):
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('ltx', default_width=704, default_height=512)
                with gr.Row():
                    frames = gr.Slider(label='Frames', minimum=1, maximum=513, step=1, value=17, elem_id="ltx_frames")
                    seed = gr.Number(label='Initial seed', value=-1, elem_id="ltx_seed", container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id="ltx_random_seed")
            with gr.Accordion(open=False, label="Condition", elem_id='ltx_condition_accordion'):
                with gr.Tabs():
                    with gr.Tab('Image', id='ltx_condition_image_tab'):
                        condition_image_strength = gr.Slider(label='Condition strength', minimum=0.1, maximum=1.0, step=0.05, value=0.8, elem_id="ltx_condition_image_strength")
                        condition_image = gr.Image(label='Image', type='filepath', elem_id="ltx_condition_image", visible=False)
                    with gr.Tab('Video', id='ltx_condition_video_tab'):
                        condition_video_strength = gr.Slider(label='Condition strength', minimum=0.1, maximum=1.0, step=0.05, value=0.8, elem_id="ltx_condition_video_strength")
                        condition_video_frames = gr.Slider(label='Condition frames', minimum=1, maximum=1024, step=1, value=15, elem_id="ltx_condition_video_frames")
                        condition_video = gr.Video(label='Video', type='filepath', elem_id="ltx_condition_video", visible=False)
            with gr.Accordion(open=False, label="Upsample", elem_id='ltx_upsample_accordion'):
                upsample_enable = gr.Checkbox(label='Enable upsampling', value=False, elem_id="ltx_upsample_enable")
                upsample_ratio = gr.Slider(label='Upsample ratio', minimum=1.0, maximum=4.0, step=0.1, value=2.0, elem_id="ltx_upsample_ratio", interactive=False)
            with gr.Accordion(open=False, label="Refine", elem_id='ltx_refine_accordion'):
                refine_enable = gr.Checkbox(label='Enable refinement', value=False, elem_id="ltx_refine_enable")
                refine_strength = gr.Slider(label='Refine strength', minimum=0.1, maximum=1.0, step=0.05, value=0.4, elem_id="ltx_refine_strength")
            with gr.Accordion(open=False, label="Advanced", elem_id='ltx_parameters_accordion'):
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "ltx", default_steps=50)
                with gr.Row():
                    decode_timestep = gr.Slider(label='Decode timestep', minimum=0.01, maximum=1.0, step=0.01, value=0.05, elem_id="ltx_decode_timestep")
                    image_cond_noise_scale = gr.Slider(label='Noise scale', minimum=0.01, maximum=1.0, step=0.01, value=0.025, elem_id="ltx_image_cond_noise_scale")

        with gr.Column(elem_id='ltx-output-column', scale=2) as _column_output:
            with gr.Row():
                # video = gr.Video(label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512, autoplay=False)
                video = gr.Gallery(value=[], label="Output", show_label=False, elem_id='ltx_output_video', elem_classes=['control-image'], height=512)
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
        condition_image_strength, condition_video_strength, condition_video_frames,
        condition_image, condition_video,
        decode_timestep, image_cond_noise_scale,
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
