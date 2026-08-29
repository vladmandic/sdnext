import os
import gradio as gr
from modules import ui_sections, ui_symbols
from modules.ui_components import ToolButton
from modules.logger import log
from modules.video_models.models_def import models
from modules.minimax import minimax_video, minimax_references


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def create_ui(prompt, _negative, styles, overrides, script_inputs, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb, mp4_scale, mp4_upscaler):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="minimax_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="minimax_generate_btn", variant='primary', visible=False)
            with gr.Row():
                minimax_models = [m.name for m in models['MiniMax']] if 'MiniMax' in models else ['None']
                model = gr.Dropdown(label='MiniMax model', choices=minimax_models, value=minimax_models[0], elem_id="minimax_model")
                btn_load = ToolButton(ui_symbols.loading, elem_id="video_model_load_minimax")
            with gr.Row():
                workflow = gr.Label(value='', label='Workflow', elem_id='minimax_workflow', show_label=False, elem_classes=['highlighted-label'])
            with gr.Accordion(open=True, label='Parameters', elem_id='minimax_param_accordion') as _param_accordion:
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('minimax', default_width=1024, default_height=576, step=32)
                with gr.Row():
                    steps = gr.Slider(minimum=2, maximum=100, step=1, label="MiniMax steps", elem_id='minimax_steps', value=30)
                    frames = gr.Slider(label='MiniMax frames', minimum=22, maximum=362, step=17, value=124, elem_id='minimax_frames')
                with gr.Row():
                    video_shift = gr.Slider(minimum=8.0, maximum=16.0, step=0.1, label="MiniMax video shift", elem_id='minimax_video_shift', value=12)
                    audio_shift = gr.Slider(minimum=1.5, maximum=6.0, step=0.1, label="MiniMax audio shift", elem_id='minimax_audio_shift', value=3)
                with gr.Row():
                    seed = gr.Number(label='Seed', value=-1, elem_id='minimax_seed', container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id='minimax_seed_random')
                    random_seed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[seed])
                with gr.Row():
                    audio_enable = gr.Checkbox(label='Enable audio', value=True, elem_id="minimax_audio_enable")
            with gr.Accordion(open=False, label="Input media", elem_id='minimax_input_media_accordion', visible=True) as input_accordion:
                with gr.Row():
                    init_image = gr.Image(label='Image', elem_id='minimax_init_image', type='pil', image_mode='RGB', width=256, height=256)
                with gr.Row():
                    last_image = gr.Image(label='Last image', elem_id='minimax_last_image', type='pil', image_mode='RGB', width=256, height=256)
            with gr.Accordion(open=False, label="Reference media", elem_id='minimax_reference_accordion', visible=True) as reference_accordion:
                caps = minimax_references.get_reference_caps('ref2va')
                gr.HTML(f"""Upload up to {caps.max_images} images, {caps.max_videos} videos, and {caps.max_audios} audio files<br>
                           The total number of files must not exceed {caps.max_references}<br><br>""", elem_id='minimax_reference_media_info', elem_classes=['smaller'])
                reference_media = gr.Files(label="Reference media", interactive=True, elem_id="minimax_reference_media", visible=True)

        with gr.Column(elem_id='minimax-output-column', scale=2) as _column_output:
            with gr.Row():
                video = gr.Video(label="Output", show_label=False, elem_id='minimax_output_video', elem_classes=['control-image'], height=512, autoplay=False)
            with gr.Row():
                text = gr.HTML('', elem_id='minimax_generation_info', show_label=False)

    def on_change(model_name: str, init_image):
        model_info = next((m for m in models['MiniMax'] if m.name == model_name), None)
        if model_info is None or model_info.name is None or model_info.name == '' or model_info.name == 'None':
            return gr.update(value='none'), gr.update(visible=False), gr.update(visible=False)
        log.debug(f'Selected: name="{model_info.name}" repo="{model_info.repo}" cls={model_info.repo_cls}')
        if model_info.workflow == 'fl2va':
            workflow = 'fl2va' if init_image is not None else 't2va'
        else:
            workflow = model_info.workflow
        return gr.update(value=f'Workflow: {workflow}'), gr.update(visible=workflow != 'ref2va'), gr.update(visible=workflow == 'ref2va')

    def on_load(model_name: str):
        model_info = next((m for m in models['MiniMax'] if m.name == model_name), None)
        minimax_video.load_model(model_info.name if model_info is not None else None)

    model.change(fn=on_change, inputs=[model, init_image], outputs=[workflow, input_accordion, reference_accordion], show_progress='hidden')
    init_image.change(fn=on_change, inputs=[model, init_image], outputs=[workflow, input_accordion, reference_accordion], show_progress='hidden')
    btn_load.click(fn=on_load, inputs=[model], outputs=[])

    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    video_inputs = [
        model, workflow,
        prompt, styles,
        width, height, frames,
        steps, seed,
        init_image, last_image, reference_media,
        video_shift, audio_shift,
        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt,
        mp4_video, mp4_frames, mp4_sf, mp4_thumb,
        mp4_scale, mp4_upscaler,
        audio_enable,
        overrides,
    ]
    video_outputs = [
        video,
        text,
    ]

    video_dict = dict(
        fn=minimax_video.generate,
        _js="submit_minimax",
        inputs=state_inputs + video_inputs + script_inputs,
        outputs=video_outputs,
        show_progress='hidden',
    )
    generate.click(**video_dict)
