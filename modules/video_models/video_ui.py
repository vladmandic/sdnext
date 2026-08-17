import os
import gradio as gr
from modules import sd_models, ui_common, ui_sections, ui_symbols, call_queue
from modules.logger import log
from modules.ui_components import ToolButton
from modules.video_models import models_def, video_utils
from modules.video_models import video_run


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None

# Engines surfaced on their own dedicated tab; hide from the general Video tab dropdown
# so users aren't offered two paths to the same models.
HIDDEN_ENGINES = {'LTX Video', 'MiniMax'}


def visible_engines():
    return [name for name in models_def.models if name not in HIDDEN_ENGINES]


def engine_change(engine):
    debug(f'Video change: engine="{engine}"')
    found = [model.name for model in models_def.models.get(engine, [])]
    return gr.update(choices=found, value=found[0] if len(found) > 0 else None)


def get_selected(engine, model):
    return models_def.find(engine, model)


def model_change(engine, model):
    debug(f'Video change: engine="{engine}" model="{model}"')
    selected = get_selected(engine, model)
    url = video_utils.get_url(selected.url if selected else None)
    return url


def model_load(engine, model):
    debug(f'Load video: engine="{engine}" model="{model}"')
    selected = get_selected(engine, model)
    if selected is None: # the dropdown lists the separators it groups models under, and they name no model
        if model and model.startswith('─'):
            msg = 'Video model not loaded: dropdown separator selected'
        elif model in (None, '', 'None'):
            sd_models.unload_model_weights()
            msg = 'Video model unloaded'
        else:
            msg = f'Video model not found: engine="{engine}" model="{model}"'
        log.warning(msg)
        yield msg
        return
    yield f'Video model loading: {selected.name}'
    from modules.video_models import video_load
    msg = video_load.load_model(selected)
    yield msg


def create_ui_outputs():
    from modules.video_models import video_codecs
    default_codec = 'libx264'
    def on_codec_change(codec):
        cfg = video_codecs.get_codec_dict(codec)
        if not cfg:
            return gr.update(value='unknown codec'), gr.update(value='mp4'), gr.update(value='')
        return gr.update(value=cfg['name']), gr.update(value=cfg['ext'], choices=cfg['allowed_exts']), gr.update(value=cfg['options'])

    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_outputs", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                mp4_fps = gr.Slider(label="Target FPS", minimum=1, maximum=60, value=24, step=1)
                mp4_interpolate = gr.Slider(label="Video interpolation", minimum=0, maximum=10, value=0, step=1)
            with gr.Row():
                mp4_codec = gr.Dropdown(label="Video codec", choices=video_codecs.get_codec_list(), value=default_codec, type='value')
                ui_common.create_refresh_button(mp4_codec, video_utils.get_codecs, elem_id="video_mp4_codec_refresh")
                mp4_info = gr.Label(value=video_codecs.get_codec_name(default_codec), label='Codec info', elem_id='video_mp4_codec_label', show_label=False, elem_classes=['codec-label'])
            with gr.Row():
                # mp4_ext = gr.Textbox(label="Video format", value='mp4', elem_id="video_mp4_ext")
                mp4_ext = gr.Dropdown(label="Video format", choices=video_codecs.get_codec_allowed_exts(default_codec), value=video_codecs.get_codec_ext(default_codec), elem_id="video_mp4_ext")
                mp4_opt = gr.Textbox(label="FFmpeg options", value=video_codecs.get_codec_options(default_codec), elem_id="video_mp4_opt")
            with gr.Row():
                mp4_video = gr.Checkbox(label='Save: video', value=True, elem_id="video_mp4_video")
                mp4_frames = gr.Checkbox(label='Save: frames', value=False, elem_id="video_mp4_frames")
                mp4_sf = gr.Checkbox(label='Save: safetensors', value=False, elem_id="video_mp4_sf")
                mp4_thumb = gr.Checkbox(label='Save: thumbnail', value=True, elem_id="video_mp4_thumb")
            mp4_codec.change(fn=on_codec_change, inputs=[mp4_codec], outputs=[mp4_info, mp4_ext, mp4_opt], show_progress='hidden')
    return mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb


def create_ui(prompt, negative, styles, overrides, script_inputs, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="video_generate_btn", variant='primary', visible=False)
            with gr.Row():
                engine = gr.Dropdown(label='Video engine', choices=visible_engines(), value='None', elem_id="video_engine")
                model = gr.Dropdown(label='Video model', choices=[''], value='None', elem_id="video_model")
                btn_load = ToolButton(ui_symbols.loading, elem_id="video_model_load")
            url = gr.HTML(label='Model URL', elem_id='video_model_url', value='<br><br>')
            with gr.Accordion(open=False, label="Parameters", elem_id='video_params_accordion'):
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('video', default_width=1024, default_height=576, step=16)
                with gr.Row():
                    frames = gr.Slider(label='Frames', minimum=1, maximum=1024, step=1, value=17, elem_id="video_frames")
                    seed = gr.Number(label='Initial seed', value=-1, elem_id="video_seed", container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id="video_seed_random")
                    reuse_seed = ToolButton(ui_symbols.reuse, elem_id="video_seed_reuse")
                    random_seed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[seed])
                with gr.Row():
                    audio = gr.Checkbox(label='Audio Enabled', value=True, elem_id="video_audio")
            with gr.Accordion(open=False, label="Advanced", elem_id='video_advanced_accordion'):
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "video", default_steps=30)
                with gr.Row():
                    sampler_shift = gr.Slider(label='Sampler shift', minimum=-1.0, maximum=20.0, step=0.1, value=-1.0, elem_id="video_scheduler_shift")
                    dynamic_shift = gr.Checkbox(label='Dynamic shift', value=False, elem_id="video_dynamic_shift")
                with gr.Row():
                    guidance_scale = gr.Slider(label='Guidance scale', minimum=-1.0, maximum=14.0, step=0.1, value=-1.0, elem_id="video_guidance_scale")
                    guidance_true = gr.Slider(label='True guidance', minimum=-1.0, maximum=14.0, step=0.1, value=-1.0, elem_id="video_guidance_true")
            with gr.Accordion(open=False, label="Inputs", elem_id='video_inputs_accordion'):
                init_strength = gr.Slider(label='Init strength', minimum=0.0, maximum=1.0, step=0.01, value=0.8, elem_id="video_denoising_strength")
                gr.HTML("<br>&nbsp Init image")
                init_image = gr.Image(elem_id="video_image", show_label=False, type="pil", image_mode="RGB", width=256, height=256)
                gr.HTML("<br>&nbsp Last image")
                last_image = gr.Image(elem_id="video_last", show_label=False, type="pil", image_mode="RGB", width=256, height=256)
            with gr.Accordion(open=False, label="Decode", elem_id='video_decode_accordion'):
                with gr.Row():
                    vae_type = gr.Dropdown(label='VAE decode', choices=['Default', 'Tiny', 'Remote', 'Upscale'], value='Default', elem_id="video_vae_type")
                    vae_tile_frames = gr.Slider(label='Tile frames', minimum=1, maximum=64, step=1, value=16, elem_id="video_vae_tile_frames")

        # output panel with gallery and video tabs
        with gr.Column(elem_id='video-output-column', scale=2) as _column_output:
            with gr.Tabs(elem_classes=['video-output-tabs'], elem_id='video-output-tabs'):
                with gr.Tab('Video', id='out-video'):
                    video = gr.Video(label="Output", show_label=False, elem_id='control_output_video', elem_classes=['control-image'], height=512, autoplay=False)
                with gr.Tab('Frames', id='out-gallery'):
                    gallery, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("video", prompt=prompt, preview=False, transfer=False, scale=2)

    ui_common.connect_reuse_seed(seed, reuse_seed, gen_info, is_subseed=False)
    engine.change(fn=engine_change, inputs=[engine], outputs=[model])
    model.change(fn=model_change, inputs=[engine, model], outputs=[url])
    btn_load.click(fn=model_load, inputs=[engine, model], outputs=[html_log])

    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    video_inputs = [
        engine, model,
        prompt, negative, styles,
        width, height,
        frames,
        steps, sampler_index,
        sampler_shift, dynamic_shift,
        seed,
        guidance_scale, guidance_true,
        init_image, init_strength, last_image,
        vae_type, vae_tile_frames, audio,
        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb,
        overrides,
    ]
    video_outputs = [
        gallery,
        video,
        gen_info,
        html_info,
        html_log,
    ]

    video_dict = dict(
        fn=call_queue.wrap_gradio_gpu_call(video_run.generate, extra_outputs=[gr.update(), gr.update(), gr.update(), gr.update()], name='Video'),
        _js="submit_video",
        inputs=state_inputs + video_inputs + script_inputs,
        outputs=video_outputs,
        show_progress='hidden',
    )
    generate.click(**video_dict)
    return engine, model, steps, sampler_index, width, height, frames, seed
