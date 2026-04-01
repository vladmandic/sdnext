import os
import gradio as gr
from modules import sd_models, ui_common, ui_sections, ui_symbols, ui_video_vlm, call_queue
from modules.logger import log
from modules.ui_components import ToolButton
from modules.video_models import models_def, video_utils
from modules.video_models import video_run


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def engine_change(engine):
    debug(f'Video change: engine="{engine}"')
    found = [model.name for model in models_def.models.get(engine, [])]
    return gr.update(choices=found, value=found[0] if len(found) > 0 else None)


def get_selected(engine, model):
    found = [model.name for model in models_def.models.get(engine, [])]
    if len(models_def.models[engine]) > 0 and len(found) > 0:
        selected = [m for m in models_def.models[engine] if m.name == model][0]
        return selected
    return None


def model_change(engine, model):
    debug(f'Video change: engine="{engine}" model="{model}"')
    found = [model.name for model in models_def.models.get(engine, [])]
    selected = [m for m in models_def.models[engine] if m.name == model][0] if len(found) > 0 else None
    url = video_utils.get_url(selected.url if selected else None)
    return url


def model_load(engine, model):
    debug(f'Video load: engine="{engine}" model="{model}"')
    selected = get_selected(engine, model)
    yield f'Video model loading: {selected.name}'
    if selected:
        if 'None' in selected.name:
            sd_models.unload_model_weights()
            msg = 'Video model unloaded'
        else:
            from modules.video_models import video_load
            msg = video_load.load_model(selected)
    else:
        sd_models.unload_model_weights()
        msg = 'Video model unloaded'
    yield msg
    return msg


def run_video(*args):
    engine, model = args[2], args[3]
    debug(f'Video run: engine="{engine}" model="{model}"')
    selected = get_selected(engine, model)
    if not selected or engine is None or model is None or engine == 'None' or model == 'None':
        return video_utils.queue_err('model not selected')
    debug(f'Video run: {str(selected)}')
    if selected and 'Hunyuan' in selected.name:
        return video_run.generate(*args)
    elif selected and 'LTX' in selected.name:
        return video_run.generate(*args)
    elif selected and 'Mochi' in selected.name:
        return video_run.generate(*args)
    elif selected and 'Cog' in selected.name:
        return video_run.generate(*args)
    elif selected and 'Allegro' in selected.name:
        return video_run.generate(*args)
    elif selected and 'WAN' in selected.name:
        return video_run.generate(*args)
    elif selected and 'Latte' in selected.name:
        return video_run.generate(*args)
    elif selected and 'anisora' in selected.name.lower():
        return video_run.generate(*args)
    elif selected and 'Kandinsky' in selected.name:
        return video_run.generate(*args)
    return video_utils.queue_err(f'model not found: engine="{engine}" model="{model}"')


def create_ui_inputs():
    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_inputs", elem_classes=['settings-column'], scale=1):
            init_strength = gr.Slider(label='Init strength', minimum=0.0, maximum=1.0, step=0.01, value=0.8, elem_id="video_denoising_strength")
            gr.HTML("<br>&nbsp Init image")
            init_image = gr.Image(elem_id="video_image", show_label=False, type="pil", image_mode="RGB", width=256, height=256)
            gr.HTML("<br>&nbsp Last image")
            last_image = gr.Image(elem_id="video_last", show_label=False, type="pil", image_mode="RGB", width=256, height=256)
    return init_image, init_strength, last_image


def create_ui_outputs():
    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_outputs", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                mp4_fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=24, step=1)
                mp4_interpolate = gr.Slider(label="Video interpolation", minimum=0, maximum=10, value=0, step=1)
            with gr.Row():
                mp4_codec = gr.Dropdown(label="Video codec", choices=['none', 'libx264'], value='libx264', type='value')
                ui_common.create_refresh_button(mp4_codec, video_utils.get_codecs, elem_id="framepack_mp4_codec_refresh")
                mp4_ext = gr.Textbox(label="Video format", value='mp4', elem_id="framepack_mp4_ext")
                mp4_opt = gr.Textbox(label="Video options", value='crf:16', elem_id="framepack_mp4_opt")
            with gr.Row():
                mp4_video = gr.Checkbox(label='Video save video', value=True, elem_id="framepack_mp4_video")
                mp4_frames = gr.Checkbox(label='Video save frames', value=False, elem_id="framepack_mp4_frames")
                mp4_sf = gr.Checkbox(label='Video save safetensors', value=False, elem_id="framepack_mp4_sf")
    return mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf


def create_ui_size():
    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_size", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                width, height = ui_sections.create_resolution_inputs('video', default_width=832, default_height=480)
            with gr.Row():
                frames = gr.Slider(label='Frames', minimum=1, maximum=1024, step=1, value=17, elem_id="video_frames")
                seed = gr.Number(label='Initial seed', value=-1, elem_id="video_seed", container=True)
                random_seed = ToolButton(ui_symbols.random, elem_id="video_seed_random")
                reuse_seed = ToolButton(ui_symbols.reuse, elem_id="video_seed_reuse")
                random_seed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[seed])
    return width, height, frames, seed, reuse_seed


def create_ui(prompt, negative, styles, overrides, init_image, init_strength, last_image, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, width, height, frames, seed, reuse_seed):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="video_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="video_generate_btn", variant='primary', visible=False)
            with gr.Row():
                engine = gr.Dropdown(label='Video engine', choices=list(models_def.models), value='None', elem_id="video_engine")
                model = gr.Dropdown(label='Video model', choices=[''], value='None', elem_id="video_model")
                btn_load = ToolButton(ui_symbols.loading, elem_id="video_model_load")
            with gr.Row():
                url = gr.HTML(label='Model URL', elem_id='video_model_url', value='<br><br>')
            with gr.Accordion(open=False, label="Parameters", elem_id='video_parameters_accordion'):
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "video", default_steps=50)
                with gr.Row():
                    sampler_shift = gr.Slider(label='Sampler shift', minimum=-1.0, maximum=20.0, step=0.1, value=-1.0, elem_id="video_scheduler_shift")
                    dynamic_shift = gr.Checkbox(label='Dynamic shift', value=False, elem_id="video_dynamic_shift")
                with gr.Row():
                    guidance_scale = gr.Slider(label='Guidance scale', minimum=-1.0, maximum=14.0, step=0.1, value=-1.0, elem_id="video_guidance_scale")
                    guidance_true = gr.Slider(label='True guidance', minimum=-1.0, maximum=14.0, step=0.1, value=-1.0, elem_id="video_guidance_true")
            with gr.Accordion(open=False, label="Decode", elem_id='video_decode_accordion'):
                with gr.Row():
                    vae_type = gr.Dropdown(label='VAE decode', choices=['Default', 'Tiny', 'Remote', 'Upscale'], value='Default', elem_id="video_vae_type")
                    vae_tile_frames = gr.Slider(label='Tile frames', minimum=1, maximum=64, step=1, value=16, elem_id="video_vae_tile_frames")

            vlm_enhance, vlm_model, vlm_system_prompt = ui_video_vlm.create_ui(prompt_element=prompt, image_element=init_image)

        # output panel with gallery and video tabs
        with gr.Column(elem_id='video-output-column', scale=2) as _column_output:
            with gr.Tabs(elem_classes=['video-output-tabs'], elem_id='video-output-tabs'):
                with gr.Tab('Video', id='out-video'):
                    video = gr.Video(label="Output", show_label=False, elem_id='control_output_video', elem_classes=['control-image'], height=512, autoplay=False)
                with gr.Tab('Frames', id='out-gallery'):
                    gallery, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("video", prompt=prompt, preview=False, transfer=False, scale=2)

    # connect reuse seed button
    ui_common.connect_reuse_seed(seed, reuse_seed, gen_info, is_subseed=False)
    # handle engine and model change
    engine.change(fn=engine_change, inputs=[engine], outputs=[model])
    model.change(fn=model_change, inputs=[engine, model], outputs=[url])
    btn_load.click(fn=model_load, inputs=[engine, model], outputs=[html_log])
    # hidden fields
    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    # generate args
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
        vae_type, vae_tile_frames,
        mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
        vlm_enhance, vlm_model, vlm_system_prompt,
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
        inputs=state_inputs + video_inputs,
        outputs=video_outputs,
        show_progress='hidden',
    )
    generate.click(**video_dict)
    return [engine, model, steps, sampler_index]
