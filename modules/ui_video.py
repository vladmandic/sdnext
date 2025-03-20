import gradio as gr
from modules import shared, sd_models, timer, images, ui_common, ui_sections, ui_symbols, call_queue, generation_parameters_copypaste
from modules.ui_components import ToolButton
from modules.video_models import models_def, video_utils, video_load


def engine_change(engine):
    found = [model.name for model in models_def.models.get(engine, [])]
    return gr.update(choices=found, value=found[0] if len(found) > 0 else None)


def model_change(engine, model):
    found = [model.name for model in models_def.models.get(engine, [])]
    selected = [m for m in models_def.models[engine] if m.name == model][0] if len(found) > 0 else None
    yield ['Video model loading',
            gr.update(visible='I2V' in selected.name) if selected else gr.update(visible=False),
            video_utils.get_url(selected.url if selected else None),
           ]
    if selected:
        if 'None' in selected.name:
            sd_models.unload_model_weights()
            msg = 'Video model unloaded'
        else:
            msg = video_load.load_model(selected)
    else:
        sd_models.unload_model_weights()
        msg = 'Video model unloaded'
    return [msg,
            gr.update(visible='I2V' in selected.name) if selected else gr.update(visible=False),
            video_utils.get_url(selected.url if selected else None),
           ]


def run_video(*args):
    engine, model = args[2], args[3]
    found = [model.name for model in models_def.models.get(engine, [])]
    selected = [m for m in models_def.models[engine] if m.name == model][0] if len(found) > 0 else None
    if selected and 'Hunyuan' in selected.name:
        from modules.video_models import run_hunyuan
        return run_hunyuan.generate(*args)
    elif selected and 'LTX' in selected.name:
        from modules.video_models import run_ltx
        return run_ltx.generate(*args)
    elif selected and 'Mochi' in selected.name:
        from modules.video_models import run_mochi
        return run_mochi.generate(*args)
    elif selected and 'Cog' in selected.name:
        from modules.video_models import run_cog
        return run_cog.generate(*args)
    elif selected and 'Allegro' in selected.name:
        from modules.video_models import run_allegro
        return run_allegro.generate(*args)
    shared.log.error(f'Video model not found: args={args}')
    return [], None, '', '', f'Video model not found: engine={engine} model={model}'


def create_ui():
    shared.log.debug('UI initialize: video')
    with gr.Blocks(analytics_enabled=False) as _video_interface:
        prompt, styles, negative, generate, _reprocess, paste, networks_button, _token_counter, _token_button, _token_counter_negative, _token_button_negative = ui_sections.create_toprow(is_img2img=False, id_part="video", negative_visible=True, reprocess_visible=False)
        prompt_image = gr.File(label="", elem_id="video_prompt_image", file_count="single", type="binary", visible=False)
        prompt_image.change(fn=images.image_data, inputs=[prompt_image], outputs=[prompt, prompt_image])

        with gr.Row(variant='compact', elem_id="video_extra_networks", elem_classes=["extra_networks_root"], visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, networks_button, 'video', skip_indexing=shared.opts.extra_network_skip_indexing)
            timer.startup.record('ui-networks')

        with gr.Row(elem_id="video_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="video_settings", scale=1):

                with gr.Row():
                    engine = gr.Dropdown(label='Engine', choices=list(models_def.models), value='None', elem_id="video_engine")
                    model = gr.Dropdown(label='Model', choices=[''], value=None, elem_id="video_model")
                with gr.Row():
                    url = gr.HTML(label='Model URL', elem_id='video_model_url', value='')
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('video', default_width=720, default_height=480)
                with gr.Row():
                    frames = gr.Slider(label='Frames', minimum=1, maximum=1024, step=1, value=15, elem_id="video_frames")
                    seed = gr.Number(label='Initial seed', value=-1, elem_id="video_seed", container=True)
                    random_seed = ToolButton(ui_symbols.random, elem_id="video_random_seed", label='Random seed')
                    reuse_seed = ToolButton(ui_symbols.reuse, elem_id="video_reuse_seed", label='Reuse seed')
                steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "video")
                with gr.Row():
                    sampler_shift = gr.Slider(label='Sampler shift', minimum=0.0, maximum=20.0, step=0.1, value=7.0, elem_id="video_scheduler_shift")
                    dynamic_shift = gr.Checkbox(label='Dynamic shift', value=False, elem_id="video_dynamic_shift", interactive=False) # TODO video: dynamic shift
                with gr.Row():
                    guidance_scale = gr.Slider(label='Guidance scale', minimum=0.0, maximum=14.0, step=0.1, value=6.0, elem_id="video_guidance_scale")
                    guidance_true = gr.Slider(label='True guidance', minimum=0.0, maximum=14.0, step=0.1, value=1.0, elem_id="video_guidance_true")
                with gr.Row():
                    vae_type = gr.Dropdown(label='VAE decode', choices=['Default', 'Tiny', 'Remote'], value='Default', elem_id="video_vae_type")
                    vae_tile_frames = gr.Slider(label='Tile frames', minimum=1, maximum=64, step=1, value=16, elem_id="video_vae_tile_frames")
                with gr.Row():
                    with gr.Group(visible=False, elem_id='video_init_image') as image_group:
                        gr.HTML("<br>&nbsp Init image")
                        init_image = gr.Image(elem_id="video_image", show_label=False, type="pil", image_mode="RGB", height=512)
                with gr.Row():
                    save_frames = gr.Checkbox(label='Save image frames', value=False, elem_id="video_save_frames")
                with gr.Row():
                    video_type, video_duration, video_loop, video_pad, video_interpolate = ui_sections.create_video_inputs(tab='video')
                override_settings = ui_common.create_override_inputs('video')

            # output panel with gallery and video tabs
            with gr.Column(elem_id='video-output-column', scale=2) as _column_output:
                with gr.Tabs(elem_classes=['video-output-tabs'], elem_id='video-output-tabs'):
                    with gr.Tab('Frames', id='out-gallery'):
                        gallery, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("video", prompt=prompt, preview=False, transfer=False, scale=2)
                    with gr.Tab('Video', id='out-video'):
                        video = gr.Video(label="Output", show_label=False, elem_id='control_output_video', elem_classes=['control-image'], height=512, autoplay=False)

            # connect reuse seed button
            ui_common.connect_reuse_seed(seed, reuse_seed, gen_info, is_subseed=False)
            random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
            # handle engine and model change
            engine.change(fn=engine_change, inputs=[engine], outputs=[model])
            model.change(fn=model_change, inputs=[engine, model], outputs=[html_log, image_group, url])
            # setup extra networks
            ui_extra_networks.setup_ui(extra_networks_ui, gallery)

        # handle restore fields
        paste_fields = [
            (prompt, "Prompt"),
            # main
            (width, "Size-1"),
            (height, "Size-2"),
            (frames, "Frames"),
        ]
        generation_parameters_copypaste.add_paste_fields("video", None, paste_fields, override_settings)
        bindings = generation_parameters_copypaste.ParamBinding(paste_button=paste, tabname="video", source_text_component=prompt, source_image_component=None)
        generation_parameters_copypaste.register_paste_params_button(bindings)
        # hidden fields
        task_id = gr.Textbox(visible=False, value='')
        ui_state = gr.Textbox(visible=False, value='')
        # generate args
        video_args = [
            task_id, ui_state,
            engine, model,
            prompt, negative, styles,
            width, height,
            frames,
            steps, sampler_index,
            sampler_shift, dynamic_shift,
            seed,
            guidance_scale, guidance_true,
            init_image,
            vae_type, vae_tile_frames,
            save_frames,
            video_type, video_duration, video_loop, video_pad, video_interpolate,
            override_settings,
        ]
        # generate function
        video_dict = dict(
            fn=call_queue.wrap_gradio_gpu_call(run_video, extra_outputs=[None, '', ''], name='Video'),
            _js="submit_video",
            inputs=video_args,
            outputs=[gallery, video, gen_info, html_info, html_log],
            show_progress=False,
        )
        prompt.submit(**video_dict)
        generate.click(**video_dict)
