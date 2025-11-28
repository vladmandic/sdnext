import gradio as gr
from modules import ui_sections, ui_video_vlm
from modules.framepack import framepack_load
from modules.framepack.framepack_worker import get_latent_paddings
from modules.framepack.framepack_wrappers import load_model, unload_model
from modules.framepack.framepack_wrappers import run_framepack # pylint: disable=wrong-import-order


def change_sections(duration, mp4_fps, mp4_interpolate, latent_ws, variant):
    num_sections = len(get_latent_paddings(mp4_fps, mp4_interpolate, latent_ws, duration, variant))
    num_frames = (latent_ws * 4 - 3) * num_sections + 1
    return gr.update(value=f'Target video: {num_frames} frames in {num_sections} sections'), gr.update(lines=max(2, 2*num_sections//3))


def create_ui(prompt, negative, styles, _overrides, init_image, last_image, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf):
    with gr.Row():
        with gr.Column(variant='compact', elem_id="framepack_settings", elem_classes=['settings-column'], scale=1):
            with gr.Row():
                generate = gr.Button('Generate', elem_id="framepack_generate_btn", variant='primary', visible=False)
            with gr.Row():
                variant = gr.Dropdown(label="FP model variant", choices=list(framepack_load.models), value='bi-directional', type='value')
            with gr.Row():
                resolution = gr.Slider(label="FP resolution", minimum=240, maximum=1088, value=640, step=16)
                duration = gr.Slider(label="FP duration", minimum=1, maximum=120, value=4, step=0.1)
                mp4_fps = gr.Slider(label="FP target FPS", minimum=1, maximum=60, value=24, step=1)
                mp4_interpolate = gr.Slider(label="FP interpolation", minimum=0, maximum=10, value=0, step=1)
            with gr.Row():
                section_html = gr.HTML(show_label=False, elem_id="framepack_section_html")
            with gr.Accordion(label="Inputs", open=False):
                with gr.Row():
                    start_weight = gr.Slider(label="FP init strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, elem_id="framepack_start_weight")
                    end_weight = gr.Slider(label="FP end strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, elem_id="framepack_end_weight")
                    vision_weight = gr.Slider(label="FP vision strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, elem_id="framepack_vision_weight")
            with gr.Accordion(label="Sections", open=False):
                section_prompt = gr.Textbox(label="FP section prompts", elem_id="framepack_section_prompt", lines=2, placeholder="Optional one-line prompt suffix per each video section", interactive=True)
            with gr.Accordion(label="Advanced", open=False):
                seed = ui_sections.create_seed_inputs('control', reuse_visible=False, subseed_visible=False, accordion=False)[0]
                latent_ws = gr.Slider(label="FP latent window size", minimum=1, maximum=33, value=9, step=1)
                with gr.Row():
                    steps = gr.Slider(label="FP steps", minimum=1, maximum=100, value=25, step=1)
                    shift = gr.Slider(label="FP sampler shift", minimum=0.0, maximum=10.0, value=3.0, step=0.01)
                with gr.Row():
                    cfg_scale = gr.Slider(label="FP CFG scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                    cfg_distilled = gr.Slider(label="FP distilled CFG scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                    cfg_rescale = gr.Slider(label="FP CFG re-scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)

            vlm_enhance, vlm_model, vlm_system_prompt = ui_video_vlm.create_ui(prompt_element=prompt, image_element=init_image)

            with gr.Accordion(label="Model", open=False):
                with gr.Row():
                    btn_load = gr.Button(value="Load model", elem_id="framepack_btn_load", interactive=True)
                    btn_unload = gr.Button(value="Unload model", elem_id="framepack_btn_unload", interactive=True)
                with gr.Row():
                    system_prompt = gr.Textbox(label="FP system prompt", elem_id="framepack_system_prompt", lines=6, placeholder="Optional system prompt for the model", interactive=True)
                with gr.Row():
                    receipe = gr.Textbox(label="FP model receipe", elem_id="framepack_model_receipe", lines=6, placeholder="Model receipe", interactive=True)
                with gr.Row():
                    receipe_get = gr.Button(value="Get receipe", elem_id="framepack_btn_get_model", interactive=True)
                    receipe_set = gr.Button(value="Set receipe", elem_id="framepack_btn_set_model", interactive=True)
                    receipe_reset = gr.Button(value="Reset receipe", elem_id="framepack_btn_reset_model", interactive=True)
                use_teacache = gr.Checkbox(label='FP enable TeaCache', value=True)
                optimized_prompt = gr.Checkbox(label='FP use optimized system prompt', value=True)
                use_cfgzero = gr.Checkbox(label='FP enable CFGZero', value=False)
                use_preview = gr.Checkbox(label='FP enable Preview', value=True)
                attention = gr.Dropdown(label="FP attention", choices=['Default', 'Xformers', 'FlashAttention', 'SageAttention'], value='Default', type='value')
                vae_type = gr.Dropdown(label="FP VAE", choices=['Full', 'Tiny', 'Remote'], value='Local', type='value')

        with gr.Column(elem_id='framepack-output-column', scale=2) as _column_output:
            with gr.Tabs():
                with gr.TabItem("Video"):
                    result_video = gr.Video(label="Video", autoplay=True, show_share_button=False, height=512, loop=True, show_label=False, elem_id="framepack_result_video")
                with gr.Tab("Preview"):
                    preview_image = gr.Image(label="Current", height=512, show_label=False, elem_id="framepack_preview_image")
            progress_desc = gr.HTML('', show_label=False, elem_id="framepack_progress_desc")

    # hidden fields
    task_id = gr.Textbox(visible=False, value='')
    ui_state = gr.Textbox(visible=False, value='')
    state_inputs = [task_id, ui_state]

    framepack_outputs = [
        result_video,
        preview_image,
        progress_desc,
    ]

    duration.change(fn=change_sections, inputs=[duration, mp4_fps, mp4_interpolate, latent_ws, variant], outputs=[section_html, section_prompt])
    mp4_fps.change(fn=change_sections, inputs=[duration, mp4_fps, mp4_interpolate, latent_ws, variant], outputs=[section_html, section_prompt])
    mp4_interpolate.change(fn=change_sections, inputs=[duration, mp4_fps, mp4_interpolate, latent_ws, variant], outputs=[section_html, section_prompt])
    btn_load.click(fn=load_model, inputs=[variant, attention], outputs=framepack_outputs)
    btn_unload.click(fn=unload_model, outputs=framepack_outputs)
    receipe_get.click(fn=framepack_load.get_model, inputs=[], outputs=receipe)
    receipe_set.click(fn=framepack_load.set_model, inputs=[receipe], outputs=[])
    receipe_reset.click(fn=framepack_load.reset_model, inputs=[], outputs=[receipe])

    framepack_inputs=[
        init_image, last_image,
        start_weight, end_weight, vision_weight,
        prompt, system_prompt, optimized_prompt, section_prompt, negative, styles,
        seed,
        resolution,
        duration,
        latent_ws,
        steps,
        cfg_scale, cfg_distilled, cfg_rescale,
        shift,
        use_teacache, use_cfgzero, use_preview,
        mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate,
        attention, vae_type, variant,
        vlm_enhance, vlm_model, vlm_system_prompt,
    ]

    framepack_dict = dict(
        fn=run_framepack,
        _js="submit_framepack",
        inputs=state_inputs + framepack_inputs,
        outputs=framepack_outputs,
        show_progress='hidden',
    )
    generate.click(**framepack_dict)
