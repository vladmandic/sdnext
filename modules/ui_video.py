# TODO hunyuanvideo: seed, scheduler, scheduler_shift, guidance_scale=1.0, true_cfg_scale=6.0, num_inference_steps=30, prompt_template, vae, offloading
# TODO modernui video tab

from dataclasses import dataclass
import gradio as gr
from modules import shared, images, ui_common, ui_sections, sd_models, call_queue, generation_parameters_copypaste
from modules.video_models import hunyuan


@dataclass
class Model():
    name: str
    repo: str
    dit: str


MODELS = {
    'None': [],
    'Hunyuan Video': [
        Model('None', None, None),
        Model('Hunyuan Video T2V', 'hunyuanvideo-community/HunyuanVideo', None),
        Model('Hunyuan Video I2V', 'hunyuanvideo-community/HunyuanVideo', 'hunyuanvideo-community/HunyuanVideo-I2V'), # https://github.com/huggingface/diffusers/pull/10983
        Model('SkyReels Hunyuan T2V', 'hunyuanvideo-community/HunyuanVideo', 'Skywork/SkyReels-V1-Hunyuan-T2V'), # https://github.com/huggingface/diffusers/pull/10837
        Model('SkyReels Hunyuan I2V', 'hunyuanvideo-community/HunyuanVideo', 'Skywork/SkyReels-V1-Hunyuan-I2V'),
        Model('Fast Hunyuan T2V', 'hunyuanvideo-community/HunyuanVideo', 'hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt'), # https://github.com/hao-ai-lab/FastVideo/blob/8a77cf22c9b9e7f931f42bc4b35d21fd91d24e45/fastvideo/models/hunyuan/inference.py#L213
    ]
}


def engine_change(engine):
    models = [model.name for model in MODELS.get(engine, [])]
    return gr.update(choices=models, value=models[0] if len(models) > 0 else None)


def model_change(engine, model):
    models = [model.name for model in MODELS.get(engine, [])]
    selected = [m for m in MODELS[engine] if m.name == model][0] if len(models) > 0 else None
    if selected:
        if 'None' in selected.name:
            sd_models.unload_model_weights()
            msg = 'Video model unloaded'
        elif 'Hunyuan' in selected.name:
            msg = hunyuan.load(selected)
        elif model != 'None':
            msg = f'Video model not found: engine={engine} model={model}'
            shared.log.error(msg)
    else:
        sd_models.unload_model_weights()
        msg = 'Video model unloaded'
    return [msg, gr.update(visible='I2V' in selected.name) if selected else gr.update(visible=False)]


def run_video(*args):
    engine, model = args[2], args[3]
    models = [model.name for model in MODELS.get(engine, [])]
    selected = [m for m in MODELS[engine] if m.name == model][0] if len(models) > 0 else None
    if selected and 'Hunyuan' in selected.name:
        return hunyuan.generate(*args)
    shared.log.error(f'Video model not found: args={args}')
    return [], '', '', f'Video model not found: engine={engine} model={model}'


def create_ui():
    shared.log.debug('UI initialize: txt2img')
    with gr.Blocks(analytics_enabled=False) as _video_interface:
        prompt, styles, _negative, generate, _reprocess, paste, _networks, _token_counter, _token_button, _token_counter_negative, _token_button_negative = ui_sections.create_toprow(is_img2img=False, id_part="video", negative_visible=False, reprocess_visible=False)
        prompt_image = gr.File(label="", elem_id="video_prompt_image", file_count="single", type="binary", visible=False)
        prompt_image.change(fn=images.image_data, inputs=[prompt_image], outputs=[prompt, prompt_image])

        with gr.Row(elem_id="video_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="video_settings", scale=1):

                with gr.Row():
                    engine = gr.Dropdown(label='Engine', choices=list(MODELS), value='None')
                    model = gr.Dropdown(label='Model', choices=[''], value=None)
                with gr.Row():
                    width, height = ui_sections.create_resolution_inputs('video', default_width=720, default_height=480)
                with gr.Row():
                    frames = gr.Slider(label='Frames', minimum=1, maximum=1024, step=1, value=15)
                with gr.Row():
                    with gr.Group(visible=False) as image_group:
                        gr.HTML("<br>&nbsp Init image")
                        image = gr.Image(elem_id="video_image", show_label=False, source="upload", interactive=True, type="pil", tool="select", image_mode="RGB", height=512)
                with gr.Row():
                    save_frames = gr.Checkbox(label='Save image frames', value=False)
                with gr.Row():
                    cc, duration, loop, pad, interpolate = ui_sections.create_video_inputs(tab='video')
                override_settings = ui_common.create_override_inputs('video')

            # output panel with gallery
            gallery, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("video", prompt=prompt, preview=False, transfer=False, scale=2)

        # handle engine and model change
        engine.change(fn=engine_change, inputs=[engine], outputs=[model])
        model.change(fn=model_change, inputs=[engine, model], outputs=[html_log, image_group])
        # handle restore fields
        paste_fields = [
            (prompt, "Prompt"),
            # main
            (width, "Size-1"),
            (height, "Size-2"),
            (frames, "Frames"),
        ]
        generation_parameters_copypaste.add_paste_fields("txt2img", None, paste_fields, override_settings)
        bindings = generation_parameters_copypaste.ParamBinding(paste_button=paste, tabname="video", source_text_component=prompt, source_image_component=None)
        generation_parameters_copypaste.register_paste_params_button(bindings)
        # hidden fields
        task_id = gr.Textbox(visible=False, value='')
        ui_state = gr.Textbox(visible=False, value='')
        # generate args
        video_args = [
            task_id, ui_state,
            engine, model,
            prompt, styles,
            width, height,
            frames,
            image,
            save_frames,
            cc, duration, loop, pad, interpolate,
        ]
        # generate function
        video_dict = dict(
            fn=call_queue.wrap_gradio_gpu_call(run_video, extra_outputs=[None, '', ''], name='Video'),
            _js="submit_video",
            inputs=video_args,
            outputs=[gallery, gen_info, html_info, html_log],
            show_progress=False,
        )
        prompt.submit(**video_dict)
        generate.click(**video_dict)
