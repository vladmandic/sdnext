import gradio as gr
from modules import shared, ui_common, generation_parameters_copypaste
from modules.interrogate import openclip


def update_vlm_params(*args):
    vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p = args
    shared.opts.interrogate_vlm_max_length = int(vlm_max_tokens)
    shared.opts.interrogate_vlm_num_beams = int(vlm_num_beams)
    shared.opts.interrogate_vlm_temperature = float(vlm_temperature)
    shared.opts.interrogate_vlm_do_sample = bool(vlm_do_sample)
    shared.opts.interrogate_vlm_top_k = int(vlm_top_k)
    shared.opts.interrogate_vlm_top_p = float(vlm_top_p)
    shared.opts.save(shared.config_filename)


def update_clip_params(*args):
    clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams = args
    shared.opts.interrogate_clip_min_length = int(clip_min_length)
    shared.opts.interrogate_clip_max_length = int(clip_max_length)
    shared.opts.interrogate_clip_min_flavors = int(clip_min_flavors)
    shared.opts.interrogate_clip_max_flavors = int(clip_max_flavors)
    shared.opts.interrogate_clip_num_beams = int(clip_num_beams)
    shared.opts.interrogate_clip_flavor_count = int(clip_flavor_count)
    shared.opts.interrogate_clip_chunk_size = int(clip_chunk_size)
    shared.opts.save(shared.config_filename)
    openclip.update_interrogate_params()


def create_ui():
    with gr.Row(equal_height=False, variant='compact', elem_classes="caption", elem_id="caption_tab"):
        with gr.Column(variant='compact', elem_id='interrogate_input'):
            with gr.Row():
                image = gr.Image(type='pil', label="Image", height=512, visible=True, image_mode='RGB', elem_id='interrogate_image')
            with gr.Tabs(elem_id="mode_caption"):
                with gr.Tab("VLM Caption", elem_id="tab_vlm_caption"):
                    from modules.interrogate import vqa
                    with gr.Row():
                        vlm_system = gr.Textbox(label="System prompt", value=vqa.vlm_system, lines=1, elem_id='vlm_system')
                    with gr.Row():
                        vlm_question = gr.Dropdown(label="Predefined question", allow_custom_value=False, choices=vqa.vlm_prompts, value=vqa.vlm_prompts[2], elem_id='vlm_question')
                    with gr.Row():
                        vlm_prompt = gr.Textbox(label="Prompt", placeholder="optionally enter custom prompt", lines=2, elem_id='vlm_prompt')
                    with gr.Row(elem_id='interrogate_buttons_query'):
                        vlm_model = gr.Dropdown(list(vqa.vlm_models), value=list(vqa.vlm_models)[0], label='VLM Model', elem_id='vlm_model')
                    with gr.Accordion(label='Advanced options', open=False, visible=True):
                        with gr.Row():
                            vlm_max_tokens = gr.Slider(label='VLM max tokens', value=shared.opts.interrogate_vlm_max_length, minimum=16, maximum=4096, step=1, elem_id='vlm_max_tokens')
                            vlm_num_beams = gr.Slider(label='VLM num beams', value=shared.opts.interrogate_vlm_num_beams, minimum=1, maximum=16, step=1, elem_id='vlm_num_beams')
                            vlm_temperature = gr.Slider(label='VLM Temperature', value=shared.opts.interrogate_vlm_temperature, minimum=0.1, maximum=1.0, step=0.01, elem_id='vlm_temperature')
                        with gr.Row():
                            vlm_top_k = gr.Slider(label='Top-K', value=shared.opts.interrogate_vlm_top_k, minimum=0, maximum=99, step=1, elem_id='vlm_top_k')
                            vlm_top_p = gr.Slider(label='Top-P', value=shared.opts.interrogate_vlm_top_p, minimum=0.0, maximum=1.0, step=0.01, elem_id='vlm_top_p')
                        with gr.Row():
                            vlm_do_sample = gr.Checkbox(label='Use sample', value=shared.opts.interrogate_vlm_do_sample, elem_id='vlm_do_sample')
                        vlm_max_tokens.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                        vlm_num_beams.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                        vlm_temperature.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                        vlm_do_sample.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                        vlm_top_k.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                        vlm_top_p.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p], outputs=[])
                    with gr.Accordion(label='Batch caption', open=False, visible=True):
                        with gr.Row():
                            vlm_batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], interactive=True, height=100, elem_id='vlm_batch_files')
                        with gr.Row():
                            vlm_batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], interactive=True, height=100, elem_id='vlm_batch_folder')
                        with gr.Row():
                            vlm_batch_str = gr.Textbox(label="Folder", value="", interactive=True, elem_id='vlm_batch_str')
                        with gr.Row():
                            vlm_save_output = gr.Checkbox(label='Save caption files', value=True, elem_id="vlm_save_output")
                            vlm_save_append = gr.Checkbox(label='Append caption files', value=False, elem_id="vlm_save_append")
                            vlm_folder_recursive = gr.Checkbox(label='Recursive', value=False, elem_id="vlm_folder_recursive")
                        with gr.Row(elem_id='interrogate_buttons_batch'):
                            btn_vlm_caption_batch = gr.Button("Batch caption", variant='primary', elem_id="btn_vlm_caption_batch")
                    with gr.Row():
                        btn_vlm_caption = gr.Button("Caption", variant='primary', elem_id="btn_vlm_caption")
                with gr.Tab("CLiP Interrogate", elem_id='tab_clip_interrogate'):
                    with gr.Row():
                        clip_model = gr.Dropdown([], value=shared.opts.interrogate_clip_model, label='CLiP model', elem_id='clip_clip_model')
                        ui_common.create_refresh_button(clip_model, openclip.refresh_clip_models, lambda: {"choices": openclip.refresh_clip_models()}, 'clip_refresh_models')
                        blip_model = gr.Dropdown(list(openclip.caption_models), value=shared.opts.interrogate_blip_model, label='Caption model', elem_id='btN_clip_blip_model')
                        clip_mode = gr.Dropdown(openclip.caption_types, label='Mode', value='fast', elem_id='clip_clip_mode')
                    with gr.Accordion(label='Advanced options', open=False, visible=True):
                        with gr.Row():
                            clip_min_length = gr.Slider(label='Min length', value=shared.opts.interrogate_clip_min_length, minimum=8, maximum=75, step=1, elem_id='clip_caption_min_length')
                            clip_max_length = gr.Slider(label='Max length', value=shared.opts.interrogate_clip_max_length, minimum=16, maximum=1024, step=1, elem_id='clip_caption_max_length')
                            clip_chunk_size = gr.Slider(label='Chunk size', value=shared.opts.interrogate_clip_chunk_size, minimum=256, maximum=4096, step=8, elem_id='clip_chunk_size')
                        with gr.Row():
                            clip_min_flavors = gr.Slider(label='Min flavors', value=shared.opts.interrogate_clip_min_flavors, minimum=1, maximum=16, step=1, elem_id='clip_min_flavors')
                            clip_max_flavors = gr.Slider(label='Max flavors', value=shared.opts.interrogate_clip_max_flavors, minimum=1, maximum=64, step=1, elem_id='clip_max_flavors')
                            clip_flavor_count = gr.Slider(label='Intermediates', value=shared.opts.interrogate_clip_flavor_count, minimum=256, maximum=4096, step=8, elem_id='clip_flavor_intermediate_count')
                        with gr.Row():
                            clip_num_beams = gr.Slider(label='CLiP num beams', value=shared.opts.interrogate_clip_num_beams, minimum=1, maximum=16, step=1, elem_id='clip_num_beams')
                        clip_min_length.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_max_length.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_chunk_size.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_min_flavors.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_max_flavors.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_flavor_count.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_num_beams.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                    with gr.Accordion(label='Batch interogate', open=False, visible=True):
                        with gr.Row():
                            clip_batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], interactive=True, height=100, elem_id='clip_batch_files')
                        with gr.Row():
                            clip_batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], interactive=True, height=100, elem_id='clip_batch_folder')
                        with gr.Row():
                            clip_batch_str = gr.Textbox(label="Folder", value="", interactive=True, elem_id='clip_batch_str')
                        with gr.Row():
                            clip_save_output = gr.Checkbox(label='Save caption files', value=True, elem_id="clip_save_output")
                            clip_save_append = gr.Checkbox(label='Append caption files', value=False, elem_id="clip_save_append")
                            clip_folder_recursive = gr.Checkbox(label='Recursive', value=False, elem_id="clip_folder_recursive")
                        with gr.Row():
                            btn_clip_interrogate_batch = gr.Button("Batch interrogate", variant='primary', elem_id="btn_clip_interrogate_batch")
                    with gr.Row():
                        btn_clip_interrogate_img = gr.Button("Interrogate", variant='primary', elem_id="btn_clip_interrogate_img")
                        btn_clip_analyze_img = gr.Button("Analyze", variant='primary', elem_id="btn_clip_analyze_img")
        with gr.Column(variant='compact', elem_id='interrogate_output'):
            with gr.Row(elem_id='interrogate_output_prompt'):
                prompt = gr.Textbox(label="Answer", lines=12, placeholder="ai generated image description")
            with gr.Row(elem_id='interrogate_output_classes'):
                medium = gr.Label(elem_id="interrogate_label_medium", label="Medium", num_top_classes=5, visible=False)
                artist = gr.Label(elem_id="interrogate_label_artist", label="Artist", num_top_classes=5, visible=False)
                movement = gr.Label(elem_id="interrogate_label_movement", label="Movement", num_top_classes=5, visible=False)
                trending = gr.Label(elem_id="interrogate_label_trending", label="Trending", num_top_classes=5, visible=False)
                flavor = gr.Label(elem_id="interrogate_label_flavor", label="Flavor", num_top_classes=5, visible=False)
            with gr.Row(elem_id='copy_buttons_interrogate'):
                copy_interrogate_buttons = generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "control", "extras"])

    btn_clip_interrogate_img.click(openclip.interrogate_image, inputs=[image, clip_model, blip_model, clip_mode], outputs=[prompt])
    btn_clip_analyze_img.click(openclip.analyze_image, inputs=[image, clip_model, blip_model], outputs=[medium, artist, movement, trending, flavor])
    btn_clip_interrogate_batch.click(fn=openclip.interrogate_batch, inputs=[clip_batch_files, clip_batch_folder, clip_batch_str, clip_model, blip_model, clip_mode, clip_save_output, clip_save_append, clip_folder_recursive], outputs=[prompt])
    btn_vlm_caption.click(fn=vqa.interrogate, inputs=[vlm_question, vlm_system, vlm_prompt, image, vlm_model], outputs=[prompt])
    btn_vlm_caption_batch.click(fn=vqa.batch, inputs=[vlm_model, vlm_system, vlm_batch_files, vlm_batch_folder, vlm_batch_str, vlm_question, vlm_prompt, vlm_save_output, vlm_save_append, vlm_folder_recursive], outputs=[prompt])

    for tabname, button in copy_interrogate_buttons.items():
        generation_parameters_copypaste.register_paste_params_button(generation_parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=image,))
    generation_parameters_copypaste.add_paste_fields("caption", image, None)
