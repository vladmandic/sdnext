import gradio as gr
from modules import shared, ui_common, generation_parameters_copypaste
from modules.interrogate import openclip


default_task = "Short Caption"

def vlm_caption_wrapper(question, system_prompt, prompt, image, model_name, prefill, thinking_mode):
    """Wrapper for vqa.interrogate that handles annotated image display."""
    from modules.interrogate import vqa
    answer = vqa.interrogate(question, system_prompt, prompt, image, model_name, prefill, thinking_mode)
    annotated_image = vqa.get_last_annotated_image()
    if annotated_image is not None:
        return answer, gr.update(value=annotated_image, visible=True)
    return answer, gr.update(visible=False)


def update_vlm_prompts_for_model(model_name):
    """Update the task dropdown choices based on selected model."""
    from modules.interrogate import vqa
    prompts = vqa.get_prompts_for_model(model_name)
    return gr.update(choices=prompts, value=prompts[0] if prompts else default_task)


def update_vlm_prompt_placeholder(question):
    """Update the prompt field placeholder based on selected task."""
    from modules.interrogate import vqa
    placeholder = vqa.get_prompt_placeholder(question)
    return gr.update(placeholder=placeholder)


def update_vlm_params(*args):
    vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode = args
    shared.opts.interrogate_vlm_max_length = int(vlm_max_tokens)
    shared.opts.interrogate_vlm_num_beams = int(vlm_num_beams)
    shared.opts.interrogate_vlm_temperature = float(vlm_temperature)
    shared.opts.interrogate_vlm_do_sample = bool(vlm_do_sample)
    shared.opts.interrogate_vlm_top_k = int(vlm_top_k)
    shared.opts.interrogate_vlm_top_p = float(vlm_top_p)
    shared.opts.interrogate_vlm_keep_prefill = bool(vlm_keep_prefill)
    shared.opts.interrogate_vlm_keep_thinking = bool(vlm_keep_thinking)
    shared.opts.interrogate_vlm_thinking_mode = bool(vlm_thinking_mode)
    shared.opts.save()


def tagger_tag_wrapper(image, model_name, general_threshold, character_threshold, include_rating, exclude_tags, max_tags, sort_alpha, use_spaces, escape_brackets):
    """Wrapper for tagger.tag that maps UI inputs to function parameters."""
    from modules.interrogate import tagger
    return tagger.tag(
        image=image,
        model_name=model_name,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        include_rating=include_rating,
        exclude_tags=exclude_tags,
        max_tags=int(max_tags),
        sort_alpha=sort_alpha,
        use_spaces=use_spaces,
        escape_brackets=escape_brackets,
    )


def tagger_batch_wrapper(model_name, batch_files, batch_folder, batch_str, save_output, save_append, recursive, general_threshold, character_threshold, include_rating, exclude_tags, max_tags, sort_alpha, use_spaces, escape_brackets):
    """Wrapper for tagger.batch that maps UI inputs to function parameters."""
    from modules.interrogate import tagger
    return tagger.batch(
        model_name=model_name,
        batch_files=batch_files,
        batch_folder=batch_folder,
        batch_str=batch_str,
        save_output=save_output,
        save_append=save_append,
        recursive=recursive,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        include_rating=include_rating,
        exclude_tags=exclude_tags,
        max_tags=int(max_tags),
        sort_alpha=sort_alpha,
        use_spaces=use_spaces,
        escape_brackets=escape_brackets,
    )


def update_tagger_ui(model_name):
    """Update UI controls based on selected tagger model.

    When DeepBooru is selected, character_threshold is disabled since DeepBooru
    doesn't support separate character threshold.
    """
    from modules.interrogate import tagger
    is_db = tagger.is_deepbooru(model_name)
    return [
        gr.update(interactive=not is_db),  # character_threshold
        gr.update(),  # include_rating - now supported by both taggers
    ]


def update_tagger_params(model_name, general_threshold, character_threshold, include_rating, max_tags, sort_alpha, use_spaces, escape_brackets, exclude_tags, show_scores):
    """Save all tagger parameters to shared.opts when UI controls change."""
    shared.opts.waifudiffusion_model = model_name
    shared.opts.tagger_threshold = float(general_threshold)
    shared.opts.waifudiffusion_character_threshold = float(character_threshold)
    shared.opts.tagger_include_rating = bool(include_rating)
    shared.opts.tagger_max_tags = int(max_tags)
    shared.opts.tagger_sort_alpha = bool(sort_alpha)
    shared.opts.tagger_use_spaces = bool(use_spaces)
    shared.opts.tagger_escape_brackets = bool(escape_brackets)
    shared.opts.tagger_exclude_tags = str(exclude_tags)
    shared.opts.tagger_show_scores = bool(show_scores)
    shared.opts.save()


def update_clip_params(*args):
    clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams = args
    shared.opts.interrogate_clip_min_length = int(clip_min_length)
    shared.opts.interrogate_clip_max_length = int(clip_max_length)
    shared.opts.interrogate_clip_min_flavors = int(clip_min_flavors)
    shared.opts.interrogate_clip_max_flavors = int(clip_max_flavors)
    shared.opts.interrogate_clip_num_beams = int(clip_num_beams)
    shared.opts.interrogate_clip_flavor_count = int(clip_flavor_count)
    shared.opts.interrogate_clip_chunk_size = int(clip_chunk_size)
    shared.opts.save()
    openclip.update_interrogate_params()


def update_clip_model_params(clip_model, blip_model, clip_mode):
    """Save CLiP model settings to shared.opts when UI controls change."""
    shared.opts.interrogate_clip_model = str(clip_model)
    shared.opts.interrogate_blip_model = str(blip_model)
    shared.opts.interrogate_clip_mode = str(clip_mode)
    shared.opts.save()


def update_vlm_model_params(vlm_model, vlm_system):
    """Save VLM model settings to shared.opts when UI controls change."""
    shared.opts.interrogate_vlm_model = str(vlm_model)
    shared.opts.interrogate_vlm_system = str(vlm_system)
    shared.opts.save()


def update_default_caption_type(caption_type):
    """Save the default caption type to shared.opts."""
    shared.opts.interrogate_default_type = str(caption_type)
    shared.opts.save()


def create_ui():
    shared.log.debug('UI initialize: tab=caption')
    with gr.Row(equal_height=False, variant='compact', elem_classes="caption", elem_id="caption_tab"):
        with gr.Column(variant='compact', elem_id='interrogate_input'):
            with gr.Row():
                image = gr.Image(type='pil', label="Image", height=512, visible=True, image_mode='RGB', elem_id='interrogate_image')
            with gr.Tabs(elem_id="mode_caption"):
                with gr.Tab("VLM Caption", elem_id="tab_vlm_caption"):
                    from modules.interrogate import vqa
                    current_vlm_model = shared.opts.interrogate_vlm_model or vqa.vlm_default
                    initial_prompts = vqa.get_prompts_for_model(current_vlm_model)
                    with gr.Row():
                        vlm_system = gr.Textbox(label="System Prompt", value=vqa.vlm_system, lines=1, elem_id='vlm_system')
                    with gr.Row():
                        vlm_question = gr.Dropdown(label="Task", allow_custom_value=False, choices=initial_prompts, value=default_task, elem_id='vlm_question')
                    with gr.Row():
                        vlm_prompt = gr.Textbox(label="Prompt", placeholder=vqa.get_prompt_placeholder(initial_prompts[0]), lines=2, elem_id='vlm_prompt')
                    with gr.Row(elem_id='interrogate_buttons_query'):
                        vlm_model = gr.Dropdown(list(vqa.vlm_models), value=current_vlm_model, label='VLM Model', elem_id='vlm_model')
                    with gr.Row():
                        vlm_load_btn = gr.Button(value='Load', elem_id='vlm_load', variant='secondary')
                        vlm_unload_btn = gr.Button(value='Unload', elem_id='vlm_unload', variant='secondary')
                    with gr.Accordion(label='Caption: Advanced Options', open=False, visible=True):
                        with gr.Row():
                            vlm_max_tokens = gr.Slider(label='VLM Max Tokens', value=shared.opts.interrogate_vlm_max_length, minimum=16, maximum=4096, step=1, elem_id='vlm_max_tokens')
                            vlm_num_beams = gr.Slider(label='VLM Num Beams', value=shared.opts.interrogate_vlm_num_beams, minimum=1, maximum=16, step=1, elem_id='vlm_num_beams')
                            vlm_temperature = gr.Slider(label='VLM Temperature', value=shared.opts.interrogate_vlm_temperature, minimum=0.0, maximum=1.0, step=0.01, elem_id='vlm_temperature')
                        with gr.Row():
                            vlm_top_k = gr.Slider(label='Top-K', value=shared.opts.interrogate_vlm_top_k, minimum=0, maximum=99, step=1, elem_id='vlm_top_k')
                            vlm_top_p = gr.Slider(label='Top-P', value=shared.opts.interrogate_vlm_top_p, minimum=0.0, maximum=1.0, step=0.01, elem_id='vlm_top_p')
                        with gr.Row():
                            vlm_do_sample = gr.Checkbox(label='Use Samplers', value=shared.opts.interrogate_vlm_do_sample, elem_id='vlm_do_sample')
                            vlm_thinking_mode = gr.Checkbox(label='Thinking Mode', value=shared.opts.interrogate_vlm_thinking_mode, elem_id='vlm_thinking_mode')
                        with gr.Row():
                            vlm_keep_thinking = gr.Checkbox(label='Keep Thinking Trace', value=shared.opts.interrogate_vlm_keep_thinking, elem_id='vlm_keep_thinking')
                            vlm_keep_prefill = gr.Checkbox(label='Keep Prefill', value=shared.opts.interrogate_vlm_keep_prefill, elem_id='vlm_keep_prefill')
                        with gr.Row():
                            vlm_prefill = gr.Textbox(label='Prefill Text', value='', lines=1, elem_id='vlm_prefill', placeholder='Optional prefill text for model to continue from')
                        vlm_max_tokens.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_num_beams.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_temperature.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_do_sample.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_top_k.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_top_p.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_keep_prefill.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_keep_thinking.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                        vlm_thinking_mode.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample, vlm_top_k, vlm_top_p, vlm_keep_prefill, vlm_keep_thinking, vlm_thinking_mode], outputs=[])
                    with gr.Accordion(label='Caption: Batch', open=False, visible=True):
                        with gr.Row():
                            vlm_batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], interactive=True, height=100, elem_id='vlm_batch_files')
                        with gr.Row():
                            vlm_batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], interactive=True, height=100, elem_id='vlm_batch_folder')
                        with gr.Row():
                            vlm_batch_str = gr.Textbox(label="Folder", value="", interactive=True, elem_id='vlm_batch_str')
                        with gr.Row():
                            vlm_save_output = gr.Checkbox(label='Save Caption Files', value=True, elem_id="vlm_save_output")
                            vlm_save_append = gr.Checkbox(label='Append Caption Files', value=False, elem_id="vlm_save_append")
                            vlm_folder_recursive = gr.Checkbox(label='Recursive', value=False, elem_id="vlm_folder_recursive")
                        with gr.Row(elem_id='interrogate_buttons_batch'):
                            btn_vlm_caption_batch = gr.Button("Batch Caption", variant='primary', elem_id="btn_vlm_caption_batch")
                    with gr.Row():
                        btn_vlm_caption = gr.Button("Caption", variant='primary', elem_id="btn_vlm_caption")
                with gr.Tab("OpenCLiP", elem_id='tab_clip_interrogate'):
                    with gr.Row():
                        clip_model = gr.Dropdown([], value=shared.opts.interrogate_clip_model, label='CLiP Model', elem_id='clip_clip_model')
                        ui_common.create_refresh_button(clip_model, openclip.refresh_clip_models, lambda: {"choices": openclip.refresh_clip_models()}, 'clip_models_refresh')
                        blip_model = gr.Dropdown(list(openclip.caption_models), value=shared.opts.interrogate_blip_model, label='Caption Model', elem_id='btN_clip_blip_model')
                        clip_mode = gr.Dropdown(openclip.caption_types, label='Mode', value='fast', elem_id='clip_clip_mode')
                    with gr.Accordion(label='Caption: Advanced Options', open=False, visible=True):
                        with gr.Row():
                            clip_min_length = gr.Slider(label='clip: min length', value=shared.opts.interrogate_clip_min_length, minimum=8, maximum=75, step=1, elem_id='clip_caption_min_length')
                            clip_max_length = gr.Slider(label='clip: max length', value=shared.opts.interrogate_clip_max_length, minimum=16, maximum=1024, step=1, elem_id='clip_caption_max_length')
                            clip_chunk_size = gr.Slider(label='clip: chunk size', value=shared.opts.interrogate_clip_chunk_size, minimum=256, maximum=4096, step=8, elem_id='clip_chunk_size')
                        with gr.Row():
                            clip_min_flavors = gr.Slider(label='clip: min flavors', value=shared.opts.interrogate_clip_min_flavors, minimum=1, maximum=16, step=1, elem_id='clip_min_flavors')
                            clip_max_flavors = gr.Slider(label='clip: max flavors', value=shared.opts.interrogate_clip_max_flavors, minimum=1, maximum=64, step=1, elem_id='clip_max_flavors')
                            clip_flavor_count = gr.Slider(label='clip: intermediates', value=shared.opts.interrogate_clip_flavor_count, minimum=256, maximum=4096, step=8, elem_id='clip_flavor_intermediate_count')
                        with gr.Row():
                            clip_num_beams = gr.Slider(label='clip: num beams', value=shared.opts.interrogate_clip_num_beams, minimum=1, maximum=16, step=1, elem_id='clip_num_beams')
                        clip_min_length.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_max_length.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_chunk_size.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_min_flavors.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_max_flavors.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_flavor_count.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                        clip_num_beams.change(fn=update_clip_params, inputs=[clip_min_length, clip_max_length, clip_chunk_size, clip_min_flavors, clip_max_flavors, clip_flavor_count, clip_num_beams], outputs=[])
                    with gr.Accordion(label='Caption: Batch', open=False, visible=True):
                        with gr.Row():
                            clip_batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], interactive=True, height=100, elem_id='clip_batch_files')
                        with gr.Row():
                            clip_batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], interactive=True, height=100, elem_id='clip_batch_folder')
                        with gr.Row():
                            clip_batch_str = gr.Textbox(label="Folder", value="", interactive=True, elem_id='clip_batch_str')
                        with gr.Row():
                            clip_save_output = gr.Checkbox(label='Save Caption Files', value=True, elem_id="clip_save_output")
                            clip_save_append = gr.Checkbox(label='Append Caption Files', value=False, elem_id="clip_save_append")
                            clip_folder_recursive = gr.Checkbox(label='Recursive', value=False, elem_id="clip_folder_recursive")
                        with gr.Row():
                            btn_clip_interrogate_batch = gr.Button("Batch Interrogate", variant='primary', elem_id="btn_clip_interrogate_batch")
                    with gr.Row():
                        btn_clip_interrogate_img = gr.Button("Interrogate", variant='primary', elem_id="btn_clip_interrogate_img")
                        btn_clip_analyze_img = gr.Button("Analyze", variant='primary', elem_id="btn_clip_analyze_img")
                with gr.Tab("Tagger", elem_id='tab_tagger'):
                    from modules.interrogate import tagger
                    with gr.Row():
                        wd_model = gr.Dropdown(tagger.get_models(), value=shared.opts.waifudiffusion_model, label='Tagger Model', elem_id='wd_model')
                        ui_common.create_refresh_button(wd_model, tagger.refresh_models, lambda: {"choices": tagger.get_models()}, 'wd_models_refresh')
                    with gr.Row():
                        wd_load_btn = gr.Button(value='Load', elem_id='wd_load', variant='secondary')
                        wd_unload_btn = gr.Button(value='Unload', elem_id='wd_unload', variant='secondary')
                    with gr.Accordion(label='Caption: Advanced Options', open=True, visible=True):
                        with gr.Row():
                            wd_general_threshold = gr.Slider(label='General threshold', value=shared.opts.tagger_threshold, minimum=0.0, maximum=1.0, step=0.01, elem_id='wd_general_threshold')
                            wd_character_threshold = gr.Slider(label='Character threshold', value=shared.opts.waifudiffusion_character_threshold, minimum=0.0, maximum=1.0, step=0.01, elem_id='wd_character_threshold')
                        with gr.Row():
                            wd_max_tags = gr.Slider(label='Max tags', value=shared.opts.tagger_max_tags, minimum=1, maximum=512, step=1, elem_id='wd_max_tags')
                            wd_include_rating = gr.Checkbox(label='Include rating', value=shared.opts.tagger_include_rating, elem_id='wd_include_rating')
                        with gr.Row():
                            wd_sort_alpha = gr.Checkbox(label='Sort alphabetically', value=shared.opts.tagger_sort_alpha, elem_id='wd_sort_alpha')
                            wd_use_spaces = gr.Checkbox(label='Use spaces', value=shared.opts.tagger_use_spaces, elem_id='wd_use_spaces')
                            wd_escape = gr.Checkbox(label='Escape brackets', value=shared.opts.tagger_escape_brackets, elem_id='wd_escape')
                        with gr.Row():
                            wd_exclude_tags = gr.Textbox(label='Exclude tags', value=shared.opts.tagger_exclude_tags, placeholder='Comma-separated tags to exclude', elem_id='wd_exclude_tags')
                        with gr.Row():
                            wd_show_scores = gr.Checkbox(label='Show confidence scores', value=shared.opts.tagger_show_scores, elem_id='wd_show_scores')
                    gr.HTML('<style>#wd_character_threshold:has(input:disabled), #wd_include_rating:has(input:disabled) { opacity: 0.5; }</style>')
                    with gr.Accordion(label='Caption: Batch', open=False, visible=True):
                        with gr.Row():
                            wd_batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], interactive=True, height=100, elem_id='wd_batch_files')
                        with gr.Row():
                            wd_batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], interactive=True, height=100, elem_id='wd_batch_folder')
                        with gr.Row():
                            wd_batch_str = gr.Textbox(label="Folder", value="", interactive=True, elem_id='wd_batch_str')
                        with gr.Row():
                            wd_save_output = gr.Checkbox(label='Save Caption Files', value=True, elem_id="wd_save_output")
                            wd_save_append = gr.Checkbox(label='Append Caption Files', value=False, elem_id="wd_save_append")
                            wd_folder_recursive = gr.Checkbox(label='Recursive', value=False, elem_id="wd_folder_recursive")
                        with gr.Row():
                            btn_wd_tag_batch = gr.Button("Batch Tag", variant='primary', elem_id="btn_wd_tag_batch")
                    with gr.Row():
                        btn_wd_tag = gr.Button("Tag", variant='primary', elem_id="btn_wd_tag")
                with gr.Tab("Interrogate", elem_id='tab_interrogate'):
                    with gr.Row():
                        default_caption_type = gr.Radio(
                            choices=["VLM", "OpenCLiP", "Tagger"],
                            value=shared.opts.interrogate_default_type,
                            label="Default Caption Type",
                            elem_id="default_caption_type"
                        )
        with gr.Column(variant='compact', elem_id='interrogate_output'):
            with gr.Row(elem_id='interrogate_output_prompt'):
                prompt = gr.Textbox(label="Answer", lines=12, placeholder="ai generated image description")
            with gr.Row(elem_id='interrogate_output_image'):
                output_image = gr.Image(type='pil', label="Annotated Image", interactive=False, visible=False, elem_id='interrogate_output_image_display')
            with gr.Row(elem_id='interrogate_output_classes'):
                medium = gr.Label(elem_id="interrogate_label_medium", label="Medium", num_top_classes=5, visible=False)
                artist = gr.Label(elem_id="interrogate_label_artist", label="Artist", num_top_classes=5, visible=False)
                movement = gr.Label(elem_id="interrogate_label_movement", label="Movement", num_top_classes=5, visible=False)
                trending = gr.Label(elem_id="interrogate_label_trending", label="Trending", num_top_classes=5, visible=False)
                flavor = gr.Label(elem_id="interrogate_label_flavor", label="Flavor", num_top_classes=5, visible=False)
                clip_labels_text = gr.Textbox(elem_id="interrogate_clip_labels_text", label="CLIP Analysis", lines=15, interactive=False, visible=False, show_label=False)
            with gr.Row(elem_id='copy_buttons_interrogate'):
                copy_interrogate_buttons = generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "control", "extras"])

    btn_clip_interrogate_img.click(openclip.interrogate_image, inputs=[image, clip_model, blip_model, clip_mode], outputs=[prompt]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])
    btn_clip_analyze_img.click(openclip.analyze_image, inputs=[image, clip_model, blip_model], outputs=[medium, artist, movement, trending, flavor, clip_labels_text]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])
    btn_clip_interrogate_batch.click(fn=openclip.interrogate_batch, inputs=[clip_batch_files, clip_batch_folder, clip_batch_str, clip_model, blip_model, clip_mode, clip_save_output, clip_save_append, clip_folder_recursive], outputs=[prompt]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])
    btn_vlm_caption.click(fn=vlm_caption_wrapper, inputs=[vlm_question, vlm_system, vlm_prompt, image, vlm_model, vlm_prefill, vlm_thinking_mode], outputs=[prompt, output_image])
    btn_vlm_caption_batch.click(fn=vqa.batch, inputs=[vlm_model, vlm_system, vlm_batch_files, vlm_batch_folder, vlm_batch_str, vlm_question, vlm_prompt, vlm_save_output, vlm_save_append, vlm_folder_recursive, vlm_prefill, vlm_thinking_mode], outputs=[prompt]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])
    btn_wd_tag.click(fn=tagger_tag_wrapper, inputs=[image, wd_model, wd_general_threshold, wd_character_threshold, wd_include_rating, wd_exclude_tags, wd_max_tags, wd_sort_alpha, wd_use_spaces, wd_escape], outputs=[prompt]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])
    btn_wd_tag_batch.click(fn=tagger_batch_wrapper, inputs=[wd_model, wd_batch_files, wd_batch_folder, wd_batch_str, wd_save_output, wd_save_append, wd_folder_recursive, wd_general_threshold, wd_character_threshold, wd_include_rating, wd_exclude_tags, wd_max_tags, wd_sort_alpha, wd_use_spaces, wd_escape], outputs=[prompt]).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[output_image])

    # Dynamic UI updates based on selected model and task
    vlm_model.change(fn=update_vlm_prompts_for_model, inputs=[vlm_model], outputs=[vlm_question])
    vlm_question.change(fn=update_vlm_prompt_placeholder, inputs=[vlm_question], outputs=[vlm_prompt])

    # Load/Unload model buttons
    vlm_load_btn.click(fn=vqa.load_model, inputs=[vlm_model], outputs=[])
    vlm_unload_btn.click(fn=vqa.unload_model, inputs=[], outputs=[])
    def tagger_load_wrapper(model_name):
        from modules.interrogate import tagger
        return tagger.load_model(model_name)
    def tagger_unload_wrapper():
        from modules.interrogate import tagger
        return tagger.unload_model()
    wd_load_btn.click(fn=tagger_load_wrapper, inputs=[wd_model], outputs=[])
    wd_unload_btn.click(fn=tagger_unload_wrapper, inputs=[], outputs=[])

    # Dynamic UI update when tagger model changes (disable controls for DeepBooru)
    wd_model.change(fn=update_tagger_ui, inputs=[wd_model], outputs=[wd_character_threshold, wd_include_rating], show_progress=False)

    # Save tagger parameters to shared.opts when UI controls change
    tagger_inputs = [wd_model, wd_general_threshold, wd_character_threshold, wd_include_rating, wd_max_tags, wd_sort_alpha, wd_use_spaces, wd_escape, wd_exclude_tags, wd_show_scores]
    wd_model.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_general_threshold.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_character_threshold.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_include_rating.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_max_tags.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_sort_alpha.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_use_spaces.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_escape.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_exclude_tags.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)
    wd_show_scores.change(fn=update_tagger_params, inputs=tagger_inputs, outputs=[], show_progress=False)

    # Save CLiP model parameters to shared.opts when UI controls change
    clip_model_inputs = [clip_model, blip_model, clip_mode]
    clip_model.change(fn=update_clip_model_params, inputs=clip_model_inputs, outputs=[], show_progress=False)
    blip_model.change(fn=update_clip_model_params, inputs=clip_model_inputs, outputs=[], show_progress=False)
    clip_mode.change(fn=update_clip_model_params, inputs=clip_model_inputs, outputs=[], show_progress=False)

    # Save VLM model parameters to shared.opts when UI controls change
    vlm_model_inputs = [vlm_model, vlm_system]
    vlm_model.change(fn=update_vlm_model_params, inputs=vlm_model_inputs, outputs=[], show_progress=False)
    vlm_system.change(fn=update_vlm_model_params, inputs=vlm_model_inputs, outputs=[], show_progress=False)

    # Save default caption type to shared.opts when UI control changes
    default_caption_type.change(fn=update_default_caption_type, inputs=[default_caption_type], outputs=[], show_progress=False)

    for tabname, button in copy_interrogate_buttons.items():
        generation_parameters_copypaste.register_paste_params_button(generation_parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=image,))
    generation_parameters_copypaste.add_paste_fields("caption", image, None)
