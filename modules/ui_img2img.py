import gradio as gr
from modules import timer, shared, call_queue, generation_parameters_copypaste, processing_vae
from modules import ui_common, ui_sections, ui_guidance


def process_caption(mode, ii_input_files, ii_input_dir, ii_output_dir, *ii_singles):
    import os
    from PIL import Image
    from modules.caption.caption import caption
    mode = int(mode)
    if mode in {0, 1, 3, 4}:
        return [caption(ii_singles[mode]), None]
    if mode == 2:
        return [caption(ii_singles[mode]["image"]), None]
    if mode == 5:
        if len(ii_input_files) > 0:
            images = [f.name for f in ii_input_files]
        else:
            if not os.path.isdir(ii_input_dir):
                shared.log.error(f"Caption: Input directory not found: {ii_input_dir}")
                return [gr.update(), None]
            images = os.listdir(ii_input_dir)
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(caption(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8')) # pylint: disable=consider-using-with
    return [gr.update(), None]


def create_ui():
    shared.log.debug('UI initialize: tab=img2img')
    import modules.img2img # pylint: disable=redefined-outer-name
    modules.scripts_manager.scripts_current = modules.scripts_manager.scripts_img2img
    modules.scripts_manager.scripts_img2img.initialize_scripts(is_img2img=True, is_control=False)
    with gr.Blocks(analytics_enabled=False) as _img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, img2img_submit, img2img_reprocess, img2img_paste, img2img_extra_networks_button, img2img_token_counter, img2img_token_button, img2img_negative_token_counter, img2img_negative_token_button = ui_sections.create_toprow(is_img2img=True, id_part="img2img")
        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)
        img2img_prompt_img.change(fn=modules.images.image_data, inputs=[img2img_prompt_img], outputs=[img2img_prompt, img2img_prompt_img])

        with gr.Row(variant='compact', elem_id="img2img_extra_networks", elem_classes=["extra_networks_root"], visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, img2img_extra_networks_button, 'img2img', skip_indexing=shared.opts.extra_network_skip_indexing)
            timer.startup.record('ui-networks')

        with gr.Row(elem_id="img2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings", elem_classes=['settings-column']):
                copy_image_buttons = []
                copy_image_destinations = {}

                def copy_image(img):
                    return img['image'] if isinstance(img, dict) and 'image' in img else img

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_{tab_name}_row"):
                        for title, name in zip(['➠ Image', '➠ Inpaint', '➠ Sketch', '➠ Composite'], ['img2img', 'inpaint', 'sketch', 'composite']):
                            if name == tab_name:
                                gr.Button(title, elem_id=f'{tab_name}_copy_to_{name}', interactive=False)
                                copy_image_destinations[name] = elem
                                continue
                            button = gr.Button(title, elem_id=f'{tab_name}_copy_to_{name}')
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0) # pylint: disable=abstract-class-instantiated
                    state = gr.Textbox(value='', visible=False)
                    with gr.TabItem('Image', id='img2img_image', elem_id="img2img_image_tab") as tab_img2img:
                        img_init = gr.Image(label="", elem_id="img2img_image", show_label=False, interactive=True, type="pil", tool="editor", image_mode="RGBA", height=512)
                        caption_btn = ui_sections.create_caption_button(tab='img2img', what='input')
                        add_copy_image_controls('img2img', img_init)

                    with gr.TabItem('Inpaint', id='img2img_inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        img_inpaint = gr.Image(label="", elem_id="img2img_inpaint", show_label=False, interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('inpaint', img_inpaint)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_sketch_tab") as tab_sketch:
                        img_sketch = gr.Image(label="", elem_id="img2img_sketch", show_label=False, interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('sketch', img_sketch)

                    with gr.TabItem('Composite', id='img2img_composite', elem_id="img2img_composite_tab") as tab_inpaint_color:
                        img_composite = gr.Image(label="", show_label=False, elem_id="img2img_composite", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        img_composite_orig = gr.State(None) # pylint: disable=abstract-class-instantiated
                        img_composite_orig_update = False

                        def fn_img_composite_upload():
                            nonlocal img_composite_orig_update
                            img_composite_orig_update = True
                        def fn_img_composite_change(img, img_composite):
                            nonlocal img_composite_orig_update
                            res = img if img_composite_orig_update else img_composite
                            img_composite_orig_update = False
                            return res

                        img_composite.upload(fn=fn_img_composite_upload, inputs=[], outputs=[])
                        img_composite.change(fn=fn_img_composite_change, inputs=[img_composite, img_composite_orig], outputs=[img_composite_orig])
                        add_copy_image_controls('composite', img_composite)

                    with gr.TabItem('Upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        gr.HTML("<p style='padding-bottom: 1em;' class=\"text-gray-500\">Run image processing on upload images or files in a folder<br>If masks are provided will run inpaint</p>")
                        img2img_batch_files = gr.Files(label="Batch Process", interactive=True, elem_id="img2img_image_batch")
                        img2img_batch_input_dir = gr.Textbox(label="Batch input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Batch output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Batch mask directory", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]
                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                for button, name, elem in copy_image_buttons:
                    button.click(fn=copy_image, inputs=[elem], outputs=[copy_image_destinations[name]])
                    button.click(fn=lambda: None, _js=f"switch_to_{name.replace(' ', '_')}", inputs=[], outputs=[])

                with gr.Group(elem_classes="settings-accordion"):

                    with gr.Accordion(open=False, label="Sampler", elem_classes=["small-accordion"], elem_id="img2img_sampler_group"):
                        steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "img2img")
                        ui_sections.create_sampler_options('img2img')
                    resize_mode, resize_name, resize_context, width, height, scale_by, selected_scale_tab = ui_sections.create_resize_inputs('img2img', [img_init, img_sketch], latent=True, non_zero=False)
                    batch_count, batch_size = ui_sections.create_batch_inputs('img2img', accordion=True)
                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = ui_sections.create_seed_inputs('img2img')

                    with gr.Accordion(open=False, label="Denoise", elem_classes=["small-accordion"], elem_id="img2img_denoise_group"):
                        with gr.Row():
                            denoising_strength = gr.Slider(minimum=0.00, maximum=0.99, step=0.01, label='Denoising strength', value=0.30, elem_id="img2img_denoising_strength")
                            refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="img2img_refiner_start")

                    guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop, cfg_scale, image_cfg_scale, diffusers_guidance_rescale, pag_scale, pag_adaptive, cfg_end = ui_guidance.create_guidance_inputs('img2img')
                    vae_type, tiling, hidiffusion, clip_skip = ui_sections.create_advanced_inputs('img2img')
                    hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio = ui_sections.create_correction_inputs('img2img')
                    enable_hr, hr_sampler_index, hr_denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, hr_refiner_start, refiner_prompt, refiner_negative = ui_sections.create_hires_inputs('img2img')
                    detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength, detailer_resolution = shared.yolo.ui('img2img')

                    # with gr.Group(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                    with gr.Accordion(open=False, label="Mask", elem_classes=["small-accordion"], elem_id="img2img_mask_group") as inpaint_controls:
                        with gr.Row():
                            mask_blur = gr.Slider(label='Blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            inpaint_full_res_padding = gr.Slider(label='Padding', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")
                            mask_alpha = gr.Slider(label="Alpha", minimum=0.0, maximum=1.0, step=0.05, value=1.0, elem_id="img2img_mask_alpha")
                        with gr.Row():
                            inpainting_mask_invert = gr.Radio(label='Inpaint Mode', choices=['masked', 'invert'], value='masked', type="index", elem_id="img2img_mask_mode")
                            inpaint_full_res = gr.Radio(label="Inpaint area", choices=["full", "masked"], value="full", type="index", elem_id="img2img_inpaint_full_res")

                        def select_img2img_tab(tab):
                            return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3)

                        for i, elem in enumerate(img2img_tabs):
                            elem.select(fn=lambda tab=i: select_img2img_tab(tab), inputs=[], outputs=[inpaint_controls, mask_alpha]) # pylint: disable=cell-var-from-loop

                    override_settings = ui_common.create_override_inputs('img2img')

                with gr.Group(elem_id="img2img_script_container"):
                    img2img_script_inputs = modules.scripts_manager.scripts_img2img.setup_ui(parent='img2img', accordion=True)

            img2img_gallery, img2img_generation_info, img2img_html_info, _img2img_html_info_formatted, img2img_html_log = ui_common.create_output_panel("img2img", prompt=img2img_prompt)

            ui_common.reuse_seed(seed, reuse_seed, subseed=False)
            ui_common.reuse_seed(subseed, reuse_subseed, subseed=True)

            dummy_component1 = gr.Textbox(visible=False, value='dummy')
            dummy_component2 = gr.Number(visible=False, value=0)
            img2img_args = [
                dummy_component1, state, dummy_component2,
                img2img_prompt, img2img_negative_prompt, img2img_prompt_styles,
                img_init, img_sketch, img_inpaint, img_composite, img_composite_orig,
                init_img_inpaint, init_mask_inpaint,
                steps,
                sampler_index,
                mask_blur, mask_alpha,
                vae_type, tiling, hidiffusion,
                detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength, detailer_resolution,
                batch_count, batch_size,
                guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop,
                cfg_scale, image_cfg_scale, diffusers_guidance_rescale, pag_scale, pag_adaptive, cfg_end,
                refiner_start,
                clip_skip,
                denoising_strength,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                selected_scale_tab,
                height, width,
                scale_by,
                resize_mode, resize_name, resize_context,
                inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio,
                enable_hr, hr_sampler_index, hr_denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, hr_refiner_start, refiner_prompt, refiner_negative,
                override_settings,
            ]
            img2img_dict = dict(
                fn=call_queue.wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', ''], name='Image'),
                _js="submit_img2img",
                inputs= img2img_args + img2img_script_inputs,
                outputs=[
                    img2img_gallery,
                    img2img_generation_info,
                    img2img_html_info,
                    img2img_html_log,
                ],
                show_progress='hidden',
            )
            img2img_prompt.submit(**img2img_dict)
            img2img_negative_prompt.submit(**img2img_dict)
            img2img_submit.click(**img2img_dict)

            dummy_component = gr.Textbox(visible=False, value='dummy')

            img2img_reprocess[1].click(fn=processing_vae.reprocess, inputs=[img2img_gallery], outputs=[img2img_gallery]) # full-decode
            img2img_reprocess[2].click(**img2img_dict) # hires-refine
            img2img_reprocess[3].click(**img2img_dict) # face-restore

            caption_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_files,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    img_init, img_sketch, img_inpaint, img_composite,
                    init_img_inpaint,
                ],
                outputs=[img2img_prompt, dummy_component],
            )
            caption_btn.click(fn=lambda *args: process_caption(*args), **caption_args)

            img2img_token_button.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[img2img_prompt], outputs=[img2img_token_counter], show_progress = 'hidden')
            img2img_negative_token_button.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[img2img_negative_prompt], outputs=[img2img_negative_token_counter], show_progress = 'hidden')

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)
            img2img_paste_fields = [
                # prompt
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (img2img_prompt_styles, "Styles"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # resize
                (resize_mode, "Resize mode"),
                (resize_name, "Resize name"),
                (width, "Size-1"),
                (height, "Size-2"),
                (scale_by, "Resize scale"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # guidance
                (guidance_name, "Guidance"),
                (guidance_scale, "Guidance scale"),
                (guidance_rescale, "Guidance rescale"),
                (guidance_start, "Guidance start"),
                (guidance_stop, "Guidance stop"),
                # advanced
                (cfg_scale, "CFG scale"),
                (cfg_end, "CFG end"),
                (image_cfg_scale, "Image CFG scale"),
                (image_cfg_scale, "Hires CFG scale"),
                (clip_skip, "Clip skip"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (vae_type, "VAE type"),
                (tiling, "Tiling"),
                (hidiffusion, "HiDiffusion"),
                # detailer
                (detailer_enabled, "Detailer"),
                (detailer_prompt, "Detailer prompt"),
                (detailer_negative, "Detailer negative"),
                (detailer_steps, "Detailer steps"),
                (detailer_strength, "Detailer strength"),
                (detailer_resolution, "Detailer resolution"),
                # second pass
                (enable_hr, "Second pass"),
                (enable_hr, "Refine"),
                (denoising_strength, "Denoising strength"),
                (hr_denoising_strength, "Hires strength"),
                (hr_sampler_index, "Hires sampler"),
                (hr_resize_mode, "Hires mode"),
                (hr_resize_context, "Hires context"),
                (hr_upscaler, "Hires upscaler"),
                (hr_force, "Hires force"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_scale, "Hires upscale"),
                (hr_scale, "Hires scale"),
                (hr_resize_x, "Hires fixed-1"),
                (hr_resize_y, "Hires fixed-2"),
                # refiner
                (refiner_start, "Refiner start"),
                (refiner_steps, "Refiner steps"),
                (refiner_prompt, "refiner prompt"),
                (refiner_negative, "Refiner negative"),
                # pag
                (pag_scale, "CFG true"),
                (pag_adaptive, "CFG adaptive"),
                # inpaint
                (mask_blur, "Mask blur"),
                (mask_alpha, "Mask alpha"),
                (inpaint_full_res_padding, "Mask padding"),
                (inpainting_mask_invert, "Mask invert"),
                (inpaint_full_res, "Mask area"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *modules.scripts_manager.scripts_img2img.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("img2img", img_init, img2img_paste_fields, override_settings)
            generation_parameters_copypaste.add_paste_fields("sketch", img_sketch, img2img_paste_fields, override_settings)
            generation_parameters_copypaste.add_paste_fields("inpaint", img_inpaint, img2img_paste_fields, override_settings)
            img2img_bindings = generation_parameters_copypaste.ParamBinding(paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(img2img_bindings)
