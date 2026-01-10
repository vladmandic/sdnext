import os
import time
import gradio as gr
from modules.control import unit
from modules import errors, shared, progress, generation_parameters_copypaste, call_queue, scripts_manager, masking, images, processing_vae, timer # pylint: disable=ungrouped-imports
from modules import ui_common, ui_sections, ui_guidance
from modules import ui_control_helpers as helpers
import installer


gr_height = 512
max_units = shared.opts.control_max_units
units: list[unit.Unit] = [] # main state variable
controls: list[gr.component] = [] # list of gr controls
debug = shared.log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')


def return_stats(t: float = None):
    if t is None:
        elapsed_text = ''
    else:
        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"Time: {elapsed_m}m {elapsed_s:.2f}s |" if elapsed_m > 0 else f"Time: {elapsed_s:.2f}s |"
    summary = timer.process.summary(total=False).replace('=', ' ')
    gpu = ''
    cpu = ''
    if not shared.mem_mon.disabled:
        mem_mon_read = shared.mem_mon.read()
        ooms = mem_mon_read.pop("oom")
        retries = mem_mon_read.pop("retries")
        vram = {k: v//1048576 for k, v in mem_mon_read.items()}
        peak = max(vram['active_peak'], vram['reserved_peak'], vram['used'])
        used = round(100.0 * peak / vram['total']) if vram['total'] > 0 else 0
        if peak > 0:
            gpu += f"| GPU {peak} MB"
            gpu += f" {used}%" if used > 0 else ''
            gpu += f" | retries {retries} oom {ooms}" if retries > 0 or ooms > 0 else ''
    ram = shared.ram_stats()
    if ram['used'] > 0:
        cpu += f"| RAM {ram['used']} GB"
        cpu += f" {round(100.0 * ram['used'] / ram['total'])}%" if ram['total'] > 0 else ''
    return f"<div class='performance'><p>{elapsed_text} {summary} {gpu} {cpu}</p></div>"


def return_controls(res, t: float = None):
    # return preview, image, video, gallery, text
    debug(f'Control received: type={type(res)} {res}')
    if t is None:
        perf = ''
    else:
        perf = return_stats(t)
    if res is None: # no response
        return [None, None, None, None, '', perf]
    elif isinstance(res, str): # error response
        return [None, None, None, None, res, perf]
    elif isinstance(res, tuple): # standard response received as tuple via control_run->yield(output_images, process_image, result_txt)
        preview_image = res[1] # may be None
        output_image = res[0][0] if isinstance(res[0], list) else res[0] # may be image or list of images
        if isinstance(res[0], list):
            output_gallery = res[0] if res[0][0] is not None else []
        else:
            output_gallery = [res[0]] if res[0] is not None else [] # must return list, but can receive single image
        result_txt = res[2] if len(res) > 2 else '' # do we have a message
        output_video = res[3] if len(res) > 3 else None # do we have a video filename
        return [preview_image, output_image, output_video, output_gallery, result_txt, perf]
    else: # unexpected
        return [None, None, None, None, f'Control: Unexpected response: {type(res)}', perf]


def get_units(*values):
    update = []
    what = None
    for c, v in zip(controls, values):
        if isinstance(c, gr.Label): # unit type indicator
            what = c.value['label']
        c.value = v
        if c.elem_id is not None and c.elem_id.startswith('control_unit'):
            _prefix, i, name = c.elem_id.split('-')
            update.append({ 'type': what, 'index': int(i), 'name': name, 'value': v })
    for u in update:
        for i in range(len(units)):
            if units[i].type == u['type'] and units[i].index == u['index']:
                setattr(units[i], u['name'], u['value'])
                break


def generate_click(job_id: str, state: str, active_tab: str, *args):
    while helpers.busy:
        time.sleep(0.01)
    from modules.control.run import control_run
    debug(f'Control: tab="{active_tab}" job={job_id} args={args}')
    progress.add_task_to_queue(job_id)
    with call_queue.queue_lock:
        yield [None, None, None, None, 'Control: starting', '']
        shared.mem_mon.reset()
        jobid = shared.state.begin('Control')
        progress.start_task(job_id)
        try:
            t = time.perf_counter()
            for results in control_run(state, units, helpers.input_source, helpers.input_init, helpers.input_mask, active_tab, True, *args):
                progress.record_results(job_id, results)
                yield return_controls(results, t)
        except GeneratorExit:
            pass  # Generator was closed by client - cleanup will run in finally
        except Exception as e:
            shared.log.error(f"Control exception: {e}")
            errors.display(e, 'Control')
            yield [None, None, None, None, f'Control: Exception: {e}', '']
        finally:
            progress.finish_task(job_id)
            shared.state.end(jobid)


def create_ui(_blocks: gr.Blocks=None):
    helpers.initialize()

    with gr.Blocks(analytics_enabled = False) as control_ui:
        prompt, styles, negative, btn_generate, btn_reprocess, btn_paste, btn_extra, prompt_counter, btn_prompt_counter, negative_counter, btn_negative_counter  = ui_sections.create_toprow(is_img2img=False, id_part='control')
        prompt_img = gr.File(label="", elem_id="control_prompt_image", file_count="single", type="binary", visible=False)
        prompt_img.change(fn=images.image_data, inputs=[prompt_img], outputs=[prompt, prompt_img])

        with gr.Group(elem_id="control_interface"):

            with gr.Row(elem_id='control_status'):
                result_txt = gr.HTML(elem_classes=['control-result'], elem_id='control-result')

            with gr.Row(elem_id='control_settings', elem_classes=['settings-column']):

                state = gr.Textbox(value='', visible=False)

                with gr.Accordion(open=False, label="Input", elem_id="control_input", elem_classes=["small-accordion"]):
                    with gr.Row():
                        show_input = gr.Checkbox(label="Show input", value=True, elem_id="control_show_input")
                        show_preview = gr.Checkbox(label="Show preview", value=False, elem_id="control_show_preview")
                    with gr.Row():
                        input_type = gr.Radio(label="Control input type", choices=['Control only', 'Init image same as control', 'Separate init image'], value='Control only', type='index', elem_id='control_input_type')
                    with gr.Row():
                        denoising_strength = gr.Slider(minimum=0.00, maximum=0.99, step=0.01, label='Denoising strength', value=0.30, elem_id="control_input_denoising_strength")

                with gr.Accordion(open=False, label="Size", elem_id="control_size", elem_classes=["small-accordion"]):
                    with gr.Tabs():
                        with gr.Tab('Initial'):
                            resize_mode_before, resize_name_before, resize_context_before, width_before, height_before, scale_by_before, selected_scale_tab_before = ui_sections.create_resize_inputs('control_before', [], accordion=False, latent=True, prefix='before')
                        with gr.Tab('Post'):
                            resize_mode_after, resize_name_after, resize_context_after, width_after, height_after, scale_by_after, selected_scale_tab_after = ui_sections.create_resize_inputs('control_after', [], accordion=False, latent=False, prefix='after')
                        with gr.Tab('Mask'):
                            resize_mode_mask, resize_name_mask, resize_context_mask, width_mask, height_mask, scale_by_mask, selected_scale_tab_mask = ui_sections.create_resize_inputs('control_mask', [], accordion=False, latent=False, prefix='mask')

                with gr.Accordion(open=False, label="Sampler", elem_id="control_sampler", elem_classes=["small-accordion"]):
                    steps, sampler_index = ui_sections.create_sampler_and_steps_selection(None, "control")
                    ui_sections.create_sampler_options('control')

                batch_count, batch_size = ui_sections.create_batch_inputs('control', accordion=True)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = ui_sections.create_seed_inputs('control')
                ui_common.reuse_seed(seed, reuse_seed, subseed=False)
                ui_common.reuse_seed(subseed, reuse_subseed, subseed=True)

                mask_controls = masking.create_segment_ui()

                guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop, cfg_scale, image_cfg_scale, diffusers_guidance_rescale, pag_scale, pag_adaptive, cfg_end = ui_guidance.create_guidance_inputs('control')
                vae_type, tiling, hidiffusion, clip_skip = ui_sections.create_advanced_inputs('control')
                hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio = ui_sections.create_correction_inputs('control')

                with gr.Accordion(open=False, label="Video", elem_id="control_video", elem_classes=["small-accordion"]):
                    with gr.Row():
                        video_skip_frames = gr.Slider(minimum=0, maximum=100, step=1, label='Skip input frames', value=0, elem_id="control_video_skip_frames")
                    with gr.Row():
                        from modules.ui_sections import create_video_inputs
                        video_type, video_duration, video_loop, video_pad, video_interpolate = create_video_inputs(tab='control')

                enable_hr, hr_sampler_index, hr_denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative = ui_sections.create_hires_inputs('control')
                detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength, detailer_resolution = shared.yolo.ui('control')

            with gr.Row():
                override_script_name = gr.State(value='', visible=False, elem_id='control_override_script_name')
                override_script_args = gr.State(value='', visible=False, elem_id='control_override_script_args')
                override_settings = ui_common.create_override_inputs('control')

            with gr.Row(variant='compact', elem_id="control_extra_networks", elem_classes=["extra_networks_root"], visible=False) as extra_networks_ui:
                from modules import ui_extra_networks
                extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, btn_extra, 'control', skip_indexing=shared.opts.extra_network_skip_indexing)
                timer.startup.record('ui-networks')

            with gr.Row(elem_id='control-inputs'):
                with gr.Column(scale=9, elem_id='control-input-column', visible=True) as column_input:
                    gr.HTML('<span id="control-input-button">Input</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-input'):
                        input_mode = gr.Label(value='select', visible=False)
                        with gr.Tab('Image', id='in-image') as tab_image:
                            if (installer.version['kanvas'] == 'disabled') or (installer.version['kanvas'] == 'unavailable'):
                                shared.log.warning(f'Kanvas: status={installer.version["kanvas"]}')
                                input_image = gr.Image(label="Input", show_label=False, type="pil", interactive=True, tool="editor", height=gr_height, image_mode='RGB', elem_id='control_input_select', elem_classes=['control-image'])
                            else:
                                input_image = gr.HTML(value='<h1 style="text-align:center;color:var(--color-error);margin:1em;">Kanvas not initialized</h1>', elem_id='kanvas-container')
                            input_changed = gr.Button('Kanvas change', elem_id='kanvas-change-button', visible=False)
                            btn_interrogate = ui_sections.create_interrogate_button('control', what='input')
                        with gr.Tab('Video', id='in-video') as tab_video:
                            input_video = gr.Video(label="Input", show_label=False, interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Batch', id='in-batch') as tab_batch:
                            input_batch = gr.File(label="Input", show_label=False, file_count='multiple', file_types=['image'], interactive=True, height=gr_height)
                        with gr.Tab('Folder', id='in-folder') as tab_folder:
                            input_folder = gr.File(label="Input", show_label=False, file_count='directory', file_types=['image'], interactive=True, height=gr_height)
                with gr.Column(scale=9, elem_id='control-init-column', visible=False) as column_init:
                    gr.HTML('<span id="control-init-button">Init input</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-init'):
                        with gr.Tab('Image', id='init-image') as tab_image_init:
                            init_image = gr.Image(label="Input", show_label=False, type="pil", interactive=True, tool="editor", height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Video', id='init-video') as tab_video_init:
                            init_video = gr.Video(label="Input", show_label=False, interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Batch', id='init-batch') as tab_batch_init:
                            init_batch = gr.File(label="Input", show_label=False, file_count='multiple', file_types=['image'], interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Folder', id='init-folder') as tab_folder_init:
                            init_folder = gr.File(label="Input", show_label=False, file_count='directory', file_types=['image'], interactive=True, height=gr_height, elem_classes=['control-image'])
                with gr.Column(scale=9, elem_id='control-output-column', visible=True) as _column_output:
                    gr.HTML('<span id="control-output-button">Output</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-output') as output_tabs:
                        with gr.Tab('Gallery', id='out-gallery'):
                            output_gallery, _output_gen_info, _output_html_info, _output_html_info_formatted, output_html_log = ui_common.create_output_panel("control", preview=False, prompt=prompt, height=gr_height, result_info=result_txt)
                        with gr.Tab('Image', id='out-image'):
                            output_image = gr.Image(label="Output", show_label=False, type="pil", interactive=False, tool="editor", height=gr_height, elem_id='control_output_image', elem_classes=['control-image'])
                        with gr.Tab('Video', id='out-video'):
                            output_video = gr.Video(label="Output", show_label=False, height=gr_height, elem_id='control_output_video', elem_classes=['control-image'])
                with gr.Column(scale=9, elem_id='control-preview-column', visible=False) as column_preview:
                    gr.HTML('<span id="control-preview-button">Preview</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-preview'):
                        with gr.Tab('Preview', id='preview-image') as _tab_preview:
                            preview_process = gr.Image(label="Preview", show_label=False, type="pil", interactive=False, height=gr_height, visible=True, elem_id='control_preview', elem_classes=['control-image'])


            from modules.ui_control_elements import create_ui_elements
            create_ui_elements(units, result_txt, preview_process)

            with gr.Row(elem_id="control_script_container"):
                input_script_args = scripts_manager.scripts_current.setup_ui(parent='control', accordion=True)

            # handlers
            # for btn in input_buttons:
            #     btn.click(fn=helpers.copy_input, inputs=[input_mode, btn, input_image, input_resize, input_inpaint], outputs=[input_image, input_resize, input_inpaint], _js='controlInputMode')
            #     btn.click(fn=helpers.transfer_input, inputs=[btn], outputs=[input_image, input_resize, input_inpaint] + input_buttons)

            # hidden button to update gradio control values
            for u in units:
                controls.extend(u.controls)
            btn_update = gr.Button('Update', interactive=True, visible=False, elem_id='control_update')
            btn_update.click(fn=get_units, inputs=controls, outputs=[], show_progress='hidden', queue=False)

            show_input.change(fn=lambda x: gr.update(visible=x), inputs=[show_input], outputs=[column_input])
            show_preview.change(fn=lambda x: gr.update(visible=x), inputs=[show_preview], outputs=[column_preview])
            input_type.change(fn=lambda x: gr.update(visible=x == 2), inputs=[input_type], outputs=[column_init])
            btn_prompt_counter.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[prompt], outputs=[prompt_counter], show_progress = 'hidden')
            btn_negative_counter.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[negative], outputs=[negative_counter], show_progress = 'hidden')

            select_dict = dict(
                fn=helpers.select_input,
                _js="controlInputMode",
                inputs=[input_mode, input_image, init_image, input_type, input_video, input_batch, input_folder],
                outputs=[output_tabs, preview_process, result_txt, width_before, height_before],
                show_progress='hidden',
                queue=False,
            )

            input_changed.click(**select_dict)
            btn_interrogate.click(**select_dict) # need to fetch input first
            btn_interrogate.click(fn=helpers.interrogate, inputs=[], outputs=[prompt])

            prompt.submit(**select_dict)
            negative.submit(**select_dict)
            for ctrl in [input_image, input_video, input_batch, input_folder, init_image, init_video, init_batch, init_folder, tab_image, tab_video, tab_batch, tab_folder, tab_image_init, tab_video_init, tab_batch_init, tab_folder_init]:
                if hasattr(ctrl, 'change'):
                    ctrl.change(**select_dict)
                if hasattr(ctrl, 'clear'):
                    ctrl.clear(**select_dict)

            tabs_state = gr.Textbox(value='none', visible=False)
            input_fields = [
                input_type,
                prompt, negative, styles,
                steps, sampler_index,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop,
                cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, pag_scale, pag_adaptive, cfg_end, vae_type, tiling, hidiffusion,
                detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength, detailer_resolution,
                hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio,
                resize_mode_before, resize_name_before, resize_context_before, width_before, height_before, scale_by_before, selected_scale_tab_before,
                resize_mode_after, resize_name_after, resize_context_after, width_after, height_after, scale_by_after, selected_scale_tab_after,
                resize_mode_mask, resize_name_mask, resize_context_mask, width_mask, height_mask, scale_by_mask, selected_scale_tab_mask,
                denoising_strength, batch_count, batch_size,
                enable_hr, hr_sampler_index, hr_denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps,
                refiner_start, refiner_prompt, refiner_negative,
                video_skip_frames, video_type, video_duration, video_loop, video_pad, video_interpolate,
                override_script_name, override_script_args, override_settings,
            ]
            output_fields = [
                preview_process,
                output_image,
                output_video,
                output_gallery,
                result_txt,
                output_html_log,
            ]
            control_dict = dict(
                fn=generate_click,
                _js="submit_control",
                inputs=[tabs_state, state, tabs_state] + input_fields + input_script_args,
                outputs=output_fields,
                show_progress='full',
            )
            prompt.submit(**control_dict)
            negative.submit(**control_dict)
            btn_generate.click(**control_dict)

            btn_reprocess[1].click(fn=processing_vae.reprocess, inputs=[output_gallery], outputs=[output_gallery]) # full-decode
            btn_reprocess[2].click(**control_dict) # hires-refine
            btn_reprocess[3].click(**control_dict) # face-restore

            paste_fields = [
                # prompt
                (prompt, "Prompt"),
                (negative, "Negative prompt"),
                (styles, "Styles"),
                # input
                (denoising_strength, "Denoising strength"),
                # size basic
                (width_before, "Size-1"),
                (height_before, "Size-2"),
                (resize_mode_before, "Resize mode"),
                (scale_by_before, "Resize scale"),
                # size control
                (width_before, "Size before-1"),
                (height_before, "Size before-2"),
                (resize_mode_before, "Size mode before"),
                (scale_by_before, "Size scale before"),
                (resize_name_before, "Size name before"),
                (width_after, "Size after-1"),
                (height_after, "Size after-2"),
                (resize_mode_after, "Size mode after"),
                (scale_by_after, "Size scale after"),
                (resize_name_after, "Size name after"),
                (width_mask, "Size mask-1"),
                (height_mask, "Size mask-2"),
                (resize_mode_mask, "Size mode mask"),
                (scale_by_mask, "Size scale mask"),
                (resize_name_mask, "Size name mask"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # mask
                (mask_controls[1], "Mask only"),
                (mask_controls[2], "Mask invert"),
                (mask_controls[3], "Mask blur"),
                (mask_controls[4], "Mask erode"),
                (mask_controls[5], "Mask dilate"),
                (mask_controls[6], "Mask auto"),
                # guidance
                (guidance_name, "Guidance"),
                (guidance_scale, "Guidance scale"),
                (guidance_rescale, "Guidance rescale"),
                (guidance_start, "Guidance start"),
                (guidance_stop, "Guidance stop"),
                # advanced
                (cfg_scale, "CFG scale"),
                (cfg_end, "CFG end"),
                (clip_skip, "Clip skip"),
                (image_cfg_scale, "Image CFG scale"),
                (image_cfg_scale, "Hires CFG scale"),
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
                (refiner_prompt, "Refiner prompt"),
                (refiner_negative, "Refiner negative"),
                # pag
                (pag_scale, "CFG true"),
                (pag_adaptive, "CFG adaptive"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *scripts_manager.scripts_control.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("control", input_image, paste_fields, override_settings)
            bindings = generation_parameters_copypaste.ParamBinding(paste_button=btn_paste, tabname="control", source_text_component=prompt, source_image_component=output_gallery)
            generation_parameters_copypaste.register_paste_params_button(bindings)

            if (installer.version['kanvas'] == 'disabled') or (installer.version['kanvas'] == 'unavailable'):
                masking.bind_controls([input_image], preview_process, output_image)
            else:
                masking.bind_kanvas(input_image, preview_process)

            if os.environ.get('SD_CONTROL_DEBUG', None) is not None: # debug only
                from modules.control.test import test_processors, test_controlnets, test_adapters, test_xs, test_lite
                gr.HTML('<br><h1>Debug</h1><br>')
                with gr.Row():
                    run_test_processors_btn = gr.Button(value="Test:Processors", variant='primary', elem_classes=['control-button'])
                    run_test_controlnets_btn = gr.Button(value="Test:ControlNets", variant='primary', elem_classes=['control-button'])
                    run_test_xs_btn = gr.Button(value="Test:ControlNets-XS", variant='primary', elem_classes=['control-button'])
                    run_test_adapters_btn = gr.Button(value="Test:Adapters", variant='primary', elem_classes=['control-button'])
                    run_test_lite_btn = gr.Button(value="Test:Control-LLLite", variant='primary', elem_classes=['control-button'])

                    run_test_processors_btn.click(fn=test_processors, inputs=[input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                    run_test_controlnets_btn.click(fn=test_controlnets, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                    run_test_xs_btn.click(fn=test_xs, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                    run_test_adapters_btn.click(fn=test_adapters, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                    run_test_lite_btn.click(fn=test_lite, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])

    ui_extra_networks.setup_ui(extra_networks_ui, output_gallery)
    return [(control_ui, 'Control', 'control')]
