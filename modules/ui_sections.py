import gradio as gr
from modules import shared, modelloader, ui_symbols, ui_common, sd_samplers
from modules.ui_components import ToolButton
from modules.caption import caption


def create_toprow(is_img2img: bool = False, id_part: str = None, generate_visible: bool = True, negative_visible: bool = True, reprocess_visible: bool = True):
    def apply_styles(prompt, prompt_neg, styles):
        prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles, wildcards=not shared.opts.extra_networks_apply_unparsed)
        prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles, wildcards=not shared.opts.extra_networks_apply_unparsed)
        return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]

    def parse_style(styles):
        return styles.split('|') if styles is not None else None

    if id_part is None:
        id_part = "img2img" if is_img2img else "txt2img"
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=4):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row(elem_id=f"{id_part}_prompt_row"):
                        prompt = gr.Textbox(elem_id=f"{id_part}_prompt", label="Prompt", show_label=False, lines=3 if negative_visible else 5, placeholder="Prompt", elem_classes=["prompt", "main-prompts"])
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row(elem_id=f"{id_part}_negative_row"):
                        negative_prompt = gr.Textbox(elem_id=f"{id_part}_neg_prompt", label="Negative prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt", "main-prompts"], visible=negative_visible)
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                reprocess = []
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary', visible=generate_visible)
                if reprocess_visible:
                    reprocess.append(gr.Button('Reprocess', elem_id=f"{id_part}_reprocess", variant='primary', visible=True))
                    reprocess.append(gr.Button('Reprocess decode', elem_id=f"{id_part}_reprocess_decode", variant='primary', visible=False))
                    reprocess.append(gr.Button('Reprocess refine', elem_id=f"{id_part}_reprocess_refine", variant='primary', visible=False))
                    reprocess.append(gr.Button('Reprocess face', elem_id=f"{id_part}_reprocess_detail", variant='primary', visible=False))
            with gr.Row(elem_id=f"{id_part}_generate_line2"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt")
                interrupt.click(fn=lambda: shared.state.interrupt(), _js="requestInterrupt", inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=lambda: shared.state.pause(), _js='checkPaused', inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                button_paste = gr.Button(value='Restore', variant='secondary', elem_id=f"{id_part}_paste") # symbols.paste
                button_clear = gr.Button(value='Clear', variant='secondary', elem_id=f"{id_part}_clear_prompt_btn") # symbols.clear
                button_extra = gr.Button(value='Networks', variant='secondary', elem_id=f"{id_part}_extra_networks_btn") # symbols.networks
                button_clear.click(fn=lambda *x: ['', ''], inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt], show_progress='hidden')
            with gr.Row(elem_id=f"{id_part}_counters"):
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"], visible=False)
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"], visible=False)
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[style.name for style in shared.prompt_styles.styles.values()], value=[], multiselect=True)
                _styles_btn_refresh = ui_common.create_refresh_button(styles, shared.prompt_styles.reload, lambda: {"choices": [style.name for style in shared.prompt_styles.styles.values()]}, f"{id_part}_styles_refresh")
                styles_btn_select = ToolButton('Select', elem_id=f"{id_part}_styles_select", visible=False)
                styles_btn_apply = ToolButton(ui_symbols.style_apply, elem_id=f"{id_part}_styles_apply", visible=True)
                styles_btn_save = ToolButton(ui_symbols.style_save, elem_id=f"{id_part}_styles_save", visible=True)
                styles_btn_select.click(_js="applyStyles", fn=parse_style, inputs=[styles], outputs=[styles], show_progress='hidden')
                styles_btn_apply.click(fn=apply_styles, inputs=[prompt, negative_prompt, styles], outputs=[prompt, negative_prompt, styles], show_progress='hidden')
                styles_btn_save.click(fn=lambda: None, _js='() => quickSaveStyle()', inputs=[], outputs=[], show_progress='hidden')
    return prompt, styles, negative_prompt, submit, reprocess, button_paste, button_extra, token_counter, token_button, negative_token_counter, negative_token_button


def ar_change(ar, width, height):
    if ar == 'AR':
        return gr.update(), gr.update()
    try:
        (w, h) = [float(x) for x in ar.split(':')]
    except Exception as e:
        shared.log.warning(f"Invalid aspect ratio: {ar} {e}")
        return gr.update(), gr.update()
    if w > h:
        return gr.update(), gr.update(value=int(width * h / w))
    elif w < h:
        return gr.update(value=int(height * w / h)), gr.update()
    else:
        return gr.update(), gr.update()


def create_resolution_inputs(tab, default_width=1024, default_height=1024):
    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=default_width, elem_id=f"{tab}_width")
    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=default_height, elem_id=f"{tab}_height")
    ar_list = ['AR'] + [x.strip() for x in shared.opts.aspect_ratios.split(',') if x.strip() != '']
    ar_dropdown = gr.Dropdown(show_label=False, interactive=True, choices=ar_list, value=ar_list[0], elem_id=f"{tab}_ar", elem_classes=["ar-dropdown"])
    for c in [ar_dropdown, width, height]:
        c.change(fn=ar_change, inputs=[ar_dropdown, width, height], outputs=[width, height], show_progress='hidden')
    res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_res_btn_swap")
    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress='hidden')
    return width, height


def create_caption_button(tab: str, inputs: list = None, outputs: str = None, what: str = ''):
    button_caption = gr.Button(ui_symbols.caption, elem_id=f"{tab}_caption_{what}", elem_classes=['caption'])
    if inputs is not None and outputs is not None:
        button_caption.click(fn=caption.caption, inputs=inputs, outputs=[outputs])
    return button_caption


def create_batch_inputs(tab, accordion=True):
    with gr.Accordion(open=False, label="Batch", elem_id=f"{tab}_batch", elem_classes=["small-accordion"]) if accordion else gr.Group():
        with gr.Row(elem_id=f"{tab}_row_batch"):
            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{tab}_batch_count", scale=5)
            batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id=f"{tab}_batch_size", scale=5)
    return batch_count, batch_size


def create_seed_inputs(tab, reuse_visible=True, accordion=True, subseed_visible=True, seed_resize_visible=False):
    with gr.Accordion(open=False, label="Seed", elem_id=f"{tab}_seed_group", elem_classes=["small-accordion"]) if accordion else gr.Group():
        with gr.Row(elem_id=f"{tab}_seed_row", variant="compact"):
            seed = gr.Number(label='Initial seed', value=-1, elem_id=f"{tab}_seed", container=True)
            random_seed = ToolButton(ui_symbols.random, elem_id=f"{tab}_seed_random")
            reuse_seed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_seed_reuse", visible=reuse_visible)
        with gr.Row(elem_id=f"{tab}_subseed_row", variant="compact", visible=subseed_visible):
            subseed = gr.Number(label='Variation', value=-1, elem_id=f"{tab}_subseed", container=True)
            random_subseed = ToolButton(ui_symbols.random, elem_id=f"{tab}_subseed_random")
            reuse_subseed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_subseed_reuse", visible=reuse_visible)
            subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{tab}_subseed_strength", elem_classes=["subseed-strength"])
        with gr.Row(visible=seed_resize_visible):
            seed_resize_from_w = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from width", value=0, elem_id=f"{tab}_seed_resize_from_w")
            seed_resize_from_h = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from height", value=0, elem_id=f"{tab}_seed_resize_from_h")
        random_seed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[seed])
        random_subseed.click(fn=lambda: -1, show_progress='hidden', inputs=[], outputs=[subseed])
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


def create_video_inputs(tab:str, show_always:bool=False):
    def video_type_change(video_type):
        return [
            gr.update(visible=video_type != 'None' or show_always),
            gr.update(visible=video_type in ['GIF', 'PNG'] or show_always),
            gr.update(visible=video_type not in ['None', 'GIF', 'PNG'] or show_always),
            gr.update(visible=video_type not in ['None', 'GIF', 'PNG'] or show_always),
        ]
    with gr.Row():
        video_codecs = ['None', 'GIF', 'PNG', 'MP4/MP4V', 'MP4/AVC1', 'MP4/JVT3', 'MKV/H264', 'AVI/DIVX', 'AVI/RGBA', 'MJPEG/MJPG', 'MPG/MPG1', 'AVR/AVR1']
        video_type = gr.Dropdown(label='Video format', choices=video_codecs, value='MP4/MP4V', elem_id=f"{tab}_video_type")
    with gr.Row():
        video_duration = gr.Slider(label='Video duration', minimum=0.25, maximum=300, step=0.25, value=2, visible=show_always, elem_id=f"{tab}_video_duration")
        video_loop = gr.Checkbox(label='Loop video', value=True, visible=show_always, elem_id=f"{tab}_video_loop")
    with gr.Row():
        video_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=show_always, elem_id=f"{tab}_video_pad")
        video_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=show_always, elem_id=f"{tab}_video_interpolate")
    video_type.change(fn=video_type_change, inputs=[video_type], outputs=[video_duration, video_loop, video_pad, video_interpolate])
    return video_type, video_duration, video_loop, video_pad, video_interpolate


def create_advanced_inputs(tab):
    with gr.Accordion(open=False, label="Advanced", elem_id=f"{tab}_advanced", elem_classes=["small-accordion"]):
        with gr.Group():
            with gr.Row():
                clip_skip = gr.Slider(label='CLiP skip', value=1, minimum=0, maximum=12, step=0.1, elem_id=f"{tab}_clip_skip", interactive=shared.opts.clip_skip_enabled)
            with gr.Row(elem_id=f"{tab}_vae_options"):
                vae_type = gr.Dropdown(label='VAE type', choices=['Full', 'Tiny', 'Remote'], value='Full', elem_id=f"{tab}_vae_type")
            with gr.Row(elem_id=f"{tab}_advanced_options"):
                tiling = gr.Checkbox(label='Texture tiling', value=False, elem_id=f"{tab}_tiling")
                hidiffusion = gr.Checkbox(label='HiDiffusion', value=False, elem_id=f"{tab}_hidiffusion")
    return vae_type, tiling, hidiffusion, clip_skip


def create_correction_inputs(tab):
    with gr.Accordion(open=False, label="Corrections", elem_id=f"{tab}_corrections", elem_classes=["small-accordion"]):
        with gr.Group():
            with gr.Row(elem_id=f"{tab}_hdr_mode_row"):
                hdr_mode = gr.Dropdown(label="Correction mode", choices=["Relative values", "Absolute values"], type="index", value="Relative values", elem_id=f"{tab}_hdr_mode", show_label=False)
                gr.HTML('<br>')
            with gr.Row(elem_id=f"{tab}_correction_row"):
                hdr_brightness = gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, value=0,  label='Brightness', elem_id=f"{tab}_hdr_brightness")
                hdr_sharpen = gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, value=0,  label='Sharpen', elem_id=f"{tab}_hdr_sharpen")
                hdr_color = gr.Slider(minimum=0.0, maximum=4.0, step=0.1, value=0.0,  label='Color', elem_id=f"{tab}_hdr_color")
            with gr.Row(elem_id=f"{tab}_hdr_clamp_row"):
                hdr_clamp = gr.Checkbox(label='HDR clamp', value=False, elem_id=f"{tab}_hdr_clamp")
                hdr_boundary = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=4.0,  label='Range', elem_id=f"{tab}_hdr_boundary")
                hdr_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.95,  label='Threshold', elem_id=f"{tab}_hdr_threshold")
            with gr.Row(elem_id=f"{tab}_hdr_max_row"):
                hdr_maximize = gr.Checkbox(label='HDR maximize', value=False, elem_id=f"{tab}_hdr_maximize")
                hdr_max_center = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.6,  label='Center', elem_id=f"{tab}_hdr_max_center")
                hdr_max_boundary = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0,  label='Max Range', elem_id=f"{tab}_hdr_max_boundary")
            with gr.Row(elem_id=f"{tab}_hdr_color_row"):
                hdr_color_picker = gr.ColorPicker(label="Color", show_label=True, container=False, value=None, elem_id=f"{tab}_hdr_color_picker")
                hdr_tint_ratio = gr.Slider(label='Color grading', minimum=-1.0, maximum=1.0, step=0.05, value=0.0, elem_id=f"{tab}_hdr_tint_ratio")
        return hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio


def create_sampler_and_steps_selection(choices, tabname, default_steps:int=20):
    if choices is None:
        sd_samplers.set_samplers()
        choices = [x for x in sd_samplers.samplers if not x.name == 'Same as primary']
    with gr.Row(elem_id=f"{tabname}_sampler_row", elem_classes=['flex-break', 'flexbox']):
        steps = gr.Slider(minimum=1, maximum=100, step=1, label="Steps", elem_id=f"{tabname}_steps", value=default_steps)
        sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value='Default', type="index")
    return steps, sampler_index


def create_sampler_options(tabname):
    def set_sampler_options(sampler_options):
        shared.opts.data['schedulers_dynamic_shift'] = 'dynamic' in sampler_options
        shared.opts.data['schedulers_use_thresholding'] = 'thresholding' in sampler_options
        shared.opts.data['schedulers_use_loworder'] = 'low order' in sampler_options
        shared.opts.data['schedulers_rescale_betas'] = 'rescale' in sampler_options
        shared.log.debug(f'Sampler set options: {sampler_options}')
        shared.opts.save(silent=True)

    def set_sampler_timesteps(timesteps):
        shared.log.debug(f'Sampler set options: timesteps={timesteps}')
        shared.opts.schedulers_timesteps = timesteps
        shared.opts.save(silent=True)

    def set_sampler_spacing(spacing):
        shared.log.debug(f'Sampler set options: spacing={spacing}')
        shared.opts.schedulers_timestep_spacing = spacing
        shared.opts.save(silent=True)

    def set_sampler_sigma(sampler_sigma):
        shared.log.debug(f'Sampler set options: sigma={sampler_sigma}')
        shared.opts.schedulers_sigma = sampler_sigma
        shared.opts.save(silent=True)

    def set_sampler_order(sampler_order):
        shared.log.debug(f'Sampler set options: order={sampler_order}')
        shared.opts.schedulers_solver_order = sampler_order
        shared.opts.save(silent=True)

    def set_sampler_prediction(sampler_prediction):
        shared.log.debug(f'Sampler set options: prediction={sampler_prediction}')
        shared.opts.schedulers_prediction_type = sampler_prediction
        shared.opts.save(silent=True)

    def set_sampler_beta(sampler_beta):
        shared.log.debug(f'Sampler set options: beta={sampler_beta}')
        shared.opts.schedulers_beta_schedule = sampler_beta
        shared.opts.save(silent=True)

    def set_sampler_shift(sampler_shift, sampler_base_shift, sampler_max_shift):
        shared.log.debug(f'Sampler set options: shift={sampler_shift} base={sampler_base_shift} max={sampler_max_shift}')
        shared.opts.schedulers_shift = sampler_shift
        shared.opts.schedulers_base_shift = sampler_base_shift
        shared.opts.schedulers_max_shift = sampler_max_shift
        shared.opts.save(silent=True)

    def set_sigma_adjust(val, start, end):
        shared.log.debug(f'Sampler set options: sigma={val} min={start} max={end}')
        shared.opts.schedulers_sigma_adjust = val
        shared.opts.schedulers_sigma_adjust_min = start
        shared.opts.schedulers_sigma_adjust_max = end
        shared.opts.save(silent=True)

    # 'linear', 'scaled_linear', 'squaredcos_cap_v2'
    def set_sampler_preset(preset):
        if preset == 'AYS SD15':
            return '999,850,736,645,545,455,343,233,124,24'
        if preset == 'AYS SDXL':
            return '999,845,730,587,443,310,193,116,53,13'
        return ''

    with gr.Row(elem_classes=['flex-break']):
        sampler_sigma = gr.Dropdown(label='Sigma method', elem_id=f"{tabname}_sampler_sigma", choices=['default', 'karras', 'betas', 'exponential', 'lambdas', 'flowmatch'], value=shared.opts.schedulers_sigma, type='value')
        sampler_spacing = gr.Dropdown(label='Timestep spacing', elem_id=f"{tabname}_sampler_spacing", choices=['default', 'linspace', 'leading', 'trailing'], value=shared.opts.schedulers_timestep_spacing, type='value')
    with gr.Row(elem_classes=['flex-break']):
        sampler_beta = gr.Dropdown(label='Beta schedule', elem_id=f"{tabname}_sampler_beta", choices=['default', 'linear', 'scaled', 'cosine', 'sigmoid', 'laplace'], value=shared.opts.schedulers_beta_schedule, type='value')
        sampler_prediction = gr.Dropdown(label='Prediction method', elem_id=f"{tabname}_sampler_prediction", choices=['default', 'epsilon', 'sample', 'v_prediction', 'flow_prediction'], value=shared.opts.schedulers_prediction_type, type='value')
    with gr.Row(elem_classes=['flex-break']):
        sampler_presets = gr.Dropdown(label='Timesteps presets', elem_id=f"{tabname}_sampler_presets", choices=['None', 'AYS SD15', 'AYS SDXL'], value='None', type='value')
        sampler_timesteps = gr.Textbox(label='Timesteps override', elem_id=f"{tabname}_sampler_timesteps", value=shared.opts.schedulers_timesteps)
    with gr.Row(elem_classes=['flex-break']):
        sampler_sigma_adjust_val = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, label='Sigma adjust', value=shared.opts.schedulers_sigma_adjust, elem_id=f"{tabname}_sampler_sigma_adjust")
        sampler_sigma_adjust_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Adjust start', value=shared.opts.schedulers_sigma_adjust_min, elem_id=f"{tabname}_sampler_sigma_adjust_min")
        sampler_sigma_adjust_max = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Adjust end', value=shared.opts.schedulers_sigma_adjust_max, elem_id=f"{tabname}_sampler_sigma_adjust_max")
    with gr.Row(elem_classes=['flex-break']):
        sampler_order = gr.Slider(minimum=0, maximum=5, step=1, label="Sampler order", value=shared.opts.schedulers_solver_order, elem_id=f"{tabname}_sampler_order")
    with gr.Row(elem_classes=['flex-break']):
        sampler_shift = gr.Slider(minimum=0, maximum=10, step=0.1, label="Flow shift", value=shared.opts.schedulers_shift, elem_id=f"{tabname}_sampler_shift")
        sampler_base_shift = gr.Slider(minimum=0, maximum=10, step=0.01, label="Base shift", value=shared.opts.schedulers_base_shift, elem_id=f"{tabname}_sampler_base_shift")
        sampler_max_shift = gr.Slider(minimum=0, maximum=10, step=0.01, label="Max shift", value=shared.opts.schedulers_max_shift, elem_id=f"{tabname}_sampler_max_shift")
    with gr.Row(elem_classes=['flex-break']):
        options = ['low order', 'thresholding', 'dynamic', 'rescale']
        values = []
        values += ['low order'] if shared.opts.data.get('schedulers_use_loworder', True) else []
        values += ['thresholding'] if shared.opts.data.get('schedulers_use_thresholding', False) else []
        values += ['dynamic'] if shared.opts.data.get('schedulers_dynamic_shift', False) else []
        values += ['rescale'] if shared.opts.data.get('schedulers_rescale_betas', False) else []
        sampler_options = gr.CheckboxGroup(label='Options', elem_id=f"{tabname}_sampler_options", choices=options, value=values, type='value')

    sampler_sigma.change(fn=set_sampler_sigma, inputs=[sampler_sigma], outputs=[])
    sampler_spacing.change(fn=set_sampler_spacing, inputs=[sampler_spacing], outputs=[])
    sampler_presets.change(fn=set_sampler_preset, inputs=[sampler_presets], outputs=[sampler_timesteps])
    sampler_timesteps.change(fn=set_sampler_timesteps, inputs=[sampler_timesteps], outputs=[])
    sampler_beta.change(fn=set_sampler_beta, inputs=[sampler_beta], outputs=[])
    sampler_prediction.change(fn=set_sampler_prediction, inputs=[sampler_prediction], outputs=[])
    sampler_order.change(fn=set_sampler_order, inputs=[sampler_order], outputs=[])
    sampler_shift.change(fn=set_sampler_shift, inputs=[sampler_shift, sampler_base_shift, sampler_max_shift], outputs=[])
    sampler_options.change(fn=set_sampler_options, inputs=[sampler_options], outputs=[])
    sampler_sigma_adjust_val.change(fn=set_sigma_adjust, inputs=[sampler_sigma_adjust_val, sampler_sigma_adjust_min, sampler_sigma_adjust_max], outputs=[])
    sampler_sigma_adjust_min.change(fn=set_sigma_adjust, inputs=[sampler_sigma_adjust_val, sampler_sigma_adjust_min, sampler_sigma_adjust_max], outputs=[])
    sampler_sigma_adjust_max.change(fn=set_sigma_adjust, inputs=[sampler_sigma_adjust_val, sampler_sigma_adjust_min, sampler_sigma_adjust_max], outputs=[])


def create_hires_inputs(tab):
    with gr.Accordion(open=False, label="Refine", elem_id=f"{tab}_refine_accordion", elem_classes=["small-accordion"]):
        with gr.Row(elem_id=f"{tab}_hires_row1"):
            enable_hr = gr.Checkbox(label='Enable refine pass', value=False, elem_id=f"{tab}_enable_hr")
        hr_resize_mode, hr_upscaler, hr_resize_context, hr_resize_x, hr_resize_y, hr_scale, _selected_scale_tab = create_resize_inputs(tab, None, accordion=False, latent=True, non_zero=False)
        with gr.Row(elem_id=f"{tab}_hires_fix_row2"):
            hr_force = gr.Checkbox(label='Force HiRes', value=False, elem_id=f"{tab}_hr_force")
        with gr.Row(elem_id=f"{tab}_hires_fix_row2"):
            hr_sampler_index = gr.Dropdown(label='Refine sampler', elem_id=f"{tab}_sampling_alt", choices=[x.name for x in sd_samplers.samplers], value='Same as primary', type="index")
        with gr.Row(elem_id=f"{tab}_hires_row2"):
            hr_second_pass_steps = gr.Slider(minimum=0, maximum=99, step=1, label='HiRes steps', elem_id=f"{tab}_steps_alt", value=20)
            denoising_strength = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, label='Strength', value=0.3, elem_id=f"{tab}_denoising_strength")
        with gr.Row(elem_id=f"{tab}_refiner_row1"):
            refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Refiner start', value=0.0, elem_id=f"{tab}_refiner_start")
            refiner_steps = gr.Slider(minimum=0, maximum=99, step=1, label="Refiner steps", elem_id=f"{tab}_refiner_steps", value=20)
        refiner_prompt = gr.Textbox(value='', lines=2, label='Refine prompt', elem_id=f"{tab}_refiner_prompt", elem_classes=["prompt"], placeholder="refine prompt or leave empty to use main prompt")
        refiner_negative = gr.Textbox(value='', lines=2, label='Refine negative prompt', elem_id=f"{tab}_refiner_neg_prompt", elem_classes=["prompt"], placeholder="refine negative prompt or leave empty to use main prompt")
    return enable_hr, hr_sampler_index, denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative


def create_resize_inputs(tab, images, accordion=True, latent=False, non_zero=True, prefix=''):
    dummy_component = gr.Number(visible=False, value=0)
    if len(prefix) > 0 and not prefix.startswith(' '):
        prefix = f' {prefix}' if prefix != 'before' else ''
    with gr.Accordion(open=False, label="Resize", elem_classes=["small-accordion"], elem_id=f"{tab}_resize_group") if accordion else gr.Group():
        with gr.Row():
            available_upscalers = [x.name for x in shared.sd_upscalers]
            if not latent:
                available_upscalers = [x for x in available_upscalers if not x.lower().startswith('latent')]
            resize_mode = gr.Dropdown(label=f"Mode{prefix}" if non_zero else "Resize mode", elem_id=f"{tab}_resize_mode", choices=shared.resize_modes, type="index", value='Fixed')
            resize_name = gr.Dropdown(label=f"Method{prefix}" if non_zero else "Resize method", elem_id=f"{tab}_resize_name", choices=available_upscalers, value=available_upscalers[0], visible=True)
            resize_context_choices = ["Add with forward", "Remove with forward", "Add with backward", "Remove with backward"]
            resize_context = gr.Dropdown(label=f"Context{prefix}", elem_id=f"{tab}_resize_context", choices=resize_context_choices, value=resize_context_choices[0], visible=False)
            resize_refresh_btn = ui_common.create_refresh_button(resize_name, modelloader.load_upscalers, lambda: {"choices": modelloader.load_upscalers()}, f'{tab}_upscalers_refresh')

            def resize_mode_change(mode):
                if mode is None or mode == 0:
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                return gr.update(visible=mode != 5), gr.update(visible=mode == 5), gr.update(visible=mode != 5)
            resize_mode.change(fn=resize_mode_change, inputs=[resize_mode], outputs=[resize_name, resize_context, resize_refresh_btn])

        with gr.Row(visible=True) as _resize_group:
            with gr.Column(elem_id=f"{tab}_column_size"):
                selected_scale_tab = gr.State(value=0 if tab != 'img2img' else 1) # pylint: disable=abstract-class-instantiated
                with gr.Tabs(elem_id=f"{tab}_scale_tabs", selected=0 if non_zero else 1):
                    with gr.Tab(label="Fixed", id=0, elem_id=f"{tab}_scale_tab_fixed") as tab_scale_to:
                        with gr.Row(elem_id=f"{tab}_resize_row_fixed"):
                            with gr.Column(elem_id=f"{tab}_column_fixed1", scale=6):
                                suffix = '_resize' if tab != 'img2img' else ''
                                width = gr.Slider(minimum=64 if non_zero else 0, maximum=8192, step=8, label=f"Width{prefix}" if non_zero else "Resize width", value=1024 if non_zero else 0, elem_id=f"{tab}{suffix}_width")
                                height = gr.Slider(minimum=64 if non_zero else 0, maximum=8192, step=8, label=f"Height{prefix}" if non_zero else "Resize height", value=1024 if non_zero else 0, elem_id=f"{tab}{suffix}_height")
                            with gr.Column(elem_id=f"{tab}_column_fixed2", scale=1):
                                ar_list = ['AR'] + [x.strip() for x in shared.opts.aspect_ratios.split(',') if x.strip() != '']
                                ar_dropdown = gr.Dropdown(show_label=False, interactive=True, choices=ar_list, value=ar_list[0], elem_id=f"{tab}_resize_ar", elem_classes=["ar-dropdown"])
                                for c in [ar_dropdown, width, height]:
                                    c.change(fn=ar_change, inputs=[ar_dropdown, width, height], outputs=[width, height], show_progress='hidden')
                                res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_resize_size_swap")
                                res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress='hidden')
                                detect_image_size_btn = ToolButton(value=ui_symbols.detect, elem_id=f"{tab}_resize_detect_size")
                                el = tab.split('_')[0]
                                detect_image_size_btn.click(fn=lambda w, h, _: (w or gr.update(), h or gr.update()), _js=f'currentImageResolution{el}', inputs=[dummy_component, dummy_component, dummy_component], outputs=[width, height], show_progress='hidden')
                    with gr.Tab(label="Scale", id=1, elem_id=f"{tab}_scale_tab_scale") as tab_scale_by:
                        scale_by = gr.Slider(minimum=0.05, maximum=8.0, step=0.05, label=f"Scale{prefix}" if non_zero else "Resize scale", value=1.0, elem_id=f"{tab}_scale")
                    if images is not None:
                        for component in images:
                            component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress='hidden')
            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])
            # resize_mode.change(fn=lambda x: gr.update(visible=x != 0), inputs=[resize_mode], outputs=[_resize_group])
    return resize_mode, resize_name, resize_context, width, height, scale_by, selected_scale_tab
