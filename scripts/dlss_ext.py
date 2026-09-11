import os
import time
import textwrap
import gradio as gr
from modules.logger import log
from modules import shared, devices, processing, timer, errors, scripts_manager, scripts_postprocessing
from scripts.dlss import controller_cli as c


registered = False
debug = os.environ.get('SD_DLSS_DEBUG', None) is not None
FPS_CHOICES = ['23.976', '25', '29.97', '30', '50', '59.94', '60', '90', '119.88', '120', '144', '165', '180', '240', '360', '480']
NR_STYLES = ['None', 'Default', 'Natural', 'Cinematic']
NR_MODELS = ['None','Default', 'J', 'K', 'L', 'M']
NR_PRESETS = ['Default', 'Preset #1', 'Preset #2', 'Preset #3']


def create_ui(parent):
    with gr.Accordion('nVidia DLSS', open=False, elem_id=f'{parent}_dlss_accordion'):
        with gr.Row():
            btn_install = gr.Button(value="Install", elem_id='dlss_install')
            btn_verify = gr.Button(value="Verify", elem_id='dlss_verify')
            btn_status = gr.Button(value="Status", elem_id='dlss_status_btn')
            btn_reset = gr.Button(value="Reset", elem_id='dlss_reset')
            btn_shutdown = gr.Button(value="Shutdown", elem_id='dlss_shutdown')
        with gr.Row():
            install_note = gr.Markdown("", elem_id='dlss_install_note', visible=False)

        with gr.Accordion('DLSS NeuralRender', open=False, elem_id='dlss_nn'):
            with gr.Row():
                nr_enabled = gr.Checkbox(label='NR enable', value=False, elem_id='dlss_nr_enabled')
                nr_append = gr.Checkbox(label='NR append result', value=False, elem_id='dlss_nr_append')
            with gr.Row():
                nr_style = gr.Dropdown(label='NR style', choices=NR_STYLES, value='Default', elem_id='dlss_nr_style')
                nr_preset = gr.Dropdown(label='NR preset', choices=NR_PRESETS, value='Default', elem_id='dlss_nr_preset')
                nr_model_preset = gr.Dropdown(label='NR model', choices=NR_MODELS, value='Default', elem_id='dlss_nr_model_preset')
            with gr.Row():
                nr_intensity = gr.Slider(label='NR intensity', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_intensity')
                nr_local_tone = gr.Slider(label='NR tone strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_local_tone')
            with gr.Row():
                nr_local_structure = gr.Slider(label='NR local structure', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_local_structure')
                nr_skin_structure = gr.Slider(label='NR skin structure', minimum=-1.0, maximum=2.0, step=0.05, value=-1.0, elem_id='dlss_nr_skin_structure')
            with gr.Row():
                nr_upscaling_factor = gr.Dropdown(label='NR upscaling factor', choices=["1.0", "1.5", "1.724", "2.0", "3.0"], value="1.0", elem_id='dlss_nr_upscaling_factor')
                nr_automatic_mask = gr.Checkbox(label='NR automatic mask', value=False, elem_id='dlss_nr_automatic_mask')

        with gr.Accordion('DLSS SuperSample', open=False, elem_id='dlss_ss'):
            with gr.Row():
                ss_enabled = gr.Checkbox(label='SS enable', value=False, elem_id='dlss_ss_enabled')
                ss_append = gr.Checkbox(label='SS append result', value=False, elem_id='dlss_ss_append')
            with gr.Row():
                ss_vsr_quality = gr.Dropdown(label='SS VSR quality', choices=["1: Low", "2: Medium", "3: High", "4: Ultra"], value="4: Ultra", type='value', elem_id='dlss_ss_vsr_quality')
            with gr.Row():
                ss_size_mode = gr.Dropdown(label='SS size mode', choices=['Scale factor', 'Target size'], value='Scale factor', elem_id='dlss_ss_size_mode')
            with gr.Row():
                ss_scale_factor = gr.Slider(label='SS scale factor', minimum=1.0, maximum=8.0, step=0.05, value=2.0, elem_id='dlss_ss_scale_factor')
            with gr.Row():
                ss_width = gr.Number(label='SS width', minimum=64, maximum=16384, step=8, value=3840, elem_id='dlss_ss_width')
                ss_height = gr.Number(label='SS height', minimum=64, maximum=16384, step=8, value=2160, elem_id='dlss_ss_height')

        with gr.Accordion('DLSS FrameGen', open=False, elem_id='dlss_fg'):
            with gr.Row():
                fg_enabled = gr.Checkbox(label='FG enable', value=False, elem_id='dlss_fg_enabled')
            with gr.Row():
                fg_source_fps = gr.Dropdown(label='Source FPS', choices=FPS_CHOICES, value='24', elem_id='dlss_fg_source_fps')
                fg_target_fps = gr.Dropdown(label='Target FPS', choices=FPS_CHOICES, value='60', elem_id='dlss_fg_target_fps')
            with gr.Row():
                fg_engine = gr.Dropdown(label='Engine', choices=['Auto', 'Native DLSSG', 'Cascade'], value='Auto', elem_id='dlss_fg_engine')

        with gr.Accordion('DLSS Status', open=True, elem_id='dlss_status'):
            ss_status = gr.JSON({ 'Status': 'unknown' if len(shared.opts.dlss_pkg_path) < 4 else 'stored'})

        with gr.Row():
            pkg_path = gr.Textbox(label='DLSS Package path', value=shared.opts.dlss_pkg_path, placeholder='path to dlss 5 visual enhancer', elem_id='dlss_pkg_path')

        btn_install.click(install, inputs=[], outputs=[install_note])
        btn_verify.click(verify, inputs=[pkg_path], outputs=[ss_status])
        btn_status.click(status, inputs=[pkg_path], outputs=[ss_status])
        btn_reset.click(reset, inputs=[pkg_path], outputs=[ss_status])
        btn_shutdown.click(shutdown, inputs=[pkg_path], outputs=[ss_status])

    return [nr_enabled, nr_append, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset, ss_enabled, ss_append, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height, fg_enabled, fg_source_fps, fg_target_fps, fg_engine]


def install():
    note = textwrap.dedent("""\
        ### Install
        1. Download and unpack: [DLSS 5 Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v7.0)
        2. Enter the path to the unpacked package
        3. Press verify
        ### Notes
        - Package info is stored for future use on successful verification
        - DLSS controller process is started on first use
        - Use status to check the current state of the DLSS controller
        - Use reset to restore the DLSS controller to its default state
        - Use shutdown to stop the DLSS controller process
    """)
    return gr.update(value=note, visible=True)


def verify(pkg_path):
    log.info(f'DLSS verify: path="{pkg_path}"')
    if not os.path.exists(pkg_path) or not os.path.isdir(pkg_path):
        log.error(f'DLSS: path="{pkg_path}" not found')
        return { 'error': 'package path not found' }
    if not c.controller.get_python(pkg_path):
        return { 'error': 'python not found in package path' }
    response = c.controller.call(pkg_path, 'verify', { 'gpu_uuid': 'auto', 'options': { 'level': 'deep' } })
    if response.get('status') != 'ok':
        error = response.get('error') or {}
        log.error(f'DLSS: {error.get("message")}')
        return { 'error': error.get('message', 'unknown error') }
    report = (response.get('result') or {}).get('report', {})
    if debug:
        log.trace(f'DLSS raw: {report}')
    checks = { 'passed': 0, 'failed': 0 }
    for check in report.get('checks', []):
        if check.get('passed', False):
            checks['passed'] += 1
        else:
            checks['failed'] += 1
            log.error(f'DLSS : {check}')
    shared.opts.dlss_pkg_path = pkg_path
    shared.opts.save()
    log.debug(f'DLSS: gpu={report.get("gpu", "unknown")} checks={checks}')
    return report


def status(pkg_path):
    log.info(f'DLSS status: path="{pkg_path}"')
    response = c.controller.call(pkg_path, 'status', {})
    if response.get('status') != 'ok':
        error = response.get('error') or {}
        log.error(f'DLSS: {error.get("message")}')
        return { 'error': error.get('message', 'unknown error') }
    return response.get('result', {})


def reset(pkg_path):
    log.info(f'DLSS reset: path="{pkg_path}"')
    response = c.controller.call(pkg_path, 'reset', {})
    if response.get('status') != 'ok':
        error = response.get('error') or {}
        log.error(f'DLSS: {error.get("message")}')
        return { 'error': error.get('message', 'unknown error') }
    return response.get('result', {})


def shutdown(pkg_path):
    log.info(f'DLSS shutdown: path="{pkg_path}"')
    if not c.controller.is_alive():
        return { 'shutdown': True, 'note': 'controller was not running' }
    c.controller.stop()
    return { 'shutdown': True }


def supersample(pkg_path, images, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height):
    try:
        options = {
            'vsr_quality': int(ss_vsr_quality[0]),
            'size_mode': ss_size_mode,
            'scale_factor': float(ss_scale_factor),
            'width': int(ss_width),
            'height': int(ss_height),
            'aspect_lock': False,
        }
        frames = c.images_to_nchw(images)
        if debug:
            log.trace(f'DLSS: method=SuperSample input={frames.shape} options={options}')
        response = c.controller.call(
            pkg_path,
            'upscale',
            { 'images': frames, 'options': options },
            timeout=300.0,
        )
        if response.get('status') != 'ok':
            error = response.get('error') or {}
            log.error(f'DLSS: {error.get("message")}')
            return None
        return c.nchw_to_images(response.get('result'))
    except Exception as e:
        log.error(f'DLSS: {e}')
        errors.display(e, 'DLSS')
        return None


def neuralrender(pkg_path, images, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset):
    try:
        options = {
            'nr_style': nr_style,
            'nr_intensity': float(nr_intensity),
            'local_tone_strength': float(nr_local_tone),
            'local_structure_strength': float(nr_local_structure),
            'skin_structure_strength': float(nr_skin_structure),
            'upscaling_factor': float(nr_upscaling_factor),
            'warmup_frames': 0,
            'nr_preset': nr_preset,
            'automatic_mask': bool(nr_automatic_mask),
            'dlss_model_preset': nr_model_preset,
        }
        frames = c.images_to_nchw(images)
        if debug:
            log.trace(f'DLSS: method=NeuralRender input={frames.shape} options={options}')
        response = c.controller.call(
            pkg_path,
            'render',
            { 'images': frames, 'options': options },
            timeout=600.0,
        )
        if response.get('status') != 'ok':
            error = response.get('error') or {}
            log.error(f'DLSS: {error.get("message")}')
            return None
        return c.nchw_to_images(response.get('result'))
    except Exception as e:
        log.error(f'DLSS: {e}')
        errors.display(e, 'DLSS')
        return None


def framegen(pkg_path, images, fg_source_fps, fg_target_fps, fg_engine):
    try:
        if len(images) < 2:
            log.warning('DLSS: FrameGen requires at least two frames, skipping')
            return None
        options = { 'ai_gpu_uuid': 'auto', 'engine': fg_engine }
        frames = c.images_to_nchw(images)
        if debug:
            log.trace(f'DLSS: method=FrameGen input={frames.shape} options={options}')
        response = c.controller.call(
            pkg_path, 'framegen',
            { 'frames': frames, 'source_fps': fg_source_fps, 'target_fps': fg_target_fps, 'options': options },
            timeout=300.0,
        )
        if response.get('status') != 'ok':
            error = response.get('error') or {}
            log.error(f'DLSS: {error.get("message")}')
            return None
        return c.nchw_to_images(response.get('result'))
    except Exception as e:
        log.error(f'DLSS: {e}')
        errors.display(e, 'DLSS')
        return None


def dlss(p: processing.StableDiffusionProcessing | None, pp: processing.Processed | scripts_postprocessing.PostprocessedImage,
            nr_enabled, nr_append, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset,nr_automatic_mask, nr_model_preset,
            ss_enabled, ss_append, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height,
            fg_enabled, fg_source_fps, fg_target_fps, fg_engine,
            *args, **kwargs
        ):
    if not (ss_enabled or nr_enabled or fg_enabled):
        return None
    pkg_path = shared.opts.dlss_pkg_path
    if not pkg_path or not c.controller.get_python(pkg_path):
        log.error('DLSS: package path not configured')
        return None
    if debug:
        log.trace(f'DLSS: path="{pkg_path}" args={args} kwargs={kwargs}')

    update = 'none'
    if hasattr(pp, 'images') and pp.images is not None and len(pp.images) > 0:
        update = 'images'
        inputs = pp.images
    elif hasattr(pp, 'image') and pp.image is not None:
        update = 'image'
        inputs = [pp.image]
    else:
        return None

    # cast to appropriate types
    nr_style = str(getattr(p, 'nr_style', nr_style))
    nr_preset = str(getattr(p, 'nr_preset', nr_preset))
    nr_model_preset = str(getattr(p, 'nr_model_preset', nr_model_preset))
    nr_intensity = float(getattr(p, 'nr_intensity', nr_intensity))
    nr_local_tone = float(getattr(p, 'nr_local_tone', nr_local_tone))
    nr_local_structure = float(getattr(p, 'nr_local_structure', nr_local_structure))
    nr_skin_structure = float(getattr(p, 'nr_skin_structure', nr_skin_structure))
    nr_upscaling_factor = float(getattr(p, 'nr_upscaling_factor', nr_upscaling_factor))
    ss_width = int(getattr(p, 'ss_width', ss_width))
    ss_height = int(getattr(p, 'ss_height', ss_height))
    ss_scale_factor = float(getattr(p, 'ss_scale_factor', ss_scale_factor))
    fg_source_fps = str(getattr(p, 'fg_source_fps', fg_source_fps))
    fg_target_fps = str(getattr(p, 'fg_target_fps', fg_target_fps))
    if (p is not None) and ('video' in p.ops): # should not add video frames
        nr_append = False
        ss_append = False

    images = []
    originals = []
    current_images = inputs
    t = timer.Timer()

    jobid = shared.state.begin('DLSS')
    t_start = time.time()

    if ss_enabled:
        t0 = time.time()
        if p:
            p.extra_generation_params["DLSSSuperSample"] = True
        log.debug(f'DLSS: method=SuperSample quality="{ss_vsr_quality}" mode="{ss_size_mode}" scale={ss_scale_factor} width={ss_width} height={ss_height}')
        if ss_append:
            originals.extend(current_images)
        output = supersample(pkg_path, current_images, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height)
        if debug:
            log.trace(f'DLSS: method=SuperSample images={len(output) if output else 0} time={time.time() - t0:.3f}')
        if output:
            images.extend(output)
            current_images = output
        t.ts('supersample', t0)

    if nr_style == 'None' or nr_model_preset == 'None':
        nr_enabled = False
    if nr_enabled:
        t0 = time.time()
        if p:
            p.extra_generation_params["DLSSNeuralRender"] = True
        log.debug(f'DLSS: method=NeuralRender style={nr_style} intensity={nr_intensity} tone={nr_local_tone} structure={nr_local_structure} skin={nr_skin_structure} scale={nr_upscaling_factor} preset={nr_preset} mask={nr_automatic_mask} model={nr_model_preset}')
        if nr_append:
            originals.extend(current_images)
        output = neuralrender(pkg_path, current_images, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset)
        if debug:
            log.trace(f'DLSS: method=NeuralRender images={len(output) if output else 0} time={time.time() - t0:.3f}')
        if output:
            images.extend(output)
            current_images = output
        t.ts('neuralrender', t0)

    if fg_enabled:
        t0 = time.time()
        if p:
            p.extra_generation_params["DLSSFrameGen"] = True
        log.debug(f'DLSS: method=FrameGen source={fg_source_fps} target={fg_target_fps} engine={fg_engine}')
        output = framegen(pkg_path, current_images, fg_source_fps, fg_target_fps, fg_engine)
        if debug:
            log.trace(f'DLSS: method=FrameGen images={len(output) if output else 0} time={time.time() - t0:.3f}')
        if output:
            images.extend(output)
            current_images = output
        t.ts('framegen', t0)

    shared.state.end(jobid)
    timer.process.ts('dlss', t_start)

    log.debug(f'DLSS: frames={len(images)} {t.summary(min_time=0)}')
    if update == 'images':
        pp.images = images
    elif update == 'image' and len(images) > 0:
        pp.image = images[-1]
    pp.originals = originals
    return pp


class DLSSScript(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.video_capable = scripts_manager.AlwaysVisible
        self.register()

    def title(self):
        return 'nVidia DLSS'

    def show(self, _is_img2img):
        if devices.backend != 'cuda':
            return False
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return create_ui(self.parent)

    def register(self): # register xyz grid elements
        global registered # pylint: disable=global-statement
        if registered:
            return
        registered = True
        def apply_field(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                setattr(p, field, x)
                self.run(p)
            return fun

        import sys
        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k]
        if xyz_classes and len(xyz_classes) > 0:
            xyz_classes = xyz_classes[0]
            options = [
                xyz_classes.AxisOption("[DLSS] NR style", str, apply_field("nr_style"), choices=lambda: NR_STYLES),
                xyz_classes.AxisOption("[DLSS] NR preset", str, apply_field("nr_preset"), choices=lambda: NR_PRESETS),
                xyz_classes.AxisOption("[DLSS] NR model", str, apply_field("nr_model_preset"), choices=lambda: NR_MODELS),
                xyz_classes.AxisOption("[DLSS] NR intensity", float, apply_field("nr_intensity")),
                xyz_classes.AxisOption("[DLSS] NR local tone", float, apply_field("nr_local_tone")),
                xyz_classes.AxisOption("[DLSS] NR local structure", float, apply_field("nr_local_structure")),
                xyz_classes.AxisOption("[DLSS] NR skin structure", float, apply_field("nr_skin_structure")),
            ]
            for option in options:
                if option not in xyz_classes.axis_options:
                    xyz_classes.axis_options.append(option)

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, *args, **kwargs):
        if p.xyz:
            pp = dlss(p, pp, *args, **kwargs)

    def postprocess(self, p: processing.StableDiffusionProcessing, pp: processing.Processed, *args, **kwargs): # pylint: disable=arguments-differ,unused-argument
        if p.xyz: # do not postprocessing when running in xyz mode
            return pp
        _pp = dlss(p, pp, *args, **kwargs)
        # postprocess triggers after initial images have already been saved
        if _pp is not None and hasattr(_pp, 'images') and _pp.images is not None:
            pp = _pp
            orig_infos = pp.infotexts if hasattr(pp, 'infotexts') else []
            out_images, out_infos = processing.process_samples(p, pp.images)
            pp.images = out_images
            pp.infotexts = out_infos
            if hasattr(pp, 'originals') and pp.originals is not None and len(pp.originals) > 0:
                pp.infotexts = orig_infos + pp.infotexts
                pp.images = pp.originals + pp.images
        return pp


class DLSSPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "nVidia DLSS"
    order = 30000

    def ui(self):
        nr_enabled, nr_append, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset, ss_enabled, ss_append, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height, fg_enabled, fg_source_fps, fg_target_fps, fg_engine = create_ui('postprocess')
        return {
            "nr_enabled": nr_enabled,
            "nr_append": nr_append,
            "nr_style": nr_style,
            "nr_intensity": nr_intensity,
            "nr_local_tone": nr_local_tone,
            "nr_local_structure": nr_local_structure,
            "nr_skin_structure": nr_skin_structure,
            "nr_upscaling_factor": nr_upscaling_factor,
            "nr_preset": nr_preset,
            "nr_automatic_mask": nr_automatic_mask,
            "nr_model_preset": nr_model_preset,
            "ss_enabled": ss_enabled,
            "ss_append": ss_append,
            "ss_vsr_quality": ss_vsr_quality,
            "ss_size_mode": ss_size_mode,
            "ss_scale_factor": ss_scale_factor,
            "ss_width": ss_width,
            "ss_height": ss_height,
            "fg_enabled": fg_enabled,
            "fg_source_fps": fg_source_fps,
            "fg_target_fps": fg_target_fps,
            "fg_engine": fg_engine,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, *args, **kwargs):
        nr_enabled = kwargs.get("nr_enabled", False)
        ss_enabled = kwargs.get("ss_enabled", False)
        fg_enabled = kwargs.get("fg_enabled", False)
        if not (nr_enabled or ss_enabled or fg_enabled):
            return
        if pp.image is None:
            return
        result = dlss(None, pp, *args, **kwargs)
        if result is None or not hasattr(result, "images") or len(result.images) == 0:
            return
        pp.image = result.images[0]
        pp.info["DLSS"] = f'NR: {nr_enabled} SS: {ss_enabled} FG: {fg_enabled}'
