import os
import time
import textwrap
import gradio as gr
from modules.logger import log
from modules import scripts_manager, shared, devices, processing, timer, errors
from scripts.dlss import controller_cli as c


debug = os.environ.get('SD_DLSS_DEBUG', None) is not None
FPS_CHOICES = ['23.976', '24', '25', '29.97', '30', '50', '59.94', '60', '90', '119.88', '120', '144', '165', '180', '240', '360', '480']


class DLSSScript(scripts_manager.Script):
    def title(self):
        return 'nVidia DLSS'

    def show(self, _is_img2img):
        if devices.backend != 'cuda':
            return False
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        with gr.Accordion('nVidia DLSS', open=False, elem_id='dlss'):
            with gr.Row():
                pkg_path = gr.Textbox(label='DLSS package path', value=shared.opts.dlss_pkg_path, placeholder='path to dlss 5 visual enhancer', elem_id='dlss_pkg_path')
            with gr.Row():
                """
                btn_verify = ui_components.ToolButton(value=ui_symbols.tools, elem_id='dlss_verify')
                btn_status = ui_components.ToolButton(value=ui_symbols.info, elem_id='dlss_status_btn')
                btn_reset = ui_components.ToolButton(value=ui_symbols.reset, elem_id='dlss_reset')
                btn_shutdown = ui_components.ToolButton(value=ui_symbols.close, elem_id='dlss_shutdown')
                """
                btn_install = gr.Button(value="Install", elem_id='dlss_install')
                btn_verify = gr.Button(value="Verify", elem_id='dlss_verify')
                btn_status = gr.Button(value="Status", elem_id='dlss_status_btn')
                btn_reset = gr.Button(value="Reset", elem_id='dlss_reset')
                btn_shutdown = gr.Button(value="Shutdown", elem_id='dlss_shutdown')
            with gr.Row():
                install_note = gr.Markdown("", elem_id='dlss_install_note', visible=False)
            with gr.Accordion('DLSS Status', open=True, elem_id='dlss_status'):
                ss_status = gr.JSON({ 'Status': 'unknown' if len(shared.opts.dlss_pkg_path) < 4 else 'stored'})
            btn_install.click(self.install, inputs=[], outputs=[install_note])
            btn_verify.click(self.verify, inputs=[pkg_path], outputs=[ss_status])
            btn_status.click(self.status, inputs=[pkg_path], outputs=[ss_status])
            btn_reset.click(self.reset, inputs=[pkg_path], outputs=[ss_status])
            btn_shutdown.click(self.shutdown, inputs=[pkg_path], outputs=[ss_status])

            with gr.Accordion('DLSS NeuralRender', open=True, elem_id='dlss_nn'):
                with gr.Row():
                    nr_enabled = gr.Checkbox(label='NR enable', value=False, elem_id='dlss_nr_enabled')
                    nr_append = gr.Checkbox(label='Append result', value=True, elem_id='dlss_nr_append')
                with gr.Row():
                    nr_style = gr.Dropdown(label='NR style', choices=['Default', 'Natural', 'Cinematic'], value='Default', elem_id='dlss_nr_style')
                    nr_intensity = gr.Slider(label='NR intensity', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_intensity')
                    nr_upscaling_factor = gr.Dropdown(label='NR upscaling factor', choices=["1.0", "1.5", "1.724", "2.0", "3.0"], value="1.0", elem_id='dlss_nr_upscaling_factor')
                with gr.Row():
                    nr_preset = gr.Dropdown(label='NR preset', choices=['Default', 'Preset #1', 'Preset #2', 'Preset #3'], value='Default', elem_id='dlss_nr_preset')
                    nr_model_preset = gr.Dropdown(label='NR model', choices=['Default', 'J', 'K', 'L', 'M'], value='Default', elem_id='dlss_nr_model_preset')
                with gr.Row():
                    nr_local_tone = gr.Slider(label='NR tone strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_local_tone')
                    nr_local_structure = gr.Slider(label='NR local structure', minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id='dlss_nr_local_structure')
                    nr_skin_structure = gr.Slider(label='NR skin structure', minimum=-1.0, maximum=2.0, step=0.05, value=-1.0, elem_id='dlss_nr_skin_structure')
                    nr_automatic_mask = gr.Checkbox(label='Automatic mask', value=False, elem_id='dlss_nr_automatic_mask')

            with gr.Accordion('DLSS SuperSample', open=True, elem_id='dlss_ss'):
                with gr.Row():
                    ss_enabled = gr.Checkbox(label='SR enable', value=False, elem_id='dlss_ss_enabled')
                    ss_append = gr.Checkbox(label='Append result', value=True, elem_id='dlss_ss_append')
                with gr.Row():
                    ss_vsr_quality = gr.Dropdown(label='SS VSR quality', choices=["1: Low", "2: Medium", "3: High", "4: Ultra"], value="4: Ultra", type='value', elem_id='dlss_ss_vsr_quality')
                with gr.Row():
                    ss_size_mode = gr.Dropdown(label='SS size mode', choices=['Scale factor', 'Target size'], value='Scale factor', elem_id='dlss_ss_size_mode')
                with gr.Row():
                    ss_scale_factor = gr.Slider(label='SS scale factor', minimum=1.0, maximum=8.0, step=0.05, value=2.0, elem_id='dlss_ss_scale_factor')
                with gr.Row():
                    ss_width = gr.Number(label='SS width', minimum=64, maximum=16384, step=8, value=3840, elem_id='dlss_ss_width')
                    ss_height = gr.Number(label='SS height', minimum=64, maximum=16384, step=8, value=2160, elem_id='dlss_ss_height')

            with gr.Accordion('DLSS FrameGen', open=True, elem_id='dlss_fg'):
                with gr.Row():
                    fg_enabled = gr.Checkbox(label='FG enable', value=False, elem_id='dlss_fg_enabled')
                with gr.Row():
                    fg_source_fps = gr.Dropdown(label='Source FPS', choices=FPS_CHOICES, value='24', elem_id='dlss_fg_source_fps')
                    fg_target_fps = gr.Dropdown(label='Target FPS', choices=FPS_CHOICES, value='60', elem_id='dlss_fg_target_fps')
                with gr.Row():
                    fg_engine = gr.Dropdown(label='Engine', choices=['Auto', 'Native DLSSG', 'Cascade'], value='Auto', elem_id='dlss_fg_engine')

        return [nr_enabled, nr_append, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset, ss_enabled, ss_append, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height, fg_enabled, fg_source_fps, fg_target_fps, fg_engine]

    def install(self):
        note = textwrap.dedent("""\
            ### Install
            1. Download and unpack: [DLSS 5 Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v7.0)
            2. Enter the path to the unpacked package
            3. Press verify
            ### Notes
            - Package info is stored for future use on sucessful verification
            - DLSS controller process is started on first use
            - Use status to check the current state of the DLSS controller
            - Use reset to restore the DLSS controller to its default state
            - Use shutdown to stop the DLSS controller process
        """)
        return gr.update(value=note, visible=True)

    def verify(self, pkg_path):
        log.info(f'DLSS verify: path="{pkg_path}"')
        if not os.path.exists(pkg_path) or not os.path.isdir(pkg_path):
            log.error(f'DLSS: path="{pkg_path}" not found')
            return { 'error': 'package path not found' }
        if not c.controller.get_python(pkg_path):
            return { 'error': 'python not found in package path' }
        shared.opts.dlss_pkg_path = pkg_path
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
        log.debug(f'DLSS: gpu={report.get("gpu", "unknown")} checks={checks}')
        return report

    def status(self, pkg_path):
        log.info(f'DLSS status: path="{pkg_path}"')
        response = c.controller.call(pkg_path, 'status', {})
        if response.get('status') != 'ok':
            error = response.get('error') or {}
            log.error(f'DLSS: {error.get("message")}')
            return { 'error': error.get('message', 'unknown error') }
        return response.get('result', {})

    def reset(self, pkg_path):
        log.info(f'DLSS reset: path="{pkg_path}"')
        response = c.controller.call(pkg_path, 'reset', {})
        if response.get('status') != 'ok':
            error = response.get('error') or {}
            log.error(f'DLSS: {error.get("message")}')
            return { 'error': error.get('message', 'unknown error') }
        return response.get('result', {})

    def shutdown(self, pkg_path):
        log.info(f'DLSS shutdown: path="{pkg_path}"')
        if not c.controller.is_alive():
            return { 'shutdown': True, 'note': 'controller was not running' }
        c.controller.stop()
        return { 'shutdown': True }

    def supersample(self, pkg_path, images, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height):
        try:
            options = {
                'vsr_quality': int(ss_vsr_quality[0]),
                'size_mode': ss_size_mode,
                'scale_factor': float(ss_scale_factor),
                'width': int(ss_width),
                'height': int(ss_height),
                'aspect_lock': False,
            }
            response = c.controller.call(pkg_path, 'upscale', { 'images': c.images_to_nchw(images), 'options': options })
            if response.get('status') != 'ok':
                error = response.get('error') or {}
                log.error(f'DLSS: {error.get("message")}')
                return None
            return c.nchw_to_images(response.get('result'))
        except Exception as e:
            log.error(f'DLSS: {e}')
            errors.display(e, 'DLSS')
            return None

    def neuralrender(self, pkg_path, images, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset):
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
            response = c.controller.call(pkg_path, 'render', { 'images': c.images_to_nchw(images), 'options': options })
            if response.get('status') != 'ok':
                error = response.get('error') or {}
                log.error(f'DLSS: {error.get("message")}')
                return None
            return c.nchw_to_images(response.get('result'))
        except Exception as e:
            log.error(f'DLSS: {e}')
            errors.display(e, 'DLSS')
            return None

    def framegen(self, pkg_path, images, fg_source_fps, fg_target_fps, fg_engine):
        try:
            if len(images) < 2:
                log.warning('DLSS: FrameGen requires at least two frames, skipping')
                return None
            options = { 'ai_gpu_uuid': 'auto', 'engine': fg_engine }
            response = c.controller.call(
                pkg_path, 'framegen',
                { 'frames': c.images_to_nchw(images), 'source_fps': fg_source_fps, 'target_fps': fg_target_fps, 'options': options },
                timeout=120.0,
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

    def dlss(self, p, pp,
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

        if hasattr(pp, 'images') and pp.images is not None and len(pp.images) > 0:
            inputs = pp.images
        elif hasattr(pp, 'image') and pp.image is not None:
            inputs = [pp.image]
        else:
            return None

        # cast to appropriate types
        nr_intensity = float(nr_intensity)
        nr_local_tone = float(nr_local_tone)
        nr_local_structure = float(nr_local_structure)
        nr_skin_structure = float(nr_skin_structure)
        nr_upscaling_factor = float(nr_upscaling_factor)
        ss_width = int(ss_width)
        ss_height = int(ss_height)
        ss_scale_factor = float(ss_scale_factor)
        fg_source_fps = float(fg_source_fps)
        fg_target_fps = float(fg_target_fps)

        images = []
        originals = []
        current_images = inputs
        t = timer.Timer()

        if ss_enabled:
            t0 = time.time()
            p.extra_generation_params["DLSSSuperSample"] = True
            log.debug(f'DLSS: method=SuperSample quality="{ss_vsr_quality}" mode="{ss_size_mode}" scale={ss_scale_factor} width={ss_width} height={ss_height}')
            if ss_append:
                originals.extend(current_images)
            output = self.supersample(pkg_path, current_images, ss_vsr_quality, ss_size_mode, ss_scale_factor, ss_width, ss_height)
            if debug:
                log.trace(f'DLSS: method=SuperSample images={len(output) if output else 0} time={time.time() - t0:.3f}')
            if output:
                images.extend(output)
                current_images = output
            t.ts('supersample', t0)

        if nr_enabled:
            t0 = time.time()
            p.extra_generation_params["DLSSNeuralRender"] = True
            log.debug(f'DLSS: method=NeuralRender style={nr_style} intensity={nr_intensity} tone={nr_local_tone} structure={nr_local_structure} skin={nr_skin_structure} scale={nr_upscaling_factor} preset={nr_preset} mask={nr_automatic_mask} model={nr_model_preset}')
            if nr_append:
                originals.extend(current_images)
            output = self.neuralrender(pkg_path, current_images, nr_style, nr_intensity, nr_local_tone, nr_local_structure, nr_skin_structure, nr_upscaling_factor, nr_preset, nr_automatic_mask, nr_model_preset)
            if debug:
                log.trace(f'DLSS: method=NeuralRender images={len(output) if output else 0} time={time.time() - t0:.3f}')
            if output:
                images.extend(output)
                current_images = output
            t.ts('neuralrender', t0)

        if fg_enabled:
            t0 = time.time()
            p.extra_generation_params["DLSSFrameGen"] = True
            log.debug(f'DLSS: method=FrameGen source={fg_source_fps} target={fg_target_fps} engine={fg_engine}')
            output = self.framegen(pkg_path, current_images, fg_source_fps, fg_target_fps, fg_engine)
            if debug:
                log.trace(f'DLSS: method=FrameGen images={len(output) if output else 0} time={time.time() - t0:.3f}')
            if output:
                images.extend(output)
                current_images = output
            t.ts('framegen', t0)

        log.debug(f'DLSS: images={len(images)} {t.summary(min_time=0)}')
        pp.images = images
        pp.originals = originals
        return pp

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, *args, **kwargs):
        # postprocess_image is intended to modify single image in-place so not suited for dlss
        pass

    def postprocess(self, p: processing.StableDiffusionProcessing, pp: processing.Processed, *args, **kwargs): # pylint: disable=arguments-differ,unused-argument
        _pp = self.dlss(p, pp, *args, **kwargs)
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


"""
add install notes
add postprocessing
add framegen
add video
"""
