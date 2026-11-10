import os

import gradio as gr
from modules import paths, processing, scripts_manager, scripts_postprocessing, shared
from modules.image.namegen import FilenameGenerator
from modules.logger import log
from scripts.hdr.hdr_process import algorithmic_hdr, resolve_hdr_format, save_hdr_image


def create_ui(parent: str):
    with gr.Accordion('HDR convert', open=False, elem_id=f'{parent}_hdr_accordion'):
        with gr.Row():
            method = gr.Dropdown(label='HDR mapping', choices=['None', 'Algorithmic'], value='None', elem_id=f'{parent}_hdr_method')
        with gr.Row():
            peak = gr.Slider(label='HDR peak brightness', minimum=100, maximum=10000, step=100, value=1000, elem_id=f'{parent}_hdr_peak')
            gamut = gr.Dropdown(label='HDR color gamut', choices=['Rec.2020', 'DCI-P3'], value='Rec.2020', elem_id=f'{parent}_hdr_gamut')
    return [method, peak, gamut]


def output_path(image, p, index: int, extension: str, pp=None) -> str:
    source_path = None
    if pp is not None and pp.output_dir:
        source_path = pp.source_path or getattr(image, 'filename', None)
        directory = pp.output_dir
    else:
        saved_path = getattr(image, 'already_saved_as', None)
        if saved_path:
            source_path = saved_path
        else:
            source_path = getattr(image, 'filename', None)
            if source_path and not os.path.isfile(source_path):
                source_path = None
        if source_path:
            directory = os.path.dirname(source_path) or '.'
        elif p is None:
            directory = paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_extras_samples)

    if p is not None:
        seed = p.all_seeds[index] if getattr(p, 'all_seeds', None) else p.seed
        prompt = p.all_prompts[index] if getattr(p, 'all_prompts', None) else p.prompt
        generator = FilenameGenerator(p, seed=seed, prompt=prompt, image=image)
        pattern = shared.opts.samples_filename_pattern or '[seq]-[prompt_words]'
        directory = getattr(p, 'outpath_samples', None) or paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save)
        basename = generator.apply(pattern)
    else:
        prompt = os.path.splitext(os.path.basename(source_path))[0] if source_path else ''
        generator = FilenameGenerator(None, seed=0, prompt=prompt, image=image)
        pattern = '[prompt]' if source_path else '[seq]-[image_hash]'
        basename = generator.apply(pattern)
        os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f'{basename}.{extension}')
    return generator.sequence(generator.sanitize(filename, base=directory))


def process_image(image, p, index: int, method: str, target_peak_nits: float, target_gamut: str, pp=None) -> bool:
    if image is None:
        return False
    if (method is None) or (method == '') or (method == 'None'):
        return False
    elif method == 'Algorithmic':
        hdr = algorithmic_hdr(image, target_peak_nits, target_gamut)
    else:
        raise ValueError(f'HDR: method={method} unsupported')
    hdr_format = resolve_hdr_format(shared.opts.samples_format)
    if hdr_format == 'jpeg-gain-map':
        extension = 'jpg'
    elif hdr_format == 'tiff16':
        extension = 'tiff'
    elif hdr_format == 'png-cicp':
        extension = 'png'
    elif hdr_format == 'exr':
        extension = 'exr'
    elif hdr_format == 'avif':
        extension = 'avif'
    elif hdr_format == 'jxl':
        extension = 'jxl'
    else:
        log.warning(f'HDR: skipped: samples_format="{shared.opts.samples_format}" is not a supported HDR format')
        return False
    filename = output_path(image, p, index, extension, pp=pp)
    saved = save_hdr_image(filename, hdr, image, hdr_format, target_peak_nits, target_gamut)
    if saved:
        log.info(f'HDR: saved="{filename}" method={method} peak={target_peak_nits} gamut={target_gamut}')
    return saved


class HDRScript(scripts_manager.Script):
    video_capable = False

    def title(self):
        return 'HDR'

    def show(self, _is_img2img):
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return create_ui(self.parent)

    def before_process(self, p, *args):
        method = args[0] if args else 'None'
        enabled = method is not None and method != '' and method != 'None'
        if enabled and not p.xyz:
            p.hdr_do_not_save_samples = p.do_not_save_samples
            if str(shared.opts.samples_format).lower() == 'exr':
                p.do_not_save_samples = True

    def postprocess(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args):
        if len(args) < 3:
            return processed
        method, peak, gamut = args
        if method is None or method == '' or method == 'None':
            return processed
        if p.xyz or not processing.get_opt(p, 'samples_save') or getattr(p, 'hdr_do_not_save_samples', p.do_not_save_samples):
            return processed
        for index, image in enumerate(processed.images):
            try:
                process_image(image, p, index, method, peak, gamut)
            except Exception as e:
                log.error(f'HDR: image={index} {e}')
        return processed


class HDRPostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = 'HDR'
    order = 31000

    def ui(self):
        method, peak, gamut = create_ui('postprocess')
        return {
            'hdr_method': method,
            'hdr_peak': peak,
            'hdr_gamut': gamut,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **kwargs):
        method = kwargs.get('hdr_method', 'None')
        if method is None or method == '' or method == 'None':
            return
        if pp.video is not None:
            log.warning('HDR: video processing is not supported')
            return
        if pp.image is None or not pp.save_output:
            return
        peak = kwargs.get('hdr_peak', 1000)
        gamut = kwargs.get('hdr_gamut', 'Rec.2020')
        hdr_format = resolve_hdr_format(shared.opts.samples_format)
        saved = process_image(pp.image, None, 0, method, peak, gamut, pp=pp)
        if saved:
            if hdr_format == 'exr':
                pp.save_output = False
            pp.info['HDR'] = f'{method}'
