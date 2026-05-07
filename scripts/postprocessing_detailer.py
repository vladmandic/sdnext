import numpy as np
from PIL import Image
from modules import scripts_postprocessing, shared, processing, images
from modules.logger import log
from modules.paths import resolve_output_path


class ScriptPostprocessingDetailer(scripts_postprocessing.ScriptPostprocessing):
    name = "Detailer"
    order = 15000

    def ui(self):
        # Reuse the existing 21-control detailer accordion. Returns a 6-tuple of
        # the per-request controls; the other 15 settings persist via shared.opts
        # through ui_settings_change at modules/postprocess/yolo.py:492-511.
        enabled, prompt, negative, steps, strength, resolution = shared.yolo.ui('extras')
        return {
            "enabled": enabled,
            "prompt": prompt,
            "negative": negative,
            "steps": steps,
            "strength": strength,
            "resolution": resolution,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage,  # pylint: disable=arguments-differ
                enabled=False, prompt='', negative='',
                steps=10, strength=0.3, resolution=1024):
        if not enabled:
            return pp
        if shared.sd_model is None or not hasattr(shared.sd_model, 'sd_checkpoint_info'):
            log.warning('Detailer postprocess: model not loaded')
            return pp

        log.info(f'Detailer postprocess: strength={strength} steps={steps} resolution={resolution}')

        outpath = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_extras_samples)

        p = processing.StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            prompt=prompt or '',
            negative_prompt=negative or '',
            init_images=[pp.image],
            outpath_samples=outpath,
            outpath_grids=outpath,
            batch_size=1,
            n_iter=1,
            steps=steps,
            denoising_strength=strength,
            width=pp.image.width,
            height=pp.image.height,
            detailer_enabled=True,
            detailer_prompt=prompt or '',
            detailer_negative=negative or '',
            detailer_steps=steps,
            detailer_strength=strength,
            detailer_resolution=resolution,
        )
        # Required by restore() at modules/postprocess/yolo.py:348-349 which indexes [0]
        p.all_prompts = [prompt or '']
        p.all_negative_prompts = [negative or '']
        p.all_seeds = [-1]
        p.all_subseeds = [-1]
        p.scripts = None
        p.is_control = False
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        try:
            result = shared.yolo.restore(np.array(pp.image), p)
        except Exception as e:
            log.error(f'Detailer postprocess: {e}')
            return pp

        # restore() returns list[ndarray] on success (length 1 or 2) or single ndarray
        # on early-return paths (interrupted, no models, denoise=0).
        if isinstance(result, list) and len(result) > 0:
            pp.image = Image.fromarray(result[0])
            if len(result) > 1 and result[1] is not None:
                annotated = result[1] if isinstance(result[1], Image.Image) else Image.fromarray(result[1])
                images.save_image(annotated, path=outpath, basename="", suffix="-detailer-detected", extension=shared.opts.samples_format, grid=False)
        elif isinstance(result, np.ndarray):
            pp.image = Image.fromarray(result)

        pp.info["Detailer"] = "Enabled"
        pp.info["Detailer strength"] = strength
        pp.info["Detailer steps"] = steps
        pp.info["Detailer resolution"] = resolution
        if prompt:
            pp.info["Detailer prompt"] = prompt
        if negative:
            pp.info["Detailer negative"] = negative
        return pp
