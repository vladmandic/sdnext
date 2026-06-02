import numpy as np
from PIL import Image
from modules import scripts_postprocessing, shared
from modules.logger import log


class ScriptPostprocessingDetailer(scripts_postprocessing.ScriptPostprocessing):
    name = "Detailer"
    order = 15000

    def ui(self):
        # The detailer accordion (built by yolo.ui) now contains the Sampler sub-accordion too, so for 'extras'
        # it returns a 7th element: a dict of the sampler-block controls. Spread it into the control map; their
        # values are stamped onto the synthetic p in process()/make_processing(), applying to this pass only.
        enabled, prompt, negative, steps, strength, resolution, sampler_block = shared.yolo.ui('extras')
        return {
            "enabled": enabled,
            "prompt": prompt,
            "negative": negative,
            "steps": steps,
            "strength": strength,
            "resolution": resolution,
            **sampler_block,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage,  # pylint: disable=arguments-differ
                enabled=False, prompt='', negative='', steps=10, strength=0.3, resolution=1024,
                sampler='Default', prediction='default', shift=3.0, cfg_scale=6.0, options=None, seed=-1):
        if not enabled:
            return pp
        if shared.sd_model is None or not hasattr(shared.sd_model, 'sd_checkpoint_info'):
            log.warning('Detailer postprocess: no base model selected')
            pp.info["Detailer"] = "skipped (no base model selected)"
            return pp

        # The sampler block is stamped onto the synthetic p. The schedulers_* values become per-job overrides in
        # processing_helpers (they beat the global opts for this pass only); a named sampler is required for them
        # to take effect, 'Default' keeps the model scheduler. cfg_scale and hr_sampler_name apply directly.
        options = options or []
        overrides = {
            'hr_sampler_name': sampler,
            'schedulers_prediction_type': prediction,
            'schedulers_shift': shift,
            'cfg_scale': cfg_scale,
            'schedulers_use_loworder': 'low order' in options,
            'schedulers_use_thresholding': 'thresholding' in options,
            'schedulers_dynamic_shift': 'dynamic' in options,
            'schedulers_rescale_betas': 'rescale' in options,
        }
        log.info(f'Detailer postprocess: strength={strength} steps={steps} resolution={resolution} sampler={sampler} cfg={cfg_scale}')
        p = shared.yolo.make_processing(pp.image, prompt=prompt, negative=negative, steps=steps, strength=strength, resolution=resolution, seed=int(seed) if seed is not None else -1, overrides=overrides)

        try:
            result = shared.yolo.restore(np.array(pp.image), p)
        except Exception as e:
            log.error(f'Detailer postprocess: {e}')
            return pp

        # restore() returns list[ndarray] (detailed image at [0], annotated debug at [1] when enabled)
        # on success, or a single ndarray on early-return paths. The postprocessing pipeline is one
        # image per input, so the annotated debug image is dropped here; use /sdapi/v1/detail for it.
        if isinstance(result, list) and len(result) > 0:
            pp.image = Image.fromarray(result[0])
        elif isinstance(result, np.ndarray):
            pp.image = Image.fromarray(result)

        pp.info["Detailer"] = "Enabled"
        pp.info["Detailer strength"] = strength
        pp.info["Detailer steps"] = steps
        pp.info["Detailer resolution"] = resolution
        pp.info["Detailer sampler"] = sampler
        if prompt:
            pp.info["Detailer prompt"] = prompt
        if negative:
            pp.info["Detailer negative"] = negative
        return pp
