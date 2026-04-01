import diffusers
from modules import shared, errors, processing
from modules.logger import log


# ['Default', 'CFG', 'Zero', 'PAG', 'APG', 'SLG', 'SEG', 'TCFG', 'FDG']
guiders = {
    # 'None': { 'cls': None, 'args': {}, },
    'Default': { 'cls': None, 'args': {}, },
    'CFG: ClassifierFreeGuidance': { 'cls': diffusers.ClassifierFreeGuidance, 'args': {} },
    'Auto: AutoGuidance': { 'cls': diffusers.AutoGuidance, 'args': { 'dropout': 1.0, 'auto_guidance_layers': [7, 8, 9], 'auto_guidance_config': None } },
    'Zero: ClassifierFreeZeroStar': { 'cls': diffusers.ClassifierFreeZeroStarGuidance, 'args': { 'zero_init_steps': 1 } },
    'PAG: PerturbedAttentionGuidance': { 'cls': diffusers.PerturbedAttentionGuidance, 'args': { 'perturbed_guidance_scale': 2.8, 'perturbed_guidance_start': 0.01, 'perturbed_guidance_stop': 0.2, 'perturbed_guidance_layers': [7, 8, 9], 'perturbed_guidance_config': None } },
    'APG: AdaptiveProjectedGuidance': { 'cls': diffusers.AdaptiveProjectedGuidance, 'args': { 'adaptive_projected_guidance_momentum': -1, 'adaptive_projected_guidance_rescale': 15.0 } },
    'SLG: SkipLayerGuidance': { 'cls': diffusers.SkipLayerGuidance, 'args': { 'skip_layer_guidance_scale': 2.8, 'skip_layer_guidance_start': 0.01, 'skip_layer_guidance_stop': 0.2, 'skip_layer_guidance_layers': [7, 8, 9], 'skip_layer_config': None } },
    'SEG: SmoothedEnergyGuidance': { 'cls': diffusers.SmoothedEnergyGuidance, 'args': { 'seg_guidance_scale': 3.0, 'seg_blur_sigma': 9999999.0, 'seg_blur_threshold_inf': 9999.0, 'seg_guidance_start': 0.0, 'seg_guidance_stop': 1.0, 'seg_guidance_layers': [7, 8, 9], 'seg_guidance_config': None } },
    'TCFG: TangentialClassifierFreeGuidance': { 'cls': diffusers.TangentialClassifierFreeGuidance, 'args': {} },
    'FDG: FrequencyDecoupledGuidance': { 'cls': diffusers.FrequencyDecoupledGuidance, 'args': { 'guidance_scales': [10.0, 5.0], 'parallel_weights': 1.0, 'guidance_rescale_space': "data" } },
}
base_args = {
    'guidance_scale': 6.0,
    'guidance_rescale': 0.0,
    'start': 0.0,
    'stop': 1.0,
}


def set_guider(p: processing.StableDiffusionProcessing):
    guidance_name = p.guidance_name or 'Default'
    if guidance_name not in guiders:
        return

    if guidance_name == 'Default':
        if hasattr(shared.sd_model, 'default_guider'):
            guider_info = shared.sd_model.default_guider
            guider_cls = guider_info.type_hint if hasattr(guider_info, 'type_hint') else type(guider_info)
            shared.sd_model.update_components(guider=guider_info)
        elif hasattr(shared.sd_model, 'get_component_spec'):
            guider_info = shared.sd_model.get_component_spec("guider")
            guider_cls = guider_info.type_hint if hasattr(guider_info, 'type_hint') else type(guider_info)
            shared.sd_model.default_guider = guider_info
        elif hasattr(shared.sd_model, 'guider') and hasattr(shared.sd_model.guider, 'config'):
            guider_info = shared.sd_model.guider
            guider_cls = type(shared.sd_model.guider)
            # shared.sd_model.default_guider = guider_info
        else:
            guider_info = None
            guider_cls = None
        if guider_info is not None and guider_cls is not None and guider_info.config is not None:
            guider_args = {k: v for k, v in guider_info.config.items() if not k.startswith('_') and v is not None}
        else:
            guider_args = {}
        log.info(f'Guider: name={guidance_name} cls={guider_cls.__name__ if guider_cls is not None else None} args={guider_args}')
        return
    if guidance_name == 'None':
        shared.sd_model.update_components(guider=None) # breaks the pipeline
        log.info(f'Guider: name={guidance_name}')
        return

    guider_info = guiders[guidance_name]
    guider_cls = guider_info['cls']
    guider_args = {}
    for k, v in base_args.items():
        if v is not None and v >= 0.0:
            guider_args[k] = v
    log.warning('Guiders: partially implemented') # TODO: guiders
    for k, v in guider_info['args'].items():
        try:
            if k is None:
                pass
            elif k.endswith('_layers') and isinstance(v, str):
                guider_args[k] = [int(x.strip()) for x in v.split(',') if x.strip().isdigit()]
            elif k.endswith('_config'):
                # if lsc_enabled
                # guider_args[k] = diffusers.LayerSkipConfig(...)
                pass
            elif isinstance(v, list) and len(v) > 0:
                guider_args[k] = v
            elif isinstance(v, int) and (v >= 0):
                guider_args[k] = int(v)
            elif isinstance(v, float) and (v >= 0.0):
                guider_args[k] = float(v)
            elif isinstance(v, str) and (len(v) > 0):
                guider_args[k] = v
        except Exception as e:
            log.error(f'Guiders: arg={k} value={v} error={e}')
            errors.display(e, 'Guiders')
    # guider_args.update(guider_info['args'])
    if guider_cls is not None:
        try:
            guider_instance = guider_cls(**guider_args)
            log.info(f'Guider: name={guidance_name} cls={guider_cls.__name__} args={guider_args}')
            shared.sd_model.update_components(guider=guider_instance)
        except Exception as e:
            log.error(f'Guider: name={guidance_name} cls={guider_cls.__name__} args={guider_args} {e}')
            return
