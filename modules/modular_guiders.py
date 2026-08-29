import inspect
import diffusers
from modules import errors, shared, processing
from modules.logger import log


guiders = {
    # 'None': { 'cls': None, 'args': {}, },
    'Default': { 'cls': None, 'args': {}, },
    'CFG: ClassifierFreeGuidance': { 'cls': diffusers.ClassifierFreeGuidance, 'args': {} },
    'Auto: AutoGuidance': { 'cls': diffusers.AutoGuidance, 'args': {} },
    'Zero: ClassifierFreeZeroStar': { 'cls': diffusers.ClassifierFreeZeroStarGuidance, 'args': {} },
    'PAG: PerturbedAttentionGuidance': { 'cls': diffusers.PerturbedAttentionGuidance, 'args': {} },
    'APG: AdaptiveProjectedGuidance': { 'cls': diffusers.AdaptiveProjectedGuidance, 'args': {} },
    'SLG: SkipLayerGuidance': { 'cls': diffusers.SkipLayerGuidance, 'args': {} },
    'SEG: SmoothedEnergyGuidance': { 'cls': diffusers.SmoothedEnergyGuidance, 'args': {} },
    'TCFG: TangentialClassifierFreeGuidance': { 'cls': diffusers.TangentialClassifierFreeGuidance, 'args': {} },
    'FDG: FrequencyDecoupledGuidance': { 'cls': diffusers.FrequencyDecoupledGuidance, 'args': {} },
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
        log.info(f'Guider: name="{guidance_name}" cls={guider_cls.__name__ if guider_cls is not None else None} args={guider_args}')
        return
    if guidance_name == 'None':
        shared.sd_model.update_components(guider=None) # breaks the pipeline
        log.info(f'Guider: name="{guidance_name}"')
        return

    guider_info = guiders[guidance_name]
    guider_cls = guider_info['cls']

    guider_args = {}
    possible = list(inspect.signature(guider_cls.__init__).parameters) if guider_cls is not None else []
    if p.guidance_scale >= 0.0 and 'guidance_scale' in possible:
        guider_args['guidance_scale'] = float(p.guidance_scale)
    if p.guidance_rescale >= 0.0 and 'guidance_rescale' in possible:
        guider_args['guidance_rescale'] = float(p.guidance_rescale)
    if p.guidance_start >= 0.0 and 'start' in possible:
        guider_args['start'] = float(p.guidance_start)
    if p.guidance_stop >= 0.0 and 'stop' in possible:
        guider_args['stop'] = float(p.guidance_stop)

    """
    import modules.ui_guidance
    for k, v in modules.ui_guidance.get_modular_args().items():
        log.trace(f'Guiders: arg={k} value={v}')
    """

    log.warning('Guiders: advanced parameters are not yet implemented') # TODO: guiders
    """
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
    """
    if guider_cls is not None:
        try:
            guider_instance: diffusers.BaseGuidance = guider_cls(**guider_args)
            log.info(f'Guider: name="{guidance_name}" cls={guider_cls.__name__} args={guider_args}')
            shared.sd_model.update_components(guider=guider_instance)
        except Exception as e:
            log.error(f'Guider: name="{guidance_name}" cls={guider_cls.__name__} args={guider_args} {e}')
            errors.display(e, 'Guiders')
            return
    else:
        log.warning(f'Guider: name="{guidance_name}" cls=None args={guider_args}')
