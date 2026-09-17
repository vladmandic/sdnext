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


def get_layers(layer_str: str):
    if not layer_str:
        return []
    try:
        # layers can be comma separated, e.g. "7, 8, 9" or range "7-9" or mixed "7, 8-10, 12"
        layers = []
        for part in layer_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                layers.extend(range(int(start), int(end) + 1))
            elif part.isdigit():
                layers.append(int(part))
        layers = sorted(set(layers))  # remove duplicates and sort
        return layers
    except Exception as e:
        log.error(f'Guiders layers: {e}')
        return []


def set_args(guidance_name: str):
    args = {}
    import modules.ui_guidance
    inputs = modules.ui_guidance.get_modular_args()
    # for k, v in inputs.items():
    #     log.trace(f'Guiders: arg={k} value={v}')
    if guidance_name.startswith('Default'):
        pass
    if guidance_name.startswith('CFG:'):
        pass
    if guidance_name.startswith('Auto:'):
        args['dropout'] = float(inputs.get('autoguidance_dropout', 1.0))
        args['auto_guidance_layers'] = get_layers(inputs.get('autoguidance_layers', []))
    if guidance_name.startswith('Zero:'):
        args['zero_init_steps'] = int(inputs.get('zerostar_init_steps', 1))
    if guidance_name.startswith('PAG:'):
        args['perturbed_guidance_scale'] = float(inputs.get('pag_scale', 1.0))
        args['perturbed_guidance_start'] = float(inputs.get('pag_start', 0.01))
        args['perturbed_guidance_stop'] = float(inputs.get('pag_stop', 0.2))
        args['perturbed_guidance_layers'] = get_layers(inputs.get('pag_layers', []))
    if guidance_name.startswith('APG:'):
        args['adaptive_projected_guidance_momentum'] = float(inputs.get('apg_momentum', None)) if inputs.get('apg_momentum', -1) >= 0 else None
        args['adaptive_projected_guidance_rescale'] = float(inputs.get('apg_rescale', 15.0))
    if guidance_name.startswith('SLG:'):
        args['skip_layer_guidance_scale'] = float(inputs.get('slg_scale', 1.0))
        args['skip_layer_guidance_start'] = float(inputs.get('slg_start', 0.01))
        args['skip_layer_guidance_stop'] = float(inputs.get('slg_stop', 0.2))
        args['skip_layer_guidance_layers'] = get_layers(inputs.get('slg_layers', []))
    if guidance_name.startswith('SEG:'):
        args['seg_guidance_scale'] = float(inputs.get('seg_scale', 3.0))
        args['seg_blur_sigma'] = float(inputs.get('seg_blur_sigma', 9999999.0))
        args['seg_blur_threshold_inf'] = float(inputs.get('seg_blur_threshold_inf', 9999.0))
        args['seg_guidance_start'] = float(inputs.get('seg_start', 0.0))
        args['seg_guidance_stop'] = float(inputs.get('seg_stop', 1.0))
        args['seg_guidance_layers'] = get_layers(inputs.get('seg_layers', []))
    if guidance_name.startswith('TCFG:'):
        pass
    if guidance_name.startswith('FDG:'):
        args['guidance_scales'] = [float(x.strip()) for x in inputs.get('fdg_scales', '5.0').split(',')]
        args['parallel_weights'] = float(inputs.get('fdg_weights', 1.0))
        args['guidance_rescale_space'] = inputs.get('fdg_rescale_space', 'data')
    return args


def set_guider(p: processing.StableDiffusionProcessing, phase: str | None = None):
    guidance_name = p.cfg_name or 'Default'
    if guidance_name not in guiders:
        return

    if not hasattr(shared.sd_model, 'default_guider'): # store default guider
        guider_info = shared.sd_model.get_component_spec("guider")
        guider_cls = guider_info.type_hint if hasattr(guider_info, 'type_hint') else type(guider_info)
        shared.sd_model.default_guider = guider_cls

    if guidance_name == 'None':
        shared.sd_model.update_components(guider=None) # breaks the pipeline
        log.info(f'Pipeline: guidance="{guidance_name}"')
        return
    elif guidance_name == 'Default':
        guider_cls = shared.sd_model.default_guider
    else:
        guider_info = guiders[guidance_name]
        guider_cls = guider_info['cls']

    guider_args = set_args(guidance_name)
    possible = inspect.signature(guider_cls.__init__).parameters if guider_cls is not None else []
    if 'guidance_scale' in list(possible):
        if (phase == 'hires' or phase == 'refine') and p.cfg_image >= 0.0:
            guider_args['guidance_scale'] = float(p.cfg_image)
        elif p.cfg_scale >= 0.0:
            guider_args['guidance_scale'] = float(p.cfg_scale)
    if p.cfg_rescale >= 0.0 and 'guidance_rescale' in list(possible):
        guider_args['guidance_rescale'] = float(p.cfg_rescale)
    if p.cfg_start >= 0.0 and 'start' in list(possible):
        guider_args['start'] = float(p.cfg_start)
    if p.cfg_stop >= 0.0 and 'stop' in list(possible):
        guider_args['stop'] = float(p.cfg_stop)

    if guider_cls is not None:
        try:
            guider_instance: diffusers.BaseGuidance = guider_cls(**guider_args)
            log.info(f'Pipeline: guidance="{guidance_name}" cls={guider_cls.__name__} args={guider_args}')
            shared.sd_model.update_components(guider=guider_instance)
        except Exception as e:
            log.error(f'Pipeline: guidance="{guidance_name}" cls={guider_cls.__name__} args={guider_args} {e}')
            errors.display(e, 'Guiders')
            return
    else:
        log.warning(f'Pipeline: guidance="{guidance_name}" cls=None args={guider_args}')
