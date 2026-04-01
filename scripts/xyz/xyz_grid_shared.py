# pylint: disable=unused-argument

import os
import re
from modules import shared, processing, sd_samplers, sd_models, sd_vae, sd_unet
from modules.logger import log


re_range = re.compile(r'([-+]?[0-9]*\.?[0-9]+)-([-+]?[0-9]*\.?[0-9]+):?([0-9]+)?')
re_plain_comma = re.compile(r"(?<!\\),")


def restore_comma(val: str):
    return val.replace(r"\,", ",")


def apply_field(field):
    def fun(p, x, xs):
        log.debug(f'XYZ grid apply field: {field}={x}')
        setattr(p, field, x)
    return fun


def apply_task_arg(field):
    def fun(p, x, xs):
        log.debug(f'XYZ grid apply task-arg: {field}={x}')
        p.task_args[field] = x
    return fun


def apply_task_args(p, x, xs):
    for section in x.split(';'):
        k, v = section.split('=')
        k, v = k.strip(), v.strip()
        if v.replace('.','',1).isdigit():
            v = float(v) if '.' in v else int(v)
        p.task_args[k] = v
        log.debug(f'XYZ grid apply task-arg: {k}={type(v)}:{v}')


def apply_processing(p, x, xs):
    for section in x.split(';'):
        k, v = section.split('=')
        k, v = k.strip(), v.strip()
        if v.replace('.','',1).isdigit():
            v = float(v) if '.' in v else int(v)
        found = 'existing' if hasattr(p, k) else 'new'
        setattr(p, k, v)
        log.debug(f'XYZ grid apply processing-arg: type={found} {k}={type(v)}:{v} ')


def apply_options(p, x, xs):
    for section in x.split(';'):
        k, v = section.split('=')
        k, v = k.strip(), v.strip()
        if v.replace('.','',1).isdigit():
            v = float(v) if '.' in v else int(v)
        found = 'existing' if v in shared.opts.data else 'new'
        shared.opts.data[k] = v
        log.debug(f'XYZ grid apply options: type={found} {k}={type(v)}:{v} ')


def apply_setting(field):
    def fun(p, x, xs):
        t = type(shared.opts.get(field))
        if t == bool:
            if isinstance(x, str):
                x = x.lower() in ['true', 't', 'yes', 'y']
            if isinstance(x, int) or isinstance(x, float):
                x = x > 0
        log.debug(f'XYZ grid apply setting: {field}={t}:{x}')
        shared.opts.data[field] = x
    return fun


def apply_seed(p, x, xs):
    p.seed = x
    p.all_seeds = None
    log.debug(f'XYZ grid apply seed: {x}')


def apply_prompt(positive, negative, p, x, xs):
    for s in xs:
        log.debug(f'XYZ grid apply prompt: fields={positive}/{negative} "{s}"="{x}"')
        orig_positive = getattr(p, positive)
        orig_negative = getattr(p, negative)
        if s in orig_positive:
            setattr(p, positive, orig_positive.replace(s, x))
        if s in orig_negative:
            setattr(p, negative, orig_negative.replace(s, x))


def apply_prompt_primary(p, x, xs):
    apply_prompt('prompt', 'negative_prompt', p, x, xs)
    p.all_prompts = None
    p.all_negative_prompts = None


def apply_prompt_refine(p, x, xs):
    apply_prompt('refiner_prompt', 'refiner_negative', p, x, xs)


def apply_prompt_detailer(p, x, xs):
    apply_prompt('detailer_prompt', 'detailer_negative', p, x, xs)


def apply_prompt_all(p, x, xs):
    apply_prompt('prompt', 'negative_prompt', p, x, xs)
    apply_prompt('refiner_prompt', 'refiner_negative', p, x, xs)
    apply_prompt('detailer_prompt', 'detailer_negative', p, x, xs)


def apply_order(p, x, xs):
    token_order = []
    for token in x:
        token_order.append((p.prompt.find(token), token))
    token_order.sort(key=lambda t: t[0])
    prompt_parts = []
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.sampler_name = sampler_name
    log.debug(f'XYZ grid apply sampler: "{x}"')


def apply_hr_sampler_name(p, x, xs):
    hr_sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if hr_sampler_name is None:
        log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.hr_sampler_name = hr_sampler_name
    log.debug(f'XYZ grid apply HR sampler: "{x}"')


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            log.warning(f"XYZ grid: unknown sampler: {x}")


def apply_sdnq_quant(p, x, xs):
    shared.opts.sdnq_quantize_weights_mode = x
    sd_models.unload_model_weights(op='model')
    sd_models.reload_model_weights()
    log.debug(f'XYZ grid apply sdnq quant: mode="{x}"')


def apply_sdnq_quant_te(p, x, xs):
    shared.opts.sdnq_quantize_weights_mode_te = x
    sd_models.unload_model_weights(op='model')
    sd_models.reload_model_weights()
    log.debug(f'XYZ grid apply sdnq quant te: mode="{x}"')


def apply_checkpoint(p, x, xs):
    if x == shared.opts.sd_model_checkpoint:
        return
    info = sd_models.get_closest_checkpoint_match(x)
    if info is None:
        log.warning(f"XYZ grid: apply checkpoint unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name
    log.debug(f'XYZ grid apply checkpoint: "{x}"')


def apply_refiner(p, x, xs):
    if x == shared.opts.sd_model_refiner:
        return
    if x == 'None':
        return
    info = sd_models.get_closest_checkpoint_match(x)
    if info is None:
        log.warning(f"XYZ grid: apply refiner unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_refiner, info)
        p.override_settings['sd_model_refiner'] = info.name
    log.debug(f'XYZ grid apply refiner: "{x}"')


def apply_unet(p, x, xs):
    if x == shared.opts.sd_unet:
        return
    if x == 'None':
        return
    p.override_settings['sd_unet'] = x
    shared.opts.data['sd_unet'] = x
    sd_unet.load_unet(shared.sd_model)
    log.debug(f'XYZ grid apply unet: "{x}"')


def apply_clip_skip(p, x, xs):
    p.clip_skip = x
    log.debug(f'XYZ grid apply clip-skip: "{x}"')


def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            log.warning(f"No VAE found for {name}; using automatic")
            return sd_vae.unspecified
        else:
            return sd_vae.vae_dict[choices[0]]


def apply_vae(p, x, xs):
    sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))
    log.debug(f'XYZ grid apply VAE: "{x}"')


def list_lora():
    import sys
    lora = [v for k, v in sys.modules.items() if k == 'networks' or k == 'modules.lora.lora_load'][0]
    loras = [v.fullname for v in lora.available_networks.values()]
    return ['None'] + sorted(loras)


def apply_lora(p, x, xs):
    p.all_prompts = None
    p.all_negative_prompts = None
    if x == 'None':
        return
    x = os.path.basename(x)
    p.prompt = p.prompt + f" <lora:{x}:{shared.opts.extra_networks_default_multiplier}>"
    log.debug(f'XYZ grid apply LoRA: "{x}"')


def apply_lora_strength(p, x, xs):
    log.debug(f'XYZ grid apply LoRA strength: "{x}"')
    p.prompt = p.prompt.replace(':1.0>', '>')
    p.prompt = p.prompt.replace(f':{shared.opts.extra_networks_default_multiplier}>', '>')
    p.all_prompts = None
    p.all_negative_prompts = None
    shared.opts.data['extra_networks_default_multiplier'] = x


def apply_te(p, x, xs):
    shared.opts.data["sd_text_encoder"] = x
    sd_models.reload_text_encoder()
    log.debug(f'XYZ grid apply text-encoder: "{x}"')


def apply_guidance(p, x, xs):
    from modules.modular_guiders import guiders
    guiders = list(guiders.keys())
    p.guidance_name = [g for g in guiders if g.lower().startswith(x.lower())][0]
    log.debug(f'XYZ grid apply guidance: "{p.guidance_name}"')


def apply_styles(p: processing.StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))
    log.debug(f'XYZ grid apply style: "{x}"')


def apply_upscaler(p: processing.StableDiffusionProcessingTxt2Img, opt, x):
    p.enable_hr = True
    p.hr_force = True
    p.denoising_strength = 0.0
    p.hr_upscaler = opt
    log.debug(f'XYZ grid apply upscaler: "{x}"')


def apply_context(p: processing.StableDiffusionProcessingTxt2Img, opt, x):
    p.resize_mode = 5
    p.resize_context = opt
    log.debug(f'XYZ grid apply resize-context: "{x}"')


def apply_detailer(p, opt, x):
    p.detailer_enabled = bool(opt)
    log.debug(f'XYZ grid apply detailer: "{x}"')


def apply_control(field):
    def fun(p, x, xs):
        init_images = getattr(p, 'orig_init_images', None) or getattr(p, 'init_images', None) or getattr(p, 'orig_init_images', None) or []
        if init_images is None or len(init_images) == 0:
            log.error('XYZ grid apply control: init image is required')
            return
        if field in ['controlnet', 't2i adapter', 'processor']:
            from modules.control import run, processor
            unit_type = 'controlnet' # set default
            if field in ['controlnet', 't2i adapter']:
                unit_type = field
                model_id = x
                process_id = run.unit.current[0].process_id if len(run.unit.current) > 0 else None
            elif field == 'processor':
                model_id = run.unit.current[0].model_id if len(run.unit.current) > 0 else None
                process_id = x
            else:
                model_id = None
                process_id = None
            start = run.unit.current[0].start if len(run.unit.current) > 0 else 0
            end = run.unit.current[0].end if len(run.unit.current) > 0 else 1.0
            strength = run.unit.current[0].model_strength if len(run.unit.current) > 0 else 1.0
            unit = run.unit.Unit(
                    index = 0,
                    enabled = True,
                    unit_type = unit_type,
                    model_id = getattr(model_id, 'value', model_id), # gradio-component-to-string
                    process_id = getattr(process_id, 'value', process_id),
                    start = getattr(start, 'value', start),
                    end = getattr(end, 'value', end),
                    strength = getattr(strength, 'value', strength),
            )
            log.debug(f'XYZ grid apply control: {field}="{x}" unit={unit}')
            if len(run.unit.current) > 0:
                if hasattr(run.unit.current[0], 'reset'):
                    run.unit.current[0].reset()
                run.unit.current[0] = unit
            else:
                run.unit.current = [unit]
            run.init_units(run.unit.current)
            active_process, active_model, active_strength, active_start, active_end, active_units = run.check_active(p, unit.type, run.unit.current)
            has_models, selected_models, control_conditioning, control_guidance_start, control_guidance_end = run.check_enabled(p, unit.type, run.unit.current, active_model, active_strength, active_start, active_end)
            pipe = run.set_pipe(p, has_models, unit.type, selected_models, active_model, active_strength, active_units, control_conditioning, control_guidance_start, control_guidance_end)
            _processed_image, _blended_image = processor.preprocess_image(p, pipe, input_image=init_images[0], unit_type=unit.type, active_process=active_process, active_model=active_model, selected_models=selected_models, has_models=has_models)
            if pipe is not None:
                shared.sd_model = pipe
        elif field == 'control_start':
            log.debug(f'XYZ grid apply control: {field}={x}')
            p.task_args['control_guidance_start'] = float(x)
        elif field == 'control_end':
            log.debug(f'XYZ grid apply control: {field}={x}')
            p.task_args['control_guidance_end'] = float(x)
        elif field == 'control_strength':
            log.debug(f'XYZ grid apply control: {field}={x}')
            p.task_args['adapter_conditioning_scale'] = float(x)
            p.task_args['controlnet_conditioning_scale'] = float(x)
    return fun


def apply_override(field):
    def fun(p, x, xs):
        p.override_settings[field] = x
        log.debug(f'XYZ grid apply override: "{field}"="{x}"')
    return fun


def format_bool(p, opt, x):
    return f"{opt.label}: {x}"


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 4)
    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 4)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def list_to_csv_string(data_list: list):
    return ",".join(data_list)
