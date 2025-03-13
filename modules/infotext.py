import os
import re
import json


if os.environ.get('SD_PASTE_DEBUG', None) is not None:
    from modules.errors import log
    debug = log.trace
else:
    debug = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment
re_size = re.compile(r"^(\d+)x(\d+)$") # int x int
re_param = re.compile(r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)') # multi-word: value
re_lora = re.compile("<lora:([^:]+):")


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


# disabled by default can be enabled if needed
def check_lora(params):
    try:
        import modules.lora.networks as networks
        from modules.errors import log # pylint: disable=redefined-outer-name
    except Exception:
        return
    loras = [s.strip() for s in params.get('LoRA hashes', '').split(',')]
    found = []
    missing = []
    for l in loras:
        lora = networks.available_network_hash_lookup.get(l, None)
        if lora is not None:
            found.append(lora.name)
        else:
            missing.append(l)
    loras = [s.strip() for s in params.get('LoRA networks', '').split(',')]
    for l in loras:
        lora = networks.available_network_aliases.get(l, None)
        if lora is not None:
            found.append(lora.name)
        else:
            missing.append(l)
    # networks.available_network_aliases.get(name, None)
    loras = re_lora.findall(params.get('Prompt', ''))
    for l in loras:
        lora = networks.available_network_aliases.get(l, None)
        if lora is not None:
            found.append(lora.name)
        else:
            missing.append(l)
    log.debug(f'LoRA: found={list(set(found))} missing={list(set(missing))}')


def parse(infotext):
    if not isinstance(infotext, str):
        return {}
    debug(f'Raw: {infotext}')

    remaining = infotext.replace('\nSteps:', ' Steps:')
    # TODO infotext: handle using regex instead
    params = [' steps:', ' seed:', ' width:', ' height:', ' sampler:', ' size:', ' cfg scale:'] # first param is one of those
    params += ['\nsteps:', '\nseed:', '\nwidth:', '\nheight:', '\nsampler:', '\nsize:', '\ncfg scale:']
    params += ['.steps:', '.seed:', '.width:', '.height:', '.sampler:', '.size:', '.cfg scale:']

    prompt_end = [remaining.lower().find(p) for p in params if p in remaining.lower()]
    prompt_end += [remaining.lower().find('negative prompt:')]
    prompt_end = [p for p in prompt_end if p > -1]
    prompt_end = min(prompt_end) if len(prompt_end) > 0 else 0
    prompt = remaining[:prompt_end]
    remaining = remaining.replace(prompt, '')
    if prompt.lower().startswith('prompt: '):
        prompt = prompt[8:]
    # debug(f'Prompt: {prompt}')

    param_idx = [remaining.lower().find(p) for p in params if p in remaining.lower()]
    param_idx = [p for p in param_idx if p > -1]
    param_idx = min(param_idx) if len(param_idx) > 0 else 0
    negative = remaining[:param_idx] if param_idx > 0 else ''
    remaining = remaining.replace(negative, '')
    if negative.lower().startswith('negative prompt: '):
        negative = negative[16:]
    # debug(f'Negative: {negative}')

    params = dict(re_param.findall(remaining))
    if len(list(params)) == 0:
        params['Prompt'] = infotext
        return params
    params['Prompt'] = prompt
    params['Negative prompt'] = negative
    for key, val in params.copy().items():
        val = unquote(val).strip(" ,\n").replace('\\\n', '')
        size = re_size.match(val)
        if val.replace('.', '', 1).isdigit():
            params[key] = float(val) if '.' in val else int(val)
        elif val == "True":
            params[key] = True
        elif val == "False":
            params[key] = False
        elif key == 'VAE' and val == 'TAESD':
            params["VAE type"] = 'Tiny'
        elif size is not None:
            params[f"{key}-1"] = int(size.group(1))
            params[f"{key}-2"] = int(size.group(2))
        elif isinstance(params[key], str):
            params[key] = val
        debug(f'Param parsed: type={type(params[key])} "{key}"={params[key]} raw="{val}"')

    # check_lora(params)
    return params


mapping = [
    # Backend
    ('Backend', 'sd_backend'),
    # Models
    ('Model hash', 'sd_model_checkpoint'),
    ('Refiner', 'sd_model_refiner'),
    ('VAE', 'sd_vae'),
    ('TE', 'sd_text_encoder'),
    ('Unet', 'sd_unet'),
    # Other
    ('Parser', 'prompt_attention'),
    ('Color correction', 'img2img_color_correction'),
    # Samplers
    ('Sampler eta delta', 'eta_noise_seed_delta'),
    ('Sampler eta multiplier', 'initial_noise_multiplier'),
    ('Sampler timesteps', 'schedulers_timesteps'),
    ('Sampler spacing', 'schedulers_timestep_spacing'),
    ('Sampler sigma', 'schedulers_sigma'),
    ('Sampler order', 'schedulers_solver_order'),
    ('Sampler type', 'schedulers_prediction_type'),
    ('Sampler beta schedule', 'schedulers_beta_schedule'),
    ('Sampler low order', 'schedulers_use_loworder'),
    ('Sampler dynamic', 'schedulers_use_thresholding'),
    ('Sampler rescale', 'schedulers_rescale_betas'),
    ('Sampler beta start', 'schedulers_beta_start'),
    ('Sampler beta end', 'schedulers_beta_end'),
    ('Sampler range', 'schedulers_timesteps_range'),
    ('Sampler shift', 'schedulers_shift'),
    ('Sampler dynamic shift', 'schedulers_dynamic_shift'),
    # Token Merging
    ('Mask weight', 'inpainting_mask_weight'),
    ('ToMe', 'tome_ratio'),
    ('ToDo', 'todo_ratio'),
]


if __name__ == '__main__':
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s | %(message)s')
    debug = log.info

    import sys
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            with open(sys.argv[1], 'r', encoding='utf8') as f:
                parse(f.read())
        else:
            parse(sys.argv[1])
