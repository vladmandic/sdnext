import os
import re
import json


if os.environ.get('SD_PASTE_DEBUG', None) is not None:
    from modules.logger import log
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
    if text is None:
        return ''
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def parse(infotext):
    if not isinstance(infotext, str) or len(infotext.strip()) == 0:
        return {}
    debug(f'Raw: {infotext}')

    remaining = infotext.replace('\nSteps:', ' Steps:')
    params = ['steps: ', 'seed: ', 'width: ', 'height: ', 'sampler: ', 'size: ', 'cfg scale: ', 'pipeline: '] # first param is one of those
    prompts = ['negative prompt: ', 'template: ', 'negative template: '] # first prompt field is implied
    search = params + prompts

    def extract(keyword, text): # attempt to extract prompt from params
        debug(f'Params extract: keyword="{keyword}" text="{text}"')
        if keyword not in text.lower():
            return '', text
        if text.lower().startswith(f'{keyword}:'):
            text = text[len(keyword) + 1:]
        text = text.strip("\n").strip()
        idxs = [text.lower().find(param) for param in search if param in text.lower()]
        idx = min(idxs) if len(idxs) > 0 else 0
        res = text[:idx]
        text = text.replace(res, '')
        res = res.strip("\n").strip()
        debug(f'Params extract: keyword="{keyword}" idx={idx} res="{res}"')
        return res, text

    prompt, remaining = extract('', remaining) # prompt is always first
    negative, remaining = extract('negative prompt', remaining)
    template, remaining = extract('template', remaining)
    negative_template, remaining = extract('negative template', remaining)

    params = dict(re_param.findall(remaining))
    if len(list(params)) == 0:
        params['Prompt'] = infotext
        return params
    params['Prompt'] = prompt if len(prompt) > 0 else None
    params['Negative prompt'] = negative if len(negative) > 0 else None
    params['Template'] = template if len(template) > 0 else None
    params['Negative template'] = negative_template if len(negative_template) > 0 else None
    if params['Template'] == params['Prompt']:
        params.pop('Template', None)
    if params['Negative template'] == params['Negative prompt']:
        params.pop('Negative template', None)
    debug(f'Params: {params}')

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
        debug(f'Param parsed: type={type(params[key])} "{key}"="{params[key]}" raw="{val}"')

    return params


mapping = [
    # Models
    ('Model', 'sd_model_checkpoint'),
    ('Model hash', 'sd_model_checkpoint'),
    ('Refiner', 'sd_model_refiner'),
    ('VAE', 'sd_vae'),
    ('TE', 'sd_text_encoder'),
    ('Unet', 'sd_unet'),
    # Other
    ('Parser', 'prompt_attention'),
    ('Color correction', 'img2img_color_correction'),
    ('Color correction method', 'color_correction_method'),
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
            with open(sys.argv[1], encoding='utf8') as f:
                parse(f.read())
        else:
            parse(sys.argv[1])
