import re
import transformers
from modules.logger import log


def get_default_args(model):
    to_remove = ['_from_model_config', 'transformers_version']
    config = {}
    for k, v in transformers.GenerationConfig._get_default_generation_params().items(): # pylint: disable=protected-access
        if v is not None:
            config[k] = v
    for k, v in transformers.GenerationConfig.from_model_config(model.config).to_dict().items():
        if v is not None:
            config[k] = v
    for k, v in model.generation_config.to_dict().items():
        if v is not None:
            config[k] = v
    config = {k: v for k, v in config.items() if k not in to_remove}
    return config

def get_custom_args(model, args_str):
    args = {}
    if args_str is not None and len(args_str) > 0:
        default_args = get_default_args(model)
        pairs = re.split(r'[;\n]+', args_str)
        for pair in pairs:
            if '=' in pair or ':' in pair:
                key, value = re.split(r'[=:]', pair, maxsplit=1)
                key = key.strip()
                value = value.strip()
                if key not in default_args:
                    log.warning(f'Prompt enhance: key="{key}" invalid')
                    continue
                default_value = default_args[key]
                try:
                    if isinstance(default_value, bool):
                        value = value.lower() in ['true', '1', 'yes']
                    elif isinstance(default_value, int):
                        value = int(value)
                    elif isinstance(default_value, float):
                        value = float(value)
                    elif isinstance(default_value, list):
                        value = [v.strip() for v in value.split(',')]
                    elif isinstance(default_value, str):
                        pass
                except ValueError:
                    log.warning(f'Prompt enhance: key="{key}" value="{value}" typecast failed')
                if key and value:
                    args[key] = value
    return args
