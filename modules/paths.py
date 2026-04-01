# this module must not have any dependencies as it is a very first import before webui even starts
import os
import sys
import json
import argparse
import tempfile
from modules.logger import log


# parse args, parse again after we have the data-dir and early-read the config file
from modules.cmd_args import get_argv, add_core_args, add_config_arg
argv = get_argv()
parser = argparse.ArgumentParser(add_help=False)
add_core_args(parser)
cli = parser.parse_known_args(argv)[0]
add_config_arg(parser, cli.data_dir)
cli = parser.parse_known_args(argv)[0]
config_path = cli.config if os.path.isabs(cli.config) else os.path.join(cli.data_dir, cli.config)

try:
    with open(config_path, encoding='utf8') as f:
        config = json.load(f)
except Exception:
    config = {}

temp_dir = config.get('temp_dir', '')
if len(temp_dir) == 0:
    temp_dir = tempfile.gettempdir()
reference_path = os.path.join('models', 'Reference')
modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
data_path = cli.data_dir
models_config = cli.models_dir or config.get('models_dir') or 'models'
models_path = models_config if os.path.isabs(models_config) else os.path.join(data_path, models_config)
params_path = os.environ.get('SD_PATH_PARAMS', os.path.join(data_path, "params.txt"))
extensions_dir = cli.extensions_dir or os.path.join(data_path, "extensions")
extensions_builtin_dir = "extensions-builtin"
sd_configs_path = os.path.join(script_path, "configs")
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")
sd_model_file = cli.ckpt or os.path.join(script_path, 'model.safetensors') # not used
default_sd_model_file = sd_model_file # not used
debug = log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PATH')
paths = {}

if os.environ.get('SD_PATH_DEBUG', None) is not None:
    log.debug(f'Paths: script-path="{script_path}" data-dir="{data_path}" models-dir="{models_path}" config="{config_path}"')


def create_path(folder):
    if folder is None or folder == '':
        return
    if os.path.exists(folder):
        return
    try:
        os.makedirs(folder, exist_ok=True)
        log.info(f'Create: folder="{folder}"')
    except Exception as e:
        log.error(f'Create failed: folder="{folder}" {e}')


def resolve_output_path(base_path: str, specific_path: str) -> str:
    """
    Resolve output path by combining base and specific paths.

    - If specific_path is absolute, return it directly (base is ignored)
    - If base_path is set and specific_path is relative, join them
    - If base_path is empty/None, return specific_path as-is
    """
    if not specific_path:
        return base_path or ''
    if os.path.isabs(specific_path):
        return specific_path
    if base_path:
        return os.path.normpath(os.path.join(base_path, specific_path))
    return specific_path


def create_paths(opts):
    def fix_path(folder):
        tgt = None
        if folder in opts.data:
            tgt = opts.data[folder]
        elif folder in opts.data_labels:
            tgt = opts.data_labels[folder].default
        else:
            log.warning(f'Path: folder="{folder}" unknown')
        if tgt is None or tgt == '':
            return tgt
        fix = tgt
        if not os.path.isabs(tgt) and len(data_path) > 0 and not tgt.startswith(data_path): # path is already relative to data_path
            fix = os.path.join(data_path, fix)
        if fix.startswith('..'):
            fix = os.path.abspath(fix)
        fix = fix if os.path.isabs(fix) else os.path.relpath(fix, script_path)
        opts.data[folder] = fix
        debug(f'Paths: folder="{folder}" original="{tgt}" target="{fix}"')
        return opts.data[folder]

    create_path(data_path)
    create_path(script_path)
    create_path(models_path)
    create_path(sd_configs_path)
    create_path(extensions_dir)
    create_path(extensions_builtin_dir)
    create_path(fix_path('temp_dir'))
    create_path(fix_path('ckpt_dir'))
    create_path(fix_path('diffusers_dir'))
    create_path(fix_path('hfcache_dir'))
    create_path(fix_path('vae_dir'))
    create_path(fix_path('unet_dir'))
    create_path(fix_path('te_dir'))
    create_path(fix_path('lora_dir'))
    create_path(fix_path('tunable_dir'))
    create_path(fix_path('embeddings_dir'))
    create_path(fix_path('onnx_temp_dir'))
    create_path(fix_path('outdir_samples'))
    create_path(fix_path('outdir_txt2img_samples'))
    create_path(fix_path('outdir_img2img_samples'))
    create_path(fix_path('outdir_control_samples'))
    create_path(fix_path('outdir_extras_samples'))
    create_path(fix_path('outdir_init_images'))
    create_path(fix_path('outdir_grids'))
    create_path(fix_path('outdir_txt2img_grids'))
    create_path(fix_path('outdir_img2img_grids'))
    create_path(fix_path('outdir_control_grids'))
    create_path(fix_path('outdir_save'))
    create_path(fix_path('outdir_video'))
    create_path(fix_path('styles_dir'))
    create_path(fix_path('yolo_dir'))
    create_path(fix_path('wildcards_dir'))

    # Create resolved output paths (base + specific)
    base_samples = opts.data.get('outdir_samples', '')
    base_grids = opts.data.get('outdir_grids', '')
    if base_samples:
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_txt2img_samples', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_img2img_samples', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_control_samples', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_extras_samples', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_save', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_video', '')))
        create_path(resolve_output_path(base_samples, opts.data.get('outdir_init_images', '')))
    if base_grids:
        create_path(resolve_output_path(base_grids, opts.data.get('outdir_txt2img_grids', '')))
        create_path(resolve_output_path(base_grids, opts.data.get('outdir_img2img_grids', '')))
        create_path(resolve_output_path(base_grids, opts.data.get('outdir_control_grids', '')))


class Prioritize:
    def __init__(self, name):
        self.name = name
        self.path = None

    def __enter__(self):
        self.path = sys.path.copy()
        sys.path = [paths[self.name]] + sys.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.path
        self.path = None
