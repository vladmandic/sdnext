from __future__ import annotations
import io
import os
import sys
import time
import contextlib

from enum import Enum
from typing import TYPE_CHECKING
import gradio as gr
from installer import print_dict # pylint: disable=unused-import
from modules.logger import log
log.debug('Initializing: shared module')

import modules.memmon
import modules.paths as paths
from modules.json_helpers import readfile # pylint: disable=W0611
from modules.shared_helpers import listdir, req # pylint: disable=W0611
from modules import errors, devices, shared_state, cmd_args, theme, history, files_cache # pylint: disable=unused-import
from modules.shared_defaults import get_default_modes
from modules.memstats import memory_stats # pylint: disable=unused-import

log.debug('Initializing: pipelines')
from modules import shared_items # pylint: disable=unused-import
from modules.caption.openclip import get_clip_models, refresh_clip_models # pylint: disable=unused-import
from modules.caption.vqa import vlm_models, vlm_prompts, vlm_system, vlm_default # pylint: disable=unused-import


if TYPE_CHECKING:
    # Behavior modified by __future__.annotations
    from diffusers import DiffusionPipeline
    from modules.shared_legacy import LegacyOption
    from modules.ui_extra_networks import ExtraNetworksPage


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


errors.install([gr])
demo: gr.Blocks | None = None
api = None
url = 'https://github.com/vladmandic/sdnext'
cmd_opts = cmd_args.parse_args()
parser = cmd_args.parser
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
listfiles = listdir
xformers_available = False
compiled_model_state = None
sd_upscalers = []
detailers = []
yolo = None
tab_names = []
extra_networks: list[ExtraNetworksPage] = []
hypernetworks = {}
settings_components = {}
restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_control_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}
resize_modes = ["None", "Fixed", "Crop", "Fill", "Outpaint", "Context aware"]
max_workers = 12
default_hfcache_dir = os.environ.get("SD_HFCACHEDIR", None) or os.path.join(paths.models_path, 'huggingface')
state = shared_state.State()
models_path = paths.models_path
script_path = paths.script_path
data_path = paths.data_path


# early select backend
backend = Backend.DIFFUSERS
if cmd_opts.use_openvino: # override for openvino
    os.environ.setdefault('PYTORCH_TRACING_MODE', 'TORCHFX')
    from modules.intel.openvino import get_device_list as get_openvino_device_list # pylint: disable=ungrouped-imports,unused-import
elif cmd_opts.use_ipex or devices.has_xpu():
    from modules.intel.ipex import ipex_init
    ok, e = ipex_init()
    if not ok:
        log.error(f'IPEX initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'IPEX')
elif cmd_opts.use_directml:
    from modules.dml import directml_init
    ok, e = directml_init()
    if not ok:
        log.error(f'DirectML initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'DirectML')
elif cmd_opts.use_rocm or devices.has_rocm():
    from modules.rocm import rocm_init
    ok, e = rocm_init()
    if not ok:
        log.error(f'ROCm initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'ROCm')
devices.backend = devices.get_backend(cmd_opts)
devices.device = devices.get_optimal_device()
mem_stat = memory_stats()
cpu_memory = round(mem_stat['ram']['total'] if "ram" in mem_stat else 0)
gpu_memory = round(mem_stat['gpu']['total'] if "gpu" in mem_stat else 0)
if gpu_memory == 0:
    gpu_memory = cpu_memory
native = backend == Backend.DIFFUSERS
if not files_cache.do_cache_folders:
    log.warning('File cache disabled: ')


def list_checkpoint_titles():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.checkpoint_titles()


list_checkpoint_tiles = list_checkpoint_titles # alias for legacy typo
default_sd_model_file = paths.default_sd_model_file
default_checkpoint = list_checkpoint_titles()[0] if len(list_checkpoint_titles()) > 0 else "model.safetensors"


def is_url(string):
    from urllib.parse import urlparse
    parsed_url = urlparse(string)
    return all([parsed_url.scheme, parsed_url.netloc])


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def refresh_vaes():
    import modules.sd_vae # pylint: disable=W0621
    modules.sd_vae.refresh_vae_list()


def refresh_upscalers():
    import modules.modelloader # pylint: disable=W0621
    modules.modelloader.load_upscalers()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers


log.debug('Initializing: default modes')
startup_offload_mode, startup_offload_min_gpu, startup_offload_max_gpu, startup_cross_attention, startup_sdp_options, startup_sdp_choices, startup_sdp_override_options, startup_sdp_override_choices, startup_offload_always, startup_offload_never = get_default_modes(cmd_opts=cmd_opts, mem_stat=mem_stat)

log.debug('Initializing: settings')
from modules import ui_definitions
from modules.ui_definitions import OptionInfo, options_section # pylint: disable=unused-import
options_templates = ui_definitions.create_settings(cmd_opts)
from modules.shared_legacy import get_legacy_options
options_templates.update(get_legacy_options())
from modules.options_handler import Options
config_filename = cmd_opts.config
secrets_filename = cmd_opts.secrets
opts = Options(options_templates, restricted=restricted_opts, filename=config_filename, secrets=secrets_filename)
cmd_opts = cmd_args.settings_args(opts, cmd_opts)
cmd_opts = cmd_args.override_args(opts, cmd_opts)
if cmd_opts.locale is not None:
    opts.data['ui_locale'] = cmd_opts.locale

log.debug('Initializing: backend')
from modules.dml import directml_do_hijack
if cmd_opts.use_xformers:
    opts.data['cross_attention_optimization'] = 'xFormers'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder # compatibility
opts.data['uni_pc_order'] = max(2, opts.schedulers_solver_order) # compatibility
log.info(f'Engine: backend={backend} compute={devices.backend} device={devices.get_optimal_device_name()} attention="{opts.cross_attention_optimization}" mode={devices.inference_context.__name__}')

profiler = None
import modules.styles
prompt_styles = modules.styles.StyleDatabase(opts)
reference_models = readfile(os.path.join('data', 'reference.json'), as_type="dict") if opts.extra_network_reference_enable else {}
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure

log.debug('Initializing: devices')
devices.args = cmd_opts
devices.opts = opts
devices.onnx = [opts.onnx_execution_provider]
devices.set_cuda_params()
if opts.onnx_cpu_fallback and 'CPUExecutionProvider' not in devices.onnx:
    devices.onnx.append('CPUExecutionProvider')
device = devices.device
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", devices.device)
history = history.History()
if devices.backend == "directml":
    directml_do_hijack()
log.debug('Quantization: registered=SDNQ')

try:
    log.info(f'Device: {print_dict(devices.get_gpu_info())}')
except Exception as ex:
    log.error(f'Device: {ex}')


def restart_server(restart=True):
    if demo is None:
        return
    log.critical('Server shutdown requested')
    try:
        sys.tracebacklimit = 0
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stdout(stderr):
            demo.server.wants_restart = restart
            demo.server.should_exit = True
            demo.server.force_exit = True
            demo.close(verbose=False)
            demo.server.close()
            demo.fns = []
        time.sleep(1)
        sys.tracebacklimit = 100
        # os._exit(0)
    except (Exception, BaseException) as err:
        log.error(f'Server shutdown error: {err}')
    if restart:
        log.info('Server will restart')


def restore_defaults(restart=True):
    if os.path.exists(cmd_opts.config):
        log.info('Restoring server defaults')
        os.remove(cmd_opts.config)
    restart_server(restart)


from modules.modeldata import Shared  # pylint: disable=ungrouped-imports

sys.modules[__name__].__class__ = Shared

if TYPE_CHECKING:
    # From Shared class
    sd_model: DiffusionPipeline | None
    sd_refiner: DiffusionPipeline | None
    sd_model_type: str
    sd_refiner_type: str
    sd_loaded: bool
