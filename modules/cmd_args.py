import os
import sys
import argparse
from modules.paths import data_path, models_path


parsed = None
parser = argparse.ArgumentParser(description="SD.Next", conflict_handler='resolve', epilog='For other options see UI Settings page', prog='', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))
parser._optionals = parser.add_argument_group('Other options') # pylint: disable=protected-access


def parse_args():
    global parsed # pylint: disable=global-statement
    if parsed is None:
        parsed, _ = parser.parse_known_args()
    return parsed


def main_args():
    # main server args
    group_config = parser.add_argument_group('Configuration')
    group_config.add_argument("--config", type=str, default=os.environ.get("SD_CONFIG", os.path.join(data_path, 'config.json')), help="Use specific server configuration file, default: %(default)s")
    group_config.add_argument("--ui-config", type=str, default=os.environ.get("SD_UICONFIG", os.path.join(data_path, 'ui-config.json')), help="Use specific UI configuration file, default: %(default)s")
    group_config.add_argument("--medvram", default=os.environ.get("SD_MEDVRAM", False), action='store_true', help="Split model stages and keep only active part in VRAM, default: %(default)s")
    group_config.add_argument("--lowvram", default=os.environ.get("SD_LOWVRAM", False), action='store_true', help="Split model components and keep only active part in VRAM, default: %(default)s")
    group_config.add_argument("--freeze", default=os.environ.get("SD_FREEZE", False), action='store_true', help="Disable editing settings")

    group_compute = parser.add_argument_group('Compute Engine')
    group_compute.add_argument("--device-id", type=str, default=os.environ.get("SD_DEVICEID", None), help="Select the default CUDA device to use, default: %(default)s")
    group_compute.add_argument('--use-directml', default=os.environ.get("SD_USEDIRECTML", False), action='store_true', help = "Use DirectML if no compatible GPU is detected, default: %(default)s")
    group_compute.add_argument('--use-zluda', default=os.environ.get("SD_USEZLUDA", False), action='store_true', help = "Force use ZLUDA, AMD GPUs only, default: %(default)s")
    group_compute.add_argument("--use-openvino", default=os.environ.get("SD_USEOPENVINO", False), action='store_true', help="Use Intel OpenVINO backend, default: %(default)s")
    group_compute.add_argument("--use-ipex", default=os.environ.get("SD_USEIPX", False), action='store_true', help="Force use Intel OneAPI XPU backend, default: %(default)s")
    group_compute.add_argument("--use-cuda", default=os.environ.get("SD_USECUDA", False), action='store_true', help="Force use nVidia CUDA backend, default: %(default)s")
    group_compute.add_argument("--use-rocm", default=os.environ.get("SD_USEROCM", False), action='store_true', help="Force use AMD ROCm backend, default: %(default)s")

    group_paths = parser.add_argument_group('Paths')
    group_paths.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
    group_paths.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
    group_paths.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", 'models'), help="Base path where all models are stored, default: %(default)s",)
    group_paths.add_argument("--extensions-dir", type=str, default=os.environ.get("SD_EXTENSIONSDIR", None), help="Base path where all extensions are stored, default: %(default)s",)

    group_diag = parser.add_argument_group('Diagnostics')
    group_diag.add_argument("--no-hashing", default=os.environ.get("SD_NOHASHING", False), action='store_true', help="Disable hashing of checkpoints, default: %(default)s")
    group_diag.add_argument("--no-metadata", default=os.environ.get("SD_NOMETADATA", False), action='store_true', help="Disable reading of metadata from models, default: %(default)s")
    group_diag.add_argument("--profile", default=os.environ.get("SD_PROFILE", False), action='store_true', help="Run profiler, default: %(default)s")
    group_diag.add_argument("--monitor", default=os.environ.get("SD_MONITOR", 0), help="Run memory monitor, default: %(default)s")
    group_diag.add_argument("--status", default=os.environ.get("SD_STATUS", 120), help="Run server is-alive status, default: %(default)s")
    group_diag.add_argument('--experimental', default=os.environ.get("SD_EXPERIMENTAL",False), action='store_true', help="Allow unsupported versions of libraries, default: %(default)s")

    group_http = parser.add_argument_group('HTTP')
    group_http.add_argument('--theme', type=str, default=os.environ.get("SD_THEME", None), help='Override UI theme')
    group_http.add_argument('--locale', type=str, default=os.environ.get("SD_LOCALE", None), help='Override UI locale')
    group_http.add_argument("--server-name", type=str, default=os.environ.get("SD_SERVERNAME", None), help="Sets hostname of server, default: %(default)s")
    group_http.add_argument("--tls-keyfile", type=str, default=os.environ.get("SD_TLSKEYFILE", None), help="Enable TLS and specify key file, default: %(default)s")
    group_http.add_argument("--tls-certfile", type=str, default=os.environ.get("SD_TLSCERTFILE", None), help="Enable TLS and specify cert file, default: %(default)s")
    group_http.add_argument("--tls-selfsign", action="store_true", default=os.environ.get("SD_TLSSELFSIGN", False), help="Enable TLS with self-signed certificates, default: %(default)s")
    group_http.add_argument("--cors-origins", type=str, default=os.environ.get("SD_CORSORIGINS", None), help="Allowed CORS origins as comma-separated list, default: %(default)s")
    group_http.add_argument("--cors-regex", type=str, default=os.environ.get("SD_CORSREGEX", None), help="Allowed CORS origins as regular expression, default: %(default)s")
    group_http.add_argument('--subpath', type=str, default=os.environ.get("SD_SUBPATH", None), help='Customize the URL subpath for usage with reverse proxy')
    group_http.add_argument("--autolaunch", default=os.environ.get("SD_AUTOLAUNCH", False), action='store_true', help="Open the UI URL in the system's default browser upon launch")
    group_http.add_argument('--docs', default=os.environ.get("SD_DOCS", False), action='store_true', help = "Mount API docs, default: %(default)s")
    group_http.add_argument("--auth", type=str, default=os.environ.get("SD_AUTH", None), help='Set access authentication like "user:pwd,user:pwd""')
    group_http.add_argument("--auth-file", type=str, default=os.environ.get("SD_AUTHFILE", None), help='Set access authentication using file, default: %(default)s')
    group_http.add_argument("--allowed-paths", nargs='+', default=[], type=str, required=False, help="add additional paths to paths allowed for web access")
    group_http.add_argument("--share", default=os.environ.get("SD_SHARE", False), action='store_true', help="Enable UI accessible through Gradio site, default: %(default)s")
    group_http.add_argument("--insecure", default=os.environ.get("SD_INSECURE", False), action='store_true', help="Enable extensions tab regardless of other options, default: %(default)s")
    group_http.add_argument("--listen", default=os.environ.get("SD_LISTEN", False), action='store_true', help="Launch web server using public IP address, default: %(default)s")
    group_http.add_argument("--port", type=int, default=os.environ.get("SD_PORT", 7860), help="Launch web server with given server port, default: %(default)s")


def compatibility_args():
    # removed args are added here as hidden in fixed format for compatbility reasons
    group_compat = parser.add_argument_group('Compatibility options')
    group_compat.add_argument('--backend', type=str, choices=['diffusers', 'original'], help=argparse.SUPPRESS)
    group_compat.add_argument('--hypernetwork-dir', default=os.path.join(models_path, 'hypernetworks'), help=argparse.SUPPRESS)
    group_compat.add_argument("--allow-code", default=os.environ.get("SD_ALLOWCODE", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--enable_insecure_extension_access", default=os.environ.get("SD_INSECURE", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--use-cpu", nargs='+', default=[], type=str.lower, help=argparse.SUPPRESS)
    group_compat.add_argument("-f", action='store_true', help=argparse.SUPPRESS)  # allows running as root; implemented outside of webui
    group_compat.add_argument('--vae', type=str, default=os.environ.get("SD_VAE", None), help=argparse.SUPPRESS)
    group_compat.add_argument("--ui-settings-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'config.json'))
    group_compat.add_argument("--ui-config-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'ui-config.json'))
    group_compat.add_argument("--hide-ui-dir-config", action='store_true', help=argparse.SUPPRESS, default=False)
    group_compat.add_argument("--disable-console-progressbars", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--disable-safe-unpickle", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--lowram", action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--disable-extension-access", default=False, action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--api", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--api-auth", type=str, help=argparse.SUPPRESS, default=None)
    group_compat.add_argument('--api-only', default=False, help=argparse.SUPPRESS)
    group_compat.add_argument("--disable-queue", default=os.environ.get("SD_DISABLEQUEUE", False), action='store_true', help=argparse.SUPPRESS)


def settings_args(opts, args):
    # removed args are added here as hidden in fixed format for compatbility reasons
    group_compat = parser.add_argument_group('Compatibility options')
    group_compat.add_argument("--allow-code", default=os.environ.get("SD_ALLOWCODE", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--use-cpu", nargs='+', default=[], type=str.lower, help=argparse.SUPPRESS)
    group_compat.add_argument("-f", action='store_true', help=argparse.SUPPRESS)  # allows running as root; implemented outside of webui
    group_compat.add_argument('--vae', type=str, default=os.environ.get("SD_VAE", None), help=argparse.SUPPRESS)
    group_compat.add_argument("--ui-settings-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'config.json'))
    group_compat.add_argument("--ui-config-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'ui-config.json'))
    group_compat.add_argument("--hide-ui-dir-config", action='store_true', help=argparse.SUPPRESS, default=False)
    group_compat.add_argument("--disable-console-progressbars", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--disable-safe-unpickle", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--lowram", action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--disable-extension-access", default=False, action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--allowed-paths", nargs='+', default=[], type=str, required=False, help="add additional paths to paths allowed for web access")
    group_compat.add_argument("--api", action='store_true', help=argparse.SUPPRESS, default=True)
    group_compat.add_argument("--api-auth", type=str, help=argparse.SUPPRESS, default=None)
    # removed args that have been moved to opts are added here as hidden with default values as defined in opts
    group_compat.add_argument("--ckpt-dir", type=str, help=argparse.SUPPRESS, default=opts.ckpt_dir)
    group_compat.add_argument("--vae-dir", type=str, help=argparse.SUPPRESS, default=opts.vae_dir)
    group_compat.add_argument("--embeddings-dir", type=str, help=argparse.SUPPRESS, default=opts.embeddings_dir)
    group_compat.add_argument("--embeddings-templates-dir", type=str, help=argparse.SUPPRESS, default=opts.embeddings_templates_dir)
    group_compat.add_argument("--codeformer-models-path", type=str, help=argparse.SUPPRESS, default=opts.codeformer_models_path)
    group_compat.add_argument("--gfpgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.gfpgan_models_path)
    group_compat.add_argument("--esrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.esrgan_models_path)
    group_compat.add_argument("--bsrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.bsrgan_models_path)
    group_compat.add_argument("--realesrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.realesrgan_models_path)
    group_compat.add_argument("--scunet-models-path", help=argparse.SUPPRESS, default=opts.scunet_models_path)
    group_compat.add_argument("--swinir-models-path", help=argparse.SUPPRESS, default=opts.swinir_models_path)
    group_compat.add_argument("--ldsr-models-path", help=argparse.SUPPRESS, default=opts.ldsr_models_path)
    group_compat.add_argument("--clip-models-path", type=str, help=argparse.SUPPRESS, default=opts.clip_models_path)
    group_compat.add_argument("--opt-channelslast", help=argparse.SUPPRESS, action='store_true', default=opts.opt_channelslast)
    group_compat.add_argument("--xformers", default=(opts.cross_attention_optimization == "xFormers"), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--disable-nan-check", help=argparse.SUPPRESS, action='store_true', default=opts.disable_nan_check)
    group_compat.add_argument("--rollback-vae", help=argparse.SUPPRESS, default=opts.rollback_vae)
    group_compat.add_argument("--no-half", help=argparse.SUPPRESS, action='store_true', default=opts.no_half)
    group_compat.add_argument("--no-half-vae", help=argparse.SUPPRESS, action='store_true', default=opts.no_half_vae)
    group_compat.add_argument("--precision", help=argparse.SUPPRESS, default=opts.precision)
    group_compat.add_argument("--sub-quad-q-chunk-size", help=argparse.SUPPRESS, default=opts.sub_quad_q_chunk_size)
    group_compat.add_argument("--sub-quad-kv-chunk-size", help=argparse.SUPPRESS, default=opts.sub_quad_kv_chunk_size)
    group_compat.add_argument("--sub-quad-chunk-threshold", help=argparse.SUPPRESS, default=opts.sub_quad_chunk_threshold)
    group_compat.add_argument("--lora-dir", help=argparse.SUPPRESS, default=opts.lora_dir)
    group_compat.add_argument("--embeddings-dir", help=argparse.SUPPRESS, default=opts.embeddings_dir)
    group_compat.add_argument("--enable-console-prompts", help=argparse.SUPPRESS, action='store_true', default=False)
    group_compat.add_argument("--safe", help=argparse.SUPPRESS, action='store_true', default=False)
    group_compat.add_argument("--use-xformers", help=argparse.SUPPRESS, action='store_true', default=False)

    # removed opts are added here with fixed values for compatibility reasons
    opts.use_old_emphasis_implementation = False
    opts.use_old_karras_scheduler_sigmas = False
    opts.no_dpmpp_sde_batch_determinism = False
    opts.lora_apply_to_outputs = False
    opts.do_not_show_images = False
    opts.add_model_hash_to_info = True
    opts.add_model_name_to_info = True
    opts.js_modal_lightbox = True
    opts.js_modal_lightbox_initially_zoomed = True
    opts.show_progress_in_title = False
    opts.sd_vae_as_default = True
    opts.enable_emphasis = True
    opts.enable_batch_seeds = True
    # opts.multiple_tqdm = False
    opts.print_hypernet_extra = False
    opts.dimensions_and_batch_together = True
    opts.enable_pnginfo = True
    opts.data['clip_skip'] = 1

    opts.onchange("lora_dir", lambda: setattr(args, "lora_dir", opts.lora_dir))

    if "USED_VSCODE_COMMAND_PICKARGS" in os.environ:
        import shlex
        argv = shlex.split(" ".join(sys.argv[1:])) if "USED_VSCODE_COMMAND_PICKARGS" in os.environ else sys.argv[1:]
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()
    return args


main_args()
compatibility_args()
