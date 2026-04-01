import os
import sys
import argparse
import shlex


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def get_argv():
    return shlex.split(" ".join(sys.argv[1:])) if "USED_VSCODE_COMMAND_PICKARGS" in os.environ else sys.argv[1:]


def add_core_args(p):
    p.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
    p.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
    p.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", None), help="Base path where all models are stored, default: %(default)s")
    p.add_argument("--embeddings-dir", type=str, default=os.environ.get("SD_EMBEDDINGSDIR", None), help="Base path where all embeddings are stored, default: %(default)s")
    p.add_argument("--vae-dir", type=str, default=os.environ.get("SD_VAEDIR", None), help="Base path where all VAEs are stored, default: %(default)s")
    p.add_argument("--lora-dir", type=str, default=os.environ.get("SD_LORADIR", None), help="Base path where all LoRAs are stored, default: %(default)s")
    p.add_argument("--extensions-dir", type=str, default=os.environ.get("SD_EXTENSIONSDIR", None), help="Base path where all extensions are stored, default: %(default)s")


def add_config_arg(p, data_dir):
    p.add_argument("--config", type=str, default=os.environ.get("SD_CONFIG", os.path.join(data_dir, 'config.json')), help="Use specific server configuration file, default: %(default)s")
    p.add_argument("--secrets", type=str, default=os.environ.get("SD_SECRETS", os.path.join(data_dir, 'secrets.json')), help="Use specific server secrets file, default: %(default)s")


def add_compute_args(p):
    p.add_argument("--device-id", type=str, default=os.environ.get("SD_DEVICEID", None), help="Select the default GPU device to use, default: %(default)s")
    p.add_argument("--use-cuda", default=env_flag("SD_USECUDA", False), action='store_true', help="Force use nVidia CUDA backend, default: %(default)s")
    p.add_argument("--use-ipex", default=env_flag("SD_USEIPEX", False), action='store_true', help="Force use Intel OneAPI XPU backend, default: %(default)s")
    p.add_argument("--use-rocm", default=env_flag("SD_USEROCM", False), action='store_true', help="Force use AMD ROCm backend, default: %(default)s")
    p.add_argument('--use-zluda', default=env_flag("SD_USEZLUDA", False), action='store_true', help="Force use ZLUDA, AMD GPUs only, default: %(default)s")
    p.add_argument("--use-openvino", default=env_flag("SD_USEOPENVINO", False), action='store_true', help="Use Intel OpenVINO backend, default: %(default)s")
    p.add_argument('--use-directml', default=env_flag("SD_USEDIRECTML", False), action='store_true', help="Use DirectML if no compatible GPU is detected, default: %(default)s")
    p.add_argument("--use-xformers", default=env_flag("SD_USEXFORMERS", False), action='store_true', help="Force use xFormers cross-optimization, default: %(default)s")
    p.add_argument("--use-nightly", default=env_flag("SD_USENIGHTLY", False), action='store_true', help="Force use nightly torch builds, default: %(default)s")
    p.add_argument("--no-half", default=env_flag("SD_NOHALF", False), action='store_true', help="Do not switch the model to 16-bit float, default: %(default)s")
    p.add_argument("--no-half-vae", default=env_flag("SD_NOHALFVAE", False), action='store_true', help="Do not switch VAE model to 16-bit float, default: %(default)s")


def add_ui_args(p):
    p.add_argument('--theme', type=str, default=os.environ.get("SD_THEME", None), help='Override UI theme')
    p.add_argument('--locale', type=str, default=os.environ.get("SD_LOCALE", None), help='Override UI locale')
    p.add_argument("--enso", default=env_flag("SD_ENSO", False), action="store_true", help="Enable Enso frontend, default: %(default)s")


def add_http_args(p):
    p.add_argument("--server-name", type=str, default=os.environ.get("SD_SERVERNAME", None), help="Sets hostname of server, default: %(default)s")
    p.add_argument("--tls-keyfile", type=str, default=os.environ.get("SD_TLSKEYFILE", None), help="Enable TLS and specify key file, default: %(default)s")
    p.add_argument("--tls-certfile", type=str, default=os.environ.get("SD_TLSCERTFILE", None), help="Enable TLS and specify cert file, default: %(default)s")
    p.add_argument("--tls-selfsign", action="store_true", default=env_flag("SD_TLSSELFSIGN", False), help="Enable TLS with self-signed certificates, default: %(default)s")
    p.add_argument("--cors-origins", type=str, default=os.environ.get("SD_CORSORIGINS", None), help="Allowed CORS origins as comma-separated list, default: %(default)s")
    p.add_argument("--cors-regex", type=str, default=os.environ.get("SD_CORSREGEX", None), help="Allowed CORS origins as regular expression, default: %(default)s")
    p.add_argument('--subpath', type=str, default=os.environ.get("SD_SUBPATH", None), help='Customize the URL subpath for usage with reverse proxy')
    p.add_argument("--autolaunch", default=env_flag("SD_AUTOLAUNCH", False), action='store_true', help="Open the UI URL in the system's default browser upon launch")
    p.add_argument("--auth", type=str, default=os.environ.get("SD_AUTH", None), help='Set access authentication like "user:pwd,user:pwd""')
    p.add_argument("--auth-file", type=str, default=os.environ.get("SD_AUTHFILE", None), help='Set access authentication using file, default: %(default)s')
    p.add_argument("--allowed-paths", nargs='+', default=[], type=str, required=False, help="add additional paths to paths allowed for web access")
    p.add_argument("--insecure", default=env_flag("SD_INSECURE", False), action='store_true', help="Enable extensions tab regardless of other options, default: %(default)s")
    p.add_argument("--listen", default=env_flag("SD_LISTEN", False), action='store_true', help="Launch web server using public IP address, default: %(default)s")
    p.add_argument("--port", type=int, default=os.environ.get("SD_PORT", 7860), help="Launch web server with given server port, default: %(default)s")


def add_diag_args(p):
    p.add_argument('--experimental', default=env_flag("SD_EXPERIMENTAL", False), action='store_true', help="Allow unsupported versions of libraries, default: %(default)s")
    p.add_argument('--ignore', default=env_flag("SD_IGNORE", False), action='store_true', help="Ignore any errors and attempt to continue")
    p.add_argument('--new', default=env_flag("SD_NEW", False), action='store_true', help="Force newer/untested version of libraries, default: %(default)s")
    p.add_argument('--safe', default=env_flag("SD_SAFE", False), action='store_true', help="Run in safe mode with no user extensions")
    p.add_argument('--test', default=env_flag("SD_TEST", False), action='store_true', help="Run test only and exit")
    p.add_argument('--version', default=False, action='store_true', help="Print version information")
    p.add_argument("--monitor", type=float, default=float(os.environ.get("SD_MONITOR", -1)), help="Run memory monitor, default: %(default)s")
    p.add_argument("--status", type=float, default=float(os.environ.get("SD_STATUS", -1)), help="Run server is-alive status, default: %(default)s")


def add_log_args(p):
    p.add_argument("--log", type=str, default=os.environ.get("SD_LOG", None), help="Set log file, default: %(default)s")
    p.add_argument('--debug', default=not env_flag("SD_NODEBUG", False), action='store_true', help="Run with debug logging, default: %(default)s")
    p.add_argument("--trace", default=env_flag("SD_TRACE", False), action='store_true', help="Run with trace logging, default: %(default)s")
    p.add_argument("--profile", default=env_flag("SD_PROFILE", False), action='store_true', help="Run profiler, default: %(default)s")
    p.add_argument('--docs', default=not env_flag("SD_NODOCS", False), action='store_true', help=argparse.SUPPRESS)
    p.add_argument("--api-log", default=not env_flag("SD_NOAPILOG", False), action='store_true', help=argparse.SUPPRESS)


parsed = None
parser = argparse.ArgumentParser(description="SD.Next", conflict_handler='resolve', epilog='For other options see UI Settings page', prog='', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))
parser._optionals = parser.add_argument_group('Other options') # pylint: disable=protected-access


def parse_args():
    global parsed # pylint: disable=global-statement
    if parsed is None:
        main_args()
        compatibility_args()
        parsed, _ = parser.parse_known_args()
    return parsed


def main_args():
    # main server args
    from modules.paths import data_path
    group_paths = parser.add_argument_group('Paths')
    add_core_args(group_paths)
    group_config = parser.add_argument_group('Configuration')
    add_config_arg(group_config, data_path)
    group_config.add_argument("--ui-config", type=str, default=os.environ.get("SD_UICONFIG", os.path.join(data_path, 'ui-config.json')), help="Use specific UI configuration file, default: %(default)s")
    group_config.add_argument("--freeze", default=os.environ.get("SD_FREEZE", False), action='store_true', help="Disable editing settings")
    group_config.add_argument("--medvram", default=os.environ.get("SD_MEDVRAM", False), action='store_true', help="Split model stages and keep only active part in VRAM, default: %(default)s")
    group_config.add_argument("--lowvram", default=os.environ.get("SD_LOWVRAM", False), action='store_true', help="Split model components and keep only active part in VRAM, default: %(default)s")
    group_config.add_argument("--disable", default=os.environ.get("SD_DISABLE", ''),  help="Disable specific UI tabs: %(default)s")

    group_compute = parser.add_argument_group('Compute Engine')
    add_compute_args(group_compute)

    group_ui = parser.add_argument_group('UI')
    add_ui_args(group_ui)

    group_http = parser.add_argument_group('HTTP')
    add_http_args(group_http)

    group_diag = parser.add_argument_group('Diagnostics')
    add_diag_args(group_diag)

    group_log = parser.add_argument_group('Logging')
    add_log_args(group_log)

    group_nargs = parser.add_argument_group('Other')
    group_nargs.add_argument('args', type=str, nargs='*', help=argparse.SUPPRESS)


def compatibility_args():
    # removed args are added here as hidden in fixed format for compatibility reasons
    from modules.paths import data_path
    group_compat = parser.add_argument_group('Compatibility options')
    group_compat.add_argument('--backend', type=str, choices=['diffusers', 'original'], help=argparse.SUPPRESS)
    group_compat.add_argument("--allow-code", default=env_flag("SD_ALLOWCODE", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--enable_insecure_extension_access", default=env_flag("SD_INSECURE", False), action='store_true', help=argparse.SUPPRESS)
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
    group_compat.add_argument("--disable-queue", default=env_flag("SD_DISABLEQUEUE", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--no-hashing", default=env_flag("SD_NOHASHING", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--no-metadata", default=env_flag("SD_NOMETADATA", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast", help=argparse.SUPPRESS)
    group_compat.add_argument("--upcast-sampling", default=env_flag("SD_UPCASTSAMPLING", False), action='store_true', help=argparse.SUPPRESS)
    group_compat.add_argument("--hypernetwork-dir", type=str, default=os.environ.get("SD_HYPERNETWORKDIR", None), help=argparse.SUPPRESS)
    group_compat.add_argument("--share", default=env_flag("SD_SHARE", False), action="store_true", help=argparse.SUPPRESS)


def settings_args(opts, args):
    # removed args are added here as hidden in fixed format for compatibility reasons
    from modules.paths import data_path
    group_compat = parser.add_argument_group('Compatibility options')
    group_compat.add_argument("--allow-code", default=env_flag("SD_ALLOWCODE", False), action='store_true', help=argparse.SUPPRESS)
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

    if "USED_VSCODE_COMMAND_PICKARGS" in os.environ:
        argv = get_argv()
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    for d in ['lora_dir', 'embeddings_dir', 'hypernetwork_dir', 'vae_dir', 'ckpt_dir', 'temp_dir', 'diffusers_dir', 'hfcache_dir', 'tunable_dir', 'unet_dir', 'te_dir', 'styles_dir', 'wildcards_dir', 'control_dir', 'yolo_dir', 'no_half', 'no_half_vae', 'precision', 'upcast_sampling', 'esrgan_models_path', 'bsrgan_models_path', 'realesrgan_models_path', 'scunet_models_path', 'swinir_models_path', 'ldsr_models_path', 'clip_models_path', 'onnx_cached_models_path']:
        if hasattr(opts, d):
            if getattr(args, d, None) is None:
                setattr(args, d, getattr(opts, d))
            opts.onchange(d, lambda d=d: setattr(args, d, getattr(opts, d)), call=False)

    return args


def override_args(opts, args):
    if opts.server_listen:
        args.listen = True
    if opts.server_status >= 0:
        args.status = opts.server_status
    if opts.server_monitor >= 0:
        args.monitor = opts.server_monitor
    return args
