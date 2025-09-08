import io
import os
import sys
import time
import glob
import signal
import asyncio
import logging
import importlib
import contextlib
from threading import Thread
import modules.loader
import modules.hashes

from installer import log, git_commit, custom_excepthook
from modules import timer, paths, shared, extensions, gr_tempdir, modelloader, modeldata
from modules.call_queue import queue_lock, wrap_queued_call, wrap_gradio_gpu_call # pylint: disable=unused-import
import modules.devices
import modules.sd_checkpoint
import modules.sd_samplers
import modules.scripts_manager
import modules.scripts
import modules.sd_models
import modules.sd_vae
import modules.sd_unet
import modules.sd_hijack
import modules.model_te
import modules.progress
import modules.ui
import modules.txt2img
import modules.img2img
import modules.upscaler
import modules.upscaler_simple
import modules.extra_networks
import modules.ui_extra_networks
import modules.textual_inversion
import modules.script_callbacks
import modules.api.middleware


if not modules.loader.initialized:
    timer.startup.record("libraries")
modules.loader.initialized = True


sys.excepthook = custom_excepthook
local_url = None
state = shared.state
backend = shared.backend
if shared.cmd_opts.server_name:
    server_name = shared.cmd_opts.server_name
else:
    server_name = "0.0.0.0" if shared.cmd_opts.listen else None
fastapi_args = {
    "version": f'0.0.{git_commit}',
    "title": "SD.Next",
    "description": "SD.Next",
    "docs_url": None,
    "redoc_url": None,
    # "docs_url": "/docs" if cmd_opts.docs else None, # custom handler in api.py
    # "redoc_url": "/redocs" if cmd_opts.docs else None,
}


def initialize():
    log.debug('Initializing')

    modules.sd_checkpoint.init_metadata()
    modules.hashes.init_cache()

    modules.sd_samplers.list_samplers()
    timer.startup.record("samplers")

    modules.sd_vae.refresh_vae_list()
    timer.startup.record("vae")

    modules.sd_unet.refresh_unet_list()
    timer.startup.record("unet")

    modules.model_te.refresh_te_list()
    timer.startup.record("te")

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    timer.startup.record("models")

    from modules.lora import lora_load
    lora_load.list_available_networks()
    timer.startup.record("lora")

    shared.prompt_styles.reload()
    timer.startup.record("styles")

    import modules.postprocess.codeformer_model as codeformer
    codeformer.setup_model(shared.opts.codeformer_models_path)
    sys.modules["modules.codeformer_model"] = codeformer
    import modules.postprocess.gfpgan_model as gfpgan
    gfpgan.setup_model(shared.opts.gfpgan_models_path)
    import modules.postprocess.yolo as yolo
    yolo.initialize()
    timer.startup.record("detailer")

    extensions.list_extensions()
    timer.startup.record("extensions")

    log.info('Load extensions')
    t_timer, t_total = modules.scripts_manager.load_scripts()
    modules.scripts.register_runners()
    timer.startup.record("extensions")
    timer.startup.records["extensions"] = t_total # scripts can reset the time
    log.debug(f'Extensions init time: {t_timer.summary()}')

    modelloader.load_upscalers()
    timer.startup.record("upscalers")

    modules.ui_extra_networks.initialize()
    modules.ui_extra_networks.register_pages()
    modules.extra_networks.initialize()
    modules.extra_networks.register_default_extra_networks()
    timer.startup.record("networks")

    from modules.models_hf import hf_init, hf_check_cache
    hf_init()
    hf_check_cache()

    if shared.cmd_opts.tls_keyfile is not None and shared.cmd_opts.tls_certfile is not None:
        try:
            if not os.path.exists(shared.cmd_opts.tls_keyfile):
                log.error("Invalid path to TLS keyfile given")
            if not os.path.exists(shared.cmd_opts.tls_certfile):
                log.error(f"Invalid path to TLS certfile: '{shared.cmd_opts.tls_certfile}'")
        except TypeError:
            shared.cmd_opts.tls_keyfile = shared.cmd_opts.tls_certfile = None
            log.error("TLS setup invalid, running webui without TLS")
        else:
            log.info("Running with TLS")
        timer.startup.record("tls")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(_sig, _frame):
        log.trace(f'State history: uptime={round(time.time() - shared.state.server_start)} jobs={len(shared.state.job_history)} tasks={len(shared.state.task_history)} latents={shared.state.latent_history} images={shared.state.image_history}')
        log.info('Exiting')
        try:
            for f in glob.glob("*.lock"):
                os.remove(f)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def load_model():
    modeldata.model_data.locked = False
    if not shared.opts.sd_checkpoint_autoload and shared.cmd_opts.ckpt is None:
        log.info('Model: autoload=False')
    else:
        shared.state.begin('Load')
        thread_model = Thread(target=lambda: shared.sd_model)
        thread_model.start()
        thread_refiner = Thread(target=lambda: shared.sd_refiner)
        thread_refiner.start()
        thread_model.join()
        thread_refiner.join()
        shared.state.end()
    timer.startup.record("checkpoint")
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(op='model')), call=False)
    shared.opts.onchange("sd_model_refiner", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(op='refiner')), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_unet", wrap_queued_call(lambda: modules.sd_unet.load_unet(shared.sd_model)), call=False)
    shared.opts.onchange("sd_text_encoder", wrap_queued_call(lambda: modules.sd_models.reload_text_encoder()), call=False)
    shared.opts.onchange("temp_dir", gr_tempdir.on_tmpdir_changed)
    timer.startup.record("onchange")


def create_api(app):
    log.debug('API initialize')
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def async_policy():
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy") else asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):
        def handle_exception(self, context):
            msg = context.get("exception", context["message"])
            log.error(f"AsyncIO loop: {msg}")

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                self.loop = super().get_event_loop()
            except (RuntimeError, AssertionError):
                self.loop = self.new_event_loop()
                self.set_event_loop(self.loop)
            return self.loop

        def __init__(self):
            super().__init__()
            self.loop = self.get_event_loop()
            self.loop.set_exception_handler(self.handle_exception)
            # log.debug(f"Event loop: {self.loop}")

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def get_external_ip():
    import socket
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        if ip_address.startswith('127.'):
            return None
        return ip_address
    except Exception:
        return None


def get_remote_ip():
    import requests
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=2)
        ip_address = response.json()['ip']
        return ip_address
    except Exception:
        return None


def start_common():
    log.debug('Entering start sequence')
    if shared.cmd_opts.data_dir is not None and len(shared.cmd_opts.data_dir) > 0:
        log.info(f'Base path: data="{shared.cmd_opts.data_dir}"')
    if shared.cmd_opts.models_dir is not None and len(shared.cmd_opts.models_dir) > 0 and shared.cmd_opts.models_dir != 'models':
        log.info(f'Base path: models="{shared.cmd_opts.models_dir}"')
    paths.create_paths(shared.opts)
    async_policy()
    initialize()
    if shared.cmd_opts.backend == 'original':
        shared.log.error('Legacy option: backend=original is no longer supported')
        shared.cmd_opts.backend = 'diffusers'
    try:
        from installer import diffusers_commit
        if diffusers_commit != 'unknown':
            shared.opts.diffusers_version = diffusers_commit # update installed diffusers version
    except Exception:
        pass
    if shared.opts.clean_temp_dir_at_start:
        gr_tempdir.cleanup_tmpdr()
        timer.startup.record("cleanup")


def mount_subpath(app):
    if shared.cmd_opts.subpath:
        shared.opts.subpath = shared.cmd_opts.subpath
    if shared.opts.subpath is None or len(shared.opts.subpath) == 0:
        return
    import gradio
    if not shared.opts.subpath.startswith('/'):
        shared.opts.subpath = f'/{shared.opts.subpath}'
    gradio.mount_gradio_app(app, shared.demo, path=shared.opts.subpath)
    shared.log.info(f'Mounted: subpath="{shared.opts.subpath}"')


def start_ui():
    log.debug('UI start sequence')
    modules.script_callbacks.before_ui_callback()
    timer.startup.record("before-ui")
    shared.demo = modules.ui.create_ui(timer.startup)
    timer.startup.record("ui")
    if shared.cmd_opts.disable_queue:
        log.info('Server queues disabled')
        shared.demo.progress_tracking = False
    else:
        shared.demo.queue(concurrency_count=64)

    gradio_auth_creds = []
    if shared.cmd_opts.auth:
        gradio_auth_creds += [x.strip() for x in shared.cmd_opts.auth.strip('"').replace('\n', '').split(',') if x.strip()]
    if shared.cmd_opts.auth_file:
        if not os.path.exists(shared.cmd_opts.auth_file):
            log.error(f"Invalid path to auth file: '{shared.cmd_opts.auth_file}'")
        else:
            with open(shared.cmd_opts.auth_file, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
    if len(gradio_auth_creds) > 0:
        log.info(f'Authentication enabled: users={len(list(gradio_auth_creds))}')

    global local_url # pylint: disable=global-statement
    stdout = io.StringIO()
    allowed_paths = [os.path.dirname(__file__)]
    if shared.cmd_opts.data_dir is not None and os.path.isdir(shared.cmd_opts.data_dir):
        allowed_paths.append(shared.cmd_opts.data_dir)
    if shared.cmd_opts.models_dir is not None and os.path.isdir(shared.cmd_opts.models_dir):
        allowed_paths.append(shared.cmd_opts.models_dir)
    if shared.cmd_opts.allowed_paths is not None:
        allowed_paths += [p for p in shared.cmd_opts.allowed_paths if os.path.isdir(p)]
    shared.log.debug(f'Root paths: {allowed_paths}')
    with contextlib.redirect_stdout(stdout):
        app, local_url, share_url = shared.demo.launch( # app is FastAPI(Starlette) instance
            share=shared.cmd_opts.share,
            server_name=server_name,
            server_port=shared.cmd_opts.port if shared.cmd_opts.port != 7860 else None,
            ssl_keyfile=shared.cmd_opts.tls_keyfile,
            ssl_certfile=shared.cmd_opts.tls_certfile,
            ssl_verify=not shared.cmd_opts.tls_selfsign,
            debug=False,
            auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
            prevent_thread_lock=True,
            max_threads=64,
            show_api=False,
            quiet=True,
            favicon_path='html/favicon.svg',
            allowed_paths=allowed_paths,
            app_kwargs=fastapi_args,
            _frontend=True and shared.cmd_opts.share,
        )
    if shared.cmd_opts.data_dir is not None:
        gr_tempdir.register_tmp_file(shared.demo, os.path.join(shared.cmd_opts.data_dir, 'x'))
    shared.log.info(f'Local URL: {local_url}')
    if shared.cmd_opts.listen:
        if not gradio_auth_creds:
            shared.log.warning('Public URL: enabled without authentication')
        if shared.cmd_opts.insecure:
            shared.log.warning('Public URL: enabled with insecure flag')
        proto = 'https' if shared.cmd_opts.tls_keyfile is not None else 'http'
        external_ip = get_external_ip()
        if external_ip is not None:
            shared.log.info(f'External URL: {proto}://{external_ip}:{shared.cmd_opts.port}')
        public_ip = get_remote_ip()
        if public_ip is not None:
            shared.log.info(f'Public URL: {proto}://{public_ip}:{shared.cmd_opts.port}')
    if shared.cmd_opts.docs:
        shared.log.info(f'API docs: {local_url[:-1]}/docs') # pylint: disable=unsubscriptable-object
        shared.log.info(f'API redocs: {local_url[:-1]}/redocs') # pylint: disable=unsubscriptable-object
    if share_url is not None:
        shared.log.info(f'Share URL: {share_url}')
    # shared.log.debug(f'Gradio functions: registered={len(shared.demo.fns)}')
    shared.demo.server.wants_restart = False
    modules.api.middleware.setup_middleware(app, shared.cmd_opts)

    timer.startup.record("launch")

    shared.api = create_api(app)
    shared.api.register()
    modules.progress.setup_progress_api()
    modules.ui_extra_networks.init_api()
    timer.startup.record("api")

    modules.script_callbacks.app_started_callback(shared.demo, app)
    timer.startup.record("app-started")

    time_sorted = sorted(modules.scripts_manager.time_setup.items(), key=lambda x: x[1], reverse=True)
    time_script = [f'{k}:{round(v,3)}' for (k,v) in time_sorted if v > 0.03]
    time_total = sum(modules.scripts_manager.time_setup.values())
    shared.log.debug(f'Scripts setup: time={time_total:.3f} {time_script}')
    time_component = [f'{k}:{round(v,3)}' for (k,v) in modules.scripts_manager.time_component.items() if v > 0.005]
    if len(time_component) > 0:
        shared.log.debug(f'Scripts components: {time_component}')
    return app


def webui(restart=False):
    if restart:
        modules.script_callbacks.app_reload_callback()
        modules.script_callbacks.script_unloaded_callback()

    start_common()
    app = start_ui()
    modules.script_callbacks.after_ui_callback()
    modules.sd_models.write_metadata()

    load_model()
    mount_subpath(app)
    shared.opts.save(shared.config_filename)

    if shared.cmd_opts.profile:
        for k, v in modules.script_callbacks.callback_map.items():
            shared.log.debug(f'Registered callbacks: {k}={len(v)} {[c.script for c in v]}')
    debug = log.trace if os.environ.get('SD_SCRIPT_DEBUG', None) is not None else lambda *args, **kwargs: None
    debug('Trace: SCRIPTS')
    for m in modules.scripts_manager.scripts_data:
        debug(f'  {m}')
    debug('Loaded postprocessing scripts:')
    for m in modules.scripts_manager.postprocessing_scripts_data:
        debug(f'  {m}')
    modules.script_callbacks.print_timers()

    if shared.cmd_opts.profile:
        log.info(f"Launch time: {timer.launch.summary(min_time=0)}")
        log.info(f"Installer time: {timer.init.summary(min_time=0)}")
        log.info(f"Startup time: {timer.startup.summary(min_time=0)}")
    else:
        timer.startup.add('launch', timer.launch.get_total())
        timer.startup.add('installer', timer.launch.get_total())
        log.info(f"Startup time: {timer.startup.summary()}")
    timer.startup.reset()

    if not restart:
        # override all loggers to use the same handlers as the main logger
        for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]: # pylint: disable=no-member
            if logger.name.startswith('uvicorn') or logger.name.startswith('sd'):
                continue
            logger.handlers = log.handlers
        # autolaunch only on initial start
        if (shared.opts.autolaunch or shared.cmd_opts.autolaunch) and local_url is not None:
            shared.cmd_opts.autolaunch = False
            shared.log.info('Launching browser')
            import webbrowser
            webbrowser.open(local_url, new=2, autoraise=True)
    else:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)

    return shared.demo.server


if __name__ == "__main__":
    webui()
