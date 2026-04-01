import gradio as gr
import gradio.routes
import gradio.utils
from modules import errors, timer, gr_hijack, shared, script_callbacks, ui_common, ui_symbols, ui_javascript, ui_sections, generation_parameters_copypaste, call_queue, scripts_manager
from modules.logger import log
from modules.api import mime


errors.install()
mime.register()
gr_hijack.init()
switch_values_symbol = ui_symbols.switch
detect_image_size_symbol = ui_symbols.detect
paste_symbol = ui_symbols.paste
clear_prompt_symbol = ui_symbols.clear
restore_progress_symbol = ui_symbols.apply
folder_symbol = ui_symbols.folder
extra_networks_symbol = ui_symbols.networks
apply_style_symbol = ui_symbols.apply
save_style_symbol = ui_symbols.save
wrap_queued_call = call_queue.wrap_queued_call # compatibility item
wrap_gradio_call = call_queue.wrap_gradio_call # compatibility item
wrap_gradio_gpu_call = call_queue.wrap_gradio_gpu_call # compatibility item
plaintext_to_html = ui_common.plaintext_to_html # compatibility item
infotext_to_html = ui_common.infotext_to_html # compatibility item
create_sampler_and_steps_selection = ui_sections.create_sampler_and_steps_selection # compatibility item
ui_system_tabs = None # required for system-info
interfaces = []


if not shared.cmd_opts.share and not shared.cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def create_override_settings_dropdown(a, _b):
    return ui_common.create_override_inputs(a) # compatibility item


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def create_output_panel(tabname, outdir): # pylint: disable=unused-argument # outdir is used by extensions
    a, b, c, _d, e = ui_common.create_output_panel(tabname)
    return a, b, c, e


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return generation_parameters_copypaste.image_from_url_text(x[0])


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    return ui_common.create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)


def connect_clear_prompt(button): # pylint: disable=unused-argument
    pass


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def create_ui(startup_timer = None) -> gr.Blocks:
    global interfaces # pylint: disable=global-statement
    if startup_timer is None:
        timer.startup = timer.Timer()
    ui_javascript.reload_javascript()
    generation_parameters_copypaste.reset()
    scripts_manager.scripts_current = None
    if hasattr(shared.cmd_opts, 'disable'):
        ui_disabled = [x.strip().lower() for x in shared.cmd_opts.disable.split(',') if x.strip()]
    else:
        ui_disabled = []
    interfaces.clear()
    shared.opts.ui_disabled = ui_disabled
    if len(ui_disabled) > 0:
        log.warning(f'UI disabled: {ui_disabled}')

    if 'txt2img' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as txt2img_interface:
            from modules import ui_txt2img
            ui_txt2img.create_ui()
            timer.startup.record("ui-txt2img")
            interfaces += [(txt2img_interface, "Text", "txt2img")]

    if 'img2img' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as img2img_interface:
            from modules import ui_img2img
            ui_img2img.create_ui()
            timer.startup.record("ui-img2img")
            interfaces += [(img2img_interface, "Image", "img2img")]

    if 'control' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as control_interface:
            from modules import ui_control
            ui_control.create_ui()
            timer.startup.record("ui-control")
            interfaces += [(control_interface, "Control", "control")]

    if 'video' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as video_interface:
            from modules import ui_video
            ui_video.create_ui()
            timer.startup.record("ui-video")
            interfaces += [(video_interface, "Video", "video")]

    if 'extras' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as extras_interface:
            from modules import ui_postprocessing
            ui_postprocessing.create_ui()
            timer.startup.record("ui-extras")
            interfaces += [(extras_interface, "Process", "process")]

    if 'caption' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as caption_interface:
            from modules import ui_caption
            ui_caption.create_ui()
            timer.startup.record("ui-caption")
            interfaces += [(caption_interface, "Caption", "caption")]

    if 'models' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as models_interface:
            from modules import ui_models
            ui_models.create_ui()
            timer.startup.record("ui-models")
            interfaces += [(models_interface, "Models", "models")]

    if 'gallery' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as gallery_interface:
            from modules import ui_gallery
            ui_gallery.create_ui()
            timer.startup.record("ui-gallery")
            interfaces += [(gallery_interface, "Gallery", "gallery")]

    interfaces += script_callbacks.ui_tabs_callback()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        from modules import ui_settings
        ui_settings.create_ui(ui_disabled)
        global ui_system_tabs # pylint: disable=global-statement
        ui_system_tabs = ui_settings.ui_system_tabs
        shared.opts.reorder()
        timer.startup.record("ui-extensions")
        interfaces += [(settings_interface, "System", "system")]

    if 'info' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as info_interface:
            from modules import ui_docs
            ui_docs.create_ui()
            timer.startup.record("ui-info")
            interfaces += [(info_interface, "Info", "info")]

    if 'extensions' not in ui_disabled:
        with gr.Blocks(analytics_enabled=False) as extensions_interface:
            from modules import ui_extensions
            ui_extensions.create_ui()
            timer.startup.record("ui-extensions")
            interfaces += [(extensions_interface, "Extensions", "extensions")]

    ui_app = ui_settings.create_quicksettings(interfaces)

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    return ui_app
