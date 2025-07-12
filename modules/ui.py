import mimetypes
import gradio as gr
import gradio.routes
import gradio.utils
from modules import errors, timer, gr_hijack, shared, script_callbacks, ui_common, ui_symbols, ui_javascript, ui_sections, generation_parameters_copypaste, call_queue, scripts_manager
from modules.paths import script_path, data_path # pylint: disable=unused-import


errors.install()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/jxl', '.jxl')
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


def create_ui(startup_timer = None):
    if startup_timer is None:
        timer.startup = timer.Timer()
    ui_javascript.reload_javascript()
    generation_parameters_copypaste.reset()
    scripts_manager.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        from modules import ui_txt2img
        ui_txt2img.create_ui()
        timer.startup.record("ui-txt2img")

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        from modules import ui_img2img
        ui_img2img.create_ui()
        timer.startup.record("ui-img2img")

    with gr.Blocks(analytics_enabled=False) as control_interface:
        from modules import ui_control
        ui_control.create_ui()
        timer.startup.record("ui-control")

    with gr.Blocks(analytics_enabled=False) as video_interface:
        from modules import ui_video
        ui_video.create_ui()
        timer.startup.record("ui-video")

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        from modules import ui_postprocessing
        ui_postprocessing.create_ui()
        timer.startup.record("ui-extras")

    with gr.Blocks(analytics_enabled=False) as caption_interface:
        from modules import ui_caption
        ui_caption.create_ui()
        timer.startup.record("ui-caption")

    with gr.Blocks(analytics_enabled=False) as models_interface:
        from modules import ui_models
        ui_models.create_ui()
        timer.startup.record("ui-models")

    with gr.Blocks(analytics_enabled=False) as gallery_interface:
        from modules import ui_gallery
        ui_gallery.create_ui()
        timer.startup.record("ui-gallery")

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        from modules import ui_settings
        ui_settings.create_ui()
        global ui_system_tabs # pylint: disable=global-statement
        ui_system_tabs = ui_settings.ui_system_tabs
        shared.opts.reorder()
        timer.startup.record("ui-extensions")

    with gr.Blocks(analytics_enabled=False) as info_interface:
        with gr.Tabs(elem_id="tabs_info"):
            with gr.TabItem("Change log", id="change_log", elem_id="system_tab_changelog"):
                from modules import ui_docs
                ui_docs.create_ui_logs()

            with gr.TabItem("Wiki", id="wiki", elem_id="system_tab_wiki"):
                from modules import ui_docs
                ui_docs.create_ui_wiki()

    with gr.Blocks(analytics_enabled=False) as extensions_interface:
        from modules import ui_extensions
        ui_extensions.create_ui()
        timer.startup.record("ui-extensions")

    interfaces = []
    interfaces += [(txt2img_interface, "Text", "txt2img")]
    interfaces += [(img2img_interface, "Image", "img2img")]
    if control_interface is not None:
        interfaces += [(control_interface, "Control", "control")]
    if video_interface is not None:
        interfaces += [(video_interface, "Video", "video")]
    interfaces += [(extras_interface, "Process", "process")]
    interfaces += [(caption_interface, "Caption", "caption")]
    interfaces += [(gallery_interface, "Gallery", "gallery")]
    interfaces += [(models_interface, "Models", "models")]
    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "System", "system")]
    interfaces += [(info_interface, "Info", "info")]
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    ui_app = ui_settings.create_quicksettings(interfaces)

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    return ui_app
