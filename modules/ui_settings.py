import os
import gradio as gr
from modules import timer, shared, paths, theme, sd_models, modelloader, generation_parameters_copypaste, call_queue, script_callbacks
from modules import ui_common, ui_loadsave, ui_history, ui_components, ui_symbols


text_settings = None # holds json of entire shared.opts
ui_system_tabs = None # required for system-info
dummy_component = gr.Textbox(visible=False, value='dummy')
loadsave = ui_loadsave.UiLoadsave(shared.cmd_opts.ui_config)
quicksettings_names = {x: i for i, x in enumerate(shared.opts.quicksettings_list) if x != 'quicksettings'}
quicksettings_list = []
hidden_list = []
components = []


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if shared.cmd_opts.freeze:
        return gr.update()
    if key == 'sd_backend':
        return gr.update()
    if key in shared.opts.disable_apply_metadata:
        gr.update()
    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)
        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()
    comp_args = shared.opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return gr.update()
    valtype = type(shared.opts.data_labels[key].default)
    oldval = shared.opts.data.get(key, None)
    shared.opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and shared.opts.data_labels[key].onchange is not None:
        shared.opts.data_labels[key].onchange()
    shared.opts.save(shared.config_filename)
    return getattr(shared.opts, key)


def get_value_for_setting(key):
    value = getattr(shared.opts, key)
    info = shared.opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision', 'multiselect', 'visible'}}
    return gr.update(value=value, **args)


def create_setting_component(key, is_quicksettings=False):
    def fun():
        return shared.opts.data[key] if key in shared.opts.data else shared.opts.data_labels[key].default

    info = shared.opts.data_labels[key]
    t = type(info.default)
    args = (info.component_args() if callable(info.component_args) else info.component_args) or {}
    if info.component is not None:
        comp = info.component
    elif t == str:
        comp = gr.Textbox
    elif t == int:
        comp = gr.Number
    elif t == bool:
        comp = gr.Checkbox
    else:
        raise ValueError(f'bad options item type: {t} for key {key}')
    elem_id = f"setting_{key}"
    dirty_indicator = None

    if not is_quicksettings:
        dirtyable_setting = gr.Group(elem_classes="dirtyable", visible=args.get("visible", True))
        dirtyable_setting.__enter__()
        dirty_indicator = gr.Button("", elem_classes="modification-indicator", elem_id=f"modification_indicator_{key}")

    if info.refresh is not None:
        if is_quicksettings:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
            ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        else:
            with gr.Row():
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
    elif info.folder is not None:
        with gr.Row():
            res = comp(label=info.label, value=fun(), elem_id=elem_id, elem_classes="folder-selector", **args)
    else:
        try:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
        except Exception as e:
            shared.log.error(f'Error creating setting: {key} {e}')
            res = None

    if res is not None and not is_quicksettings:
        try:
            res.change(fn=None, inputs=res, _js=f'(val) => markIfModified("{key}", val)')
        except Exception as e:
            shared.log.error(f'Quicksetting: component={res} {e}')
        if dirty_indicator is not None:
            dirty_indicator.click(fn=lambda: shared.opts.get_default(key), outputs=[res], show_progress=False)
        dirtyable_setting.__exit__()

    return res

def create_dirty_indicator(key, keys_to_reset, **kwargs):
    def get_default_values():
        values = [shared.opts.get_default(key) for key in keys_to_reset]
        shared.log.debug(f'Settings restore: section={key} keys={keys_to_reset} values={values}')
        return values

    elements_to_reset = [shared.settings_components[_key] for _key in keys_to_reset if shared.settings_components[_key] is not None]
    indicator = gr.Button('', elem_classes="modification-indicator", elem_id=f"modification_indicator_{key}", **kwargs)
    indicator.click(fn=get_default_values, outputs=elements_to_reset, show_progress=True)
    return indicator


def run_settings(*args):
    changed = []
    for key, value, comp in zip(shared.opts.data_labels.keys(), args, components):
        if comp == dummy_component or value=='dummy': # or getattr(comp, 'visible', True) is False or key in hidden_list:
            # actual = shared.opts.data.get(key, None)  # ensure the key is in data
            # default = shared.opts.data_labels[key].default
            # shared.log.warning(f'Setting skip: key={key} value={value} actual={actual} default={default} comp={comp}')
            continue
        if not shared.opts.same_type(value, shared.opts.data_labels[key].default):
            shared.log.error(f'Setting bad value: {key}={value} expecting={type(shared.opts.data_labels[key].default).__name__}')
            continue
        if shared.opts.set(key, value):
            changed.append(key)
    if shared.opts.cuda_compile_backend == "olive-ai":
        from modules.onnx_impl import install_olive, initialize_onnx_pipelines
        install_olive()
        initialize_onnx_pipelines()
    if shared.cmd_opts.use_directml:
        from modules.dml import directml_override_opts
        directml_override_opts()
    if shared.cmd_opts.use_openvino:
        if "Model" not in shared.opts.cuda_compile:
            shared.log.warning("OpenVINO: Enabling Torch Compile Model")
            shared.opts.cuda_compile.append("Model")
        if shared.opts.cuda_compile_backend != "openvino_fx":
            shared.log.warning("OpenVINO: Setting Torch Compiler backend to OpenVINO FX")
            shared.opts.cuda_compile_backend = "openvino_fx"
    if shared.opts.sd_backend != "diffusers":
        shared.log.error('Legacy option: backend=original is no longer supported')
        shared.opts.sd_backend = "diffusers"
    try:
        if len(changed) > 0:
            shared.opts.save(shared.config_filename)
            shared.log.info(f'Settings: changed={len(changed)} {changed}')
    except RuntimeError:
        shared.log.error(f'Settings failed: change={len(changed)} {changed}')
        return shared.opts.dumpjson(), f'{len(changed)} Settings changed without save: {", ".join(changed)}'
    return shared.opts.dumpjson(), f'{len(changed)} Settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}'

def run_settings_single(value, key, progress=False):
    if not shared.opts.same_type(value, shared.opts.data_labels[key].default):
        return gr.update(visible=True), shared.opts.dumpjson()
    if not shared.opts.set(key, value):
        return gr.update(value=getattr(shared.opts, key)), shared.opts.dumpjson()
    if key == "cuda_compile_backend" and value == "olive-ai":
        from modules.onnx_impl import install_olive
        install_olive()
    if shared.cmd_opts.use_directml:
        from modules.dml import directml_override_opts
        directml_override_opts()
    shared.opts.save(shared.config_filename)
    if key not in ['sd_model_checkpoint', 'sd_model_refiner', 'sd_vae', 'sd_te', 'sd_unet']:
        shared.log.debug(f'Setting changed: {key}={value} progress={progress}')
    return get_value_for_setting(key), shared.opts.dumpjson()


def create_ui():
    global text_settings # pylint: disable=global-statement
    text_settings = gr.Textbox(elem_id="settings_json", elem_classes=["settings_json"], value=lambda: shared.opts.dumpjson(), visible=False)
    with gr.Row(elem_id="system_row"):
        restart_submit = gr.Button(value="Restart server", variant='primary', elem_id="restart_submit")
        shutdown_submit = gr.Button(value="Shutdown server", variant='primary', elem_id="shutdown_submit")
        unload_sd_model = gr.Button(value='Unload model', variant='primary', elem_id="sett_unload_sd_model")
        reload_sd_model = gr.Button(value='Reload model', variant='primary', elem_id="sett_reload_sd_model")
        enable_profiling = gr.Button(value='Start profiling', variant='primary', elem_id="enable_profiling")

    with gr.Tabs(elem_id="system") as system_tabs:
        global ui_system_tabs # pylint: disable=global-statement
        ui_system_tabs = system_tabs
        with gr.TabItem("Settings", id="system_settings", elem_id="tab_settings"):
            with gr.Row(elem_id="settings_row"):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
                preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
                defaults_submit = gr.Button(value="Restore defaults", variant='primary', elem_id="defaults_submit")
            with gr.Row():
                _settings_search = gr.Textbox(label="Search", elem_id="settings_search")

            result = gr.HTML(elem_id="settings_result")
            script_callbacks.ui_settings_callback() # let extensions create settings
            sections = []
            options_count = len(shared.opts.data_labels)
            for item in shared.opts.data_labels.values(): # get unique sections from all items
                if len(item.section) == 2:
                    section_id, section_text = item.section
                elif len(item.section) == 3: # compatibility item with a1111 extensions
                    _category, section_id, section_text = item.section
                    item.section = section_id, section_text
                else:
                    section_id = None
                    item.section = None, 'Hidden'
                if (section_id, section_text) not in sections:
                    sections.append((section_id, section_text))

            shared.log.debug(f'Settings: sections={len(sections)} settings={len(shared.opts.list())}/{len(list(shared.opts.data_labels))} quicksettings={len(quicksettings_list)}')
            with gr.Tabs(elem_id="settings"):
                quicksettings_list.clear()
                for (section_id, section_text) in sections:
                    items = [item for item in shared.opts.data_labels.items() if item[1].section[0] == section_id] # find all items in this section
                    hidden = section_id is None or 'hidden' in section_id.lower() or 'hidden' in section_text.lower()
                    # shared.log.trace(f'Settings: section="{section_id}" title="{section_text}" items={len(items)} hidden={hidden}')
                    if hidden:
                        for (key, _item) in items:
                            hidden_list.append(key)
                            components.append(dummy_component)
                    else:
                        with gr.TabItem(elem_id=f"settings_section_tab_{section_id}", label=section_text):
                            current_items = []
                            for (key, item) in items:
                                if key in quicksettings_names:
                                    quicksettings_list.append((key, item))
                                    components.append(dummy_component)
                                else:
                                    with gr.Row(elem_id=f"settings_section_row_{section_id}"): # only so we can add dirty indicator at the start of the row
                                        component = create_setting_component(key)
                                        shared.settings_components[key] = component
                                        current_items.append(key)
                                        components.append(component)
                        create_dirty_indicator(section_id, current_items)
                components_count = len(components)
                if components_count != options_count:
                    shared.log.error(f'Settings: count mismatch: options={options_count} components={components_count}')

                with gr.TabItem("Show all pages", elem_id="settings_show_all_pages"):
                    create_dirty_indicator("show_all_pages", [])
                request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications", visible=False)

        with gr.TabItem("Update", id="system_update", elem_id="tab_update"):
            from modules import update
            update.create_ui()

        with gr.TabItem("User interface", id="system_config", elem_id="tab_config"):
            loadsave.create_ui()
            create_dirty_indicator("tab_defaults", [], interactive=False)

        with gr.TabItem("History", id="system_history", elem_id="tab_history"):
            ui_history.create_ui()

        with gr.TabItem("GPU Monitor", id="system_gpu", elem_id="tab_gpu"):
            with gr.Row(elem_id='gpu-controls'):
                gpu_start = gr.Button(value="Start", elem_id="gpu_start", variant="primary")
                gpu_stop = gr.Button(value="Stop", elem_id="gpu_stop", variant="primary")
                gpu_start.click(fn=lambda: None, _js='startGPU', inputs=[], outputs=[])
                gpu_stop.click(fn=lambda: None, _js='disableGPU', inputs=[], outputs=[])
            gr.HTML('''
                <div class="gpu" id="gpu">
                    <table class="gpu-table" id="gpu-table">
                        <thead><tr><th></th><th></th></tr></thead>
                        <tbody></tbody>
                    </table>
                    <div id="gpuChart"></div>
                </div>
            ''', elem_id='gpu-container', visible=True)

        with gr.TabItem("ONNX", id="onnx_config", elem_id="tab_onnx"):
            from modules.onnx_impl import ui as ui_onnx
            ui_onnx.create_ui()

    def unload_sd_weights():
        sd_models.unload_model_weights(op='model')
        sd_models.unload_model_weights(op='refiner')

    def reload_sd_weights():
        sd_models.reload_model_weights(force=True)

    def switch_profiling():
        shared.cmd_opts.profile = not shared.cmd_opts.profile
        shared.log.warning(f'Profiling: {shared.cmd_opts.profile}')
        return 'Stop profiling' if shared.cmd_opts.profile else 'Start profiling'

    unload_sd_model.click(fn=unload_sd_weights, inputs=[], outputs=[])
    reload_sd_model.click(fn=reload_sd_weights, inputs=[], outputs=[])
    enable_profiling.click(fn=switch_profiling, inputs=[], outputs=[enable_profiling])
    request_notifications.click(fn=lambda: None, inputs=[], outputs=[], _js='function(){}')
    preview_theme.click(fn=None, _js='previewTheme', inputs=[], outputs=[])
    settings_submit.click(
        fn=call_queue.wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
        inputs=components,
        outputs=[text_settings, result],
    )
    defaults_submit.click(fn=lambda: shared.restore_defaults(restart=True), _js="restartReload")
    restart_submit.click(fn=lambda: shared.restart_server(restart=True), _js="restartReload")
    shutdown_submit.click(fn=lambda: shared.restart_server(restart=False), _js="restartReload")


def reset_quicksettings(quick_components):
    quick_components = quick_components.split(',')
    updates = []
    for key in quick_components:
        shared.log.warning(f'Reset: setting={key}')
        updates.append(gr.update(value=shared.opts.get_default(key)))
    return updates


def create_quicksettings(interfaces):
    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=theme.gradio_theme, analytics_enabled=False, title="SD.Next") as ui_app:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            quicksetting_components = []
            quicksetting_keys = []
            for k, _item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                quicksetting_components.append(component)
                quicksetting_keys.append(k)
                shared.settings_components[k] = component
            quicksetting_keys = gr.State(value=','.join(quicksetting_keys), elem_id="quicksettings_keys")
            btn_reset = ui_components.ToolButton(value=ui_symbols.clear, visible=True, elem_id="quicksettings_reset")
            btn_reset.click(fn=reset_quicksettings, inputs=[quicksetting_keys], outputs=quicksetting_components)

        generation_parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                if interface is None:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()
            for interface, _label, ifid in interfaces:
                if interface is None:
                    continue
                if ifid in ["extensions", "system"]:
                    continue
                loadsave.add_block(interface, ifid)
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()

        if shared.opts.notification_audio_enable and os.path.exists(os.path.join(paths.script_path, shared.opts.notification_audio_path)):
            gr.Audio(interactive=False, value=os.path.join(paths.script_path, shared.opts.notification_audio_path), elem_id="audio_notification", visible=False)

        for k, _item in quicksettings_list:
            component = shared.settings_components[k]
            info = shared.opts.data_labels[k]
            if isinstance(component, gr.components.Textbox):
                change_handlers = [component.blur, component.submit]
            else:
                change_handlers = [component.release if hasattr(component, 'release') else component.change]
            for change_handler in change_handlers:
                change_handler(
                    fn=lambda value, k=k, progress=info.refresh is not None: run_settings_single(value, key=k, progress=progress),
                    inputs=[component],
                    outputs=[component, text_settings],
                    show_progress=info.refresh is not None,
                )

        button_set_checkpoint = gr.Button('Change model', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[shared.settings_components['sd_model_checkpoint'], dummy_component],
            outputs=[shared.settings_components['sd_model_checkpoint'], text_settings],
        )
        button_set_refiner = gr.Button('Change refiner', elem_id='change_refiner', visible=False)
        button_set_refiner.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[shared.settings_components['sd_model_refiner'], dummy_component],
            outputs=[shared.settings_components['sd_model_refiner'], text_settings],
        )
        button_set_vae = gr.Button('Change VAE', elem_id='change_vae', visible=False)
        button_set_vae.click(
            fn=lambda value, _: run_settings_single(value, key='sd_vae'),
            _js="function(v){ var res = desiredVAEName; desiredVAEName = ''; return [res || v, null]; }",
            inputs=[shared.settings_components['sd_vae'], dummy_component],
            outputs=[shared.settings_components['sd_vae'], text_settings],
        )

        def reference_submit(model):
            if '@' not in model: # diffusers
                loaded = modelloader.load_reference(model)
                return model if loaded else shared.opts.sd_model_checkpoint
            else: # civitai
                model, url = model.split('@')
                loaded = modelloader.load_civitai(model, url)
                return loaded if loaded is not None else shared.opts.sd_model_checkpoint

        button_set_reference = gr.Button('Change reference', elem_id='change_reference', visible=False)
        button_set_reference.click(
            fn=reference_submit,
            _js="function(v){ return desiredCheckpointName; }",
            inputs=[shared.settings_components['sd_model_checkpoint']],
            outputs=[shared.settings_components['sd_model_checkpoint']],
        )
        component_keys = [k for k in shared.opts.data_labels.keys() if k in shared.settings_components]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        ui_app.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[shared.settings_components[k] for k in component_keys if shared.settings_components[k] is not None],
            queue=False,
        )

    timer.startup.record("ui-defaults")
    loadsave.dump_defaults()
    ui_app.ui_loadsave = loadsave
    return ui_app
