from __future__ import annotations
import base64
import io
import os
from PIL import Image
import gradio as gr
from modules import shared, gr_tempdir, script_callbacks, images
from modules.logger import log
from modules.infotext import parse, mapping # pylint: disable=unused-import


type_of_gr_update = type(gr.update())
paste_fields: dict[str, dict] = {}
field_names = {}
registered_param_bindings: list[ParamBinding] = []
debug = log.trace if os.environ.get('SD_PASTE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PASTE')
parse_generation_parameters = parse # compatibility
infotext_to_setting_name_mapping = mapping # compatibility

# Mapping of aliases to metadata parameter names, populated automatically from component labels/elem_ids
# This allows users to use component labels, elem_ids, or metadata names in the "skip params" setting
param_aliases: dict[str, str] = {}


class ParamBinding:
    def __init__(self, paste_button, tabname: str, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []
        # debug(f'ParamBinding: {vars(self)}')


def reset():
    paste_fields.clear()
    field_names.clear()


def image_from_url_text(filedata):
    if filedata is None:
        return None
    if type(filedata) == list and len(filedata) > 0 and type(filedata[0]) == dict and filedata[0].get("is_file", False):
        filedata = filedata[0]
    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]
        is_in_right_dir = gr_tempdir.check_tmp_file(shared.demo, filename)
        if is_in_right_dir:
            filename = filename.rsplit('?', 1)[0]
            if not os.path.exists(filename):
                log.error(f'Image file not found: {filename}')
                image = Image.new('RGB', (512, 512))
                image.info['parameters'] = f'Image file not found: {filename}'
                return image
            image = Image.open(filename)
            geninfo, _items = images.read_info_from_image(image)
            image.info['parameters'] = geninfo
            return image
        else:
            log.warning(f'File access denied: {filename}')
            return None
    if type(filedata) == list:
        if len(filedata) == 0:
            return None
        filedata = filedata[0]
    if not isinstance(filedata, str):
        log.warning('Incorrect filedata received')
        return None
    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]
    if filedata.startswith("data:image/webp;base64,"):
        filedata = filedata[len("data:image/webp;base64,"):]
    if filedata.startswith("data:image/jpeg;base64,"):
        filedata = filedata[len("data:image/jpeg;base64,"):]
    if filedata.startswith("data:image/jxl;base64,"):
        filedata = filedata[len("data:image/jxl;base64,"):]
    filebytes = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filebytes))
    image.load()
    # images.read_info_from_image(image)
    return image


def add_paste_fields(tabname: str, init_img: gr.Image | gr.HTML | None, fields: list[tuple[gr.components.Component, str]] | None, override_settings_component=None):
    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}
    try:
        field_names[tabname] = [f[1] for f in fields if f[1] is not None and not callable(f[1])] if fields is not None else [] # tuple (component, label)
    except Exception as e:
        log.error(f"Paste fields: tab={tabname} fields={fields} {e}")
        field_names[tabname] = []

    # Build param_aliases automatically from component labels and elem_ids
    if fields is not None:
        for component, metadata_name in fields:
            if metadata_name is None or callable(metadata_name):
                continue
            metadata_lower = metadata_name.lower()
            # Extract label from component (e.g., "Batch size" -> maps to "Batch-2")
            label = getattr(component, 'label', None)
            if label and isinstance(label, str):
                label_lower = label.lower()
                if label_lower != metadata_lower and label_lower not in param_aliases:
                    param_aliases[label_lower] = metadata_lower
            # Extract elem_id and derive variable name (e.g., "txt2img_batch_size" -> "batch_size")
            elem_id = getattr(component, 'elem_id', None)
            if elem_id and isinstance(elem_id, str):
                # Strip common prefixes like "txt2img_", "img2img_", "control_"
                var_name = elem_id
                for prefix in ['txt2img_', 'img2img_', 'control_', 'video_', 'extras_']:
                    if var_name.startswith(prefix):
                        var_name = var_name[len(prefix):]
                        break
                var_name_lower = var_name.lower()
                if var_name_lower != metadata_lower and var_name_lower not in param_aliases:
                    param_aliases[var_name_lower] = metadata_lower

    # backwards compatibility for existing extensions
    debug(f'Paste fields: tab={tabname} fields={field_names[tabname]}')
    debug(f'All fields: {get_all_fields()}')
    debug(f'Param aliases: {param_aliases}')
    import modules.ui
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields # compatibility
    elif tabname == 'img2img':
        modules.ui.img2img_paste_fields = fields # compatibility
    elif tabname == 'control':
        modules.ui.control_paste_fields = fields
    elif tabname == 'video':
        modules.ui.video_paste_fields = fields


def get_all_fields():
    all_fields = []
    for _tab, fields in field_names.items():
        for field in fields:
            field = field.replace('-1', '').replace('-2', '').lower()
            if field not in all_fields:
                all_fields.append(field)
    return all_fields


def create_buttons(tabs_list: list[str]) -> dict[str, gr.Button]:
    buttons = {}
    for tab in tabs_list:
        name = tab
        if name == 'txt2img':
            name = 'Text'
        elif name == 'img2img':
            name = 'Image'
        elif name == 'inpaint':
            name = 'Inpaint'
        elif name == 'extras':
            name = 'Process'
        elif name == 'control':
            name = 'Control'
        elif name == 'caption':
            name = 'Caption'
        buttons[tab] = gr.Button(f"➠ {name}", elem_id=f"{tab}_tab")
    return buttons


def should_skip(param: str):
    skip_params = [p.strip().lower() for p in shared.opts.disable_apply_params.split(",")]
    if not shared.opts.clip_skip_enabled:
        skip_params += ['clip skip']

    # Expand skip_params with aliases (e.g., "batch_size" -> "batch-2")
    expanded_skip = set(skip_params)
    for skip in skip_params:
        if skip in param_aliases:
            expanded_skip.add(param_aliases[skip])

    # Check if param should be skipped
    param_lower = param.lower()
    # Also check normalized name (without -1/-2) so "batch" skips both "batch-1" and "batch-2"
    param_normalized = param_lower.replace('-1', '').replace('-2', '')

    all_params = [p.lower() for p in get_all_fields()]
    valid = any(p in all_params for p in skip_params)
    skip = param_lower in expanded_skip or param_normalized in expanded_skip
    debug(f'Check: param="{param}" valid={valid} skip={skip} expanded={expanded_skip}')
    return skip


def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


def connect_paste_params_buttons():
    binding: ParamBinding
    for binding in registered_param_bindings:
        if binding.tabname not in paste_fields:
            debug(f"Not not registered: tab={binding.tabname}")
            continue
        fields: list[tuple[gr.components.Component, str]] = paste_fields[binding.tabname]["fields"]

        destination_image_component = paste_fields[binding.tabname]["init_img"]
        if binding.source_image_component:
            if isinstance(destination_image_component, gr.Image):
                binding.paste_button.click(
                    _js="extract_image_from_gallery" if isinstance(binding.source_image_component, gr.Gallery) else None,
                    fn=send_image,
                    inputs=[binding.source_image_component],
                    outputs=[destination_image_component],
                    show_progress='hidden',
                )
            elif isinstance(destination_image_component, gr.HTML): # kanvas
                binding.paste_button.click(
                    _js="send_to_kanvas",
                    fn=None,
                    inputs=[binding.source_image_component],
                    outputs=[],
                    show_progress='hidden',
                )
        override_settings_component = binding.override_settings_component or paste_fields[binding.tabname]["override_settings_component"]
        if binding.source_text_component is not None and fields is not None:
            connect_paste(binding.paste_button, fields, binding.source_text_component, override_settings_component, binding.tabname)
        if binding.source_tabname is not None and fields is not None and binding.source_tabname in paste_fields:
            paste_field_names = ['Prompt', 'Negative prompt', 'Steps'] + (["Seed"] if shared.opts.send_seed else []) + binding.paste_field_names
            if "fields" in paste_fields[binding.source_tabname] and paste_fields[binding.source_tabname]["fields"] is not None:
                binding.paste_button.click(
                    fn=lambda *x: x,
                    inputs=[field for field, name in paste_fields[binding.source_tabname]["fields"] if name in paste_field_names],
                    outputs=[field for field, name in fields if name in paste_field_names],
                )
        binding.paste_button.click(
            fn=None,
            _js=f"switch_to_{binding.tabname}",
            inputs=[],
            outputs=[],
            show_progress='hidden',
        )


def send_image(x):
    image = x if isinstance(x, Image.Image) else image_from_url_text(x)
    return image


def create_override_settings_dict(text_pairs):
    res = {}
    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)
        params[k] = v.strip()
    for param_name, setting_name in mapping:
        value = params.get(param_name, None)
        if value is None:
            continue
        res[setting_name] = shared.opts.cast_value(setting_name, value)
    return res


def connect_paste(button, local_paste_fields, input_comp, override_settings_component, tabname):

    def paste_func(prompt):
        from modules.paths import params_path
        if prompt is None or len(prompt.strip()) == 0:
            if os.path.exists(params_path):
                with open(params_path, encoding="utf8") as file:
                    prompt = file.read()
                log.debug(f'Prompt parse: type="params" prompt="{prompt}"')
            else:
                prompt = ''
        else:
            log.debug(f'Prompt parse: type="current" prompt="{prompt}"')
        params = parse(prompt)
        script_callbacks.infotext_pasted_callback(prompt, params)
        res = []
        applied = {}
        skipped = {}
        for output, key in local_paste_fields:
            if callable(key):
                v = key(params)
            else:
                v = params.get(key, None)
            if v is None:
                res.append(gr.update()) # triggers update for each gradio component even if there are no updates
            elif isinstance(v, type_of_gr_update):
                res.append(v)
                applied[key] = v
            else:
                if isinstance(v, str) and v.strip() == '' and key in {'Prompt', 'Negative prompt'}:
                    debug(f'Paste skip empty: "{key}"')
                    res.append(gr.update())
                    skipped[key] = v
                    continue
                if should_skip(key):
                    debug(f'Paste skip: "{key}"="{v}"')
                    res.append(gr.update())
                    skipped[key] = v
                    continue
                try:
                    valtype = type(output.value)
                    if hasattr(output, "step") and type(output.step) == float:
                        valtype = float
                    debug(f'Paste: "{key}"="{v}" type={valtype} var={vars(output)}')
                    if valtype == bool:
                        val = False if str(v).lower() == "false" else True
                    elif valtype == list:
                        val = v if isinstance(v, list) else [item.strip() for item in v.split(',')]
                    else:
                        val = valtype(v)
                    res.append(gr.update(value=val))
                    applied[key] = val
                except Exception as e:
                    log.error(f'Paste param: key="{key}" value="{v}" error="{e}"')
                    res.append(gr.update())
        list_applied = [{k: v} for k, v in applied.items() if not callable(v) and not callable(k)]
        log.debug(f"Prompt restore: apply={list_applied} skip={skipped}")
        return res

    if override_settings_component is not None:
        def paste_settings(params):
            params.pop('Prompt', None)
            params.pop('Negative prompt', None)
            if not params:
                gr.Dropdown.update(value=[], choices=[], visible=False)
            vals = {}
            for param_name, setting_name in infotext_to_setting_name_mapping:
                v = params.get(param_name, None)
                if v is None:
                    continue
                if setting_name == 'sd_backend':
                    continue
                if setting_name in shared.opts.disable_apply_metadata:
                    continue
                if should_skip(param_name) or should_skip(setting_name):
                    continue
                v = shared.opts.cast_value(setting_name, v)
                current_value = getattr(shared.opts, setting_name, None)
                if v == current_value:
                    continue
                if type(current_value) == str and v == os.path.splitext(current_value)[0]:
                    continue
                vals[param_name] = v
            vals_pairs = [f"{k}: {v}" for k, v in vals.items()]
            if len(vals_pairs) > 0:
                log.debug(f'Settings overrides: {vals_pairs}')
            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=len(vals_pairs) > 0)

        local_paste_fields = local_paste_fields + [(override_settings_component, paste_settings)]

    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in local_paste_fields],
        show_progress='hidden',
    )
    button.click(
        fn=None,
        _js=f"recalculate_prompts_{tabname}",
        inputs=[],
        outputs=[],
        show_progress='hidden',
    )
