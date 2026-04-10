from typing import Any
from modules.shared import opts


def get_options():
    options = {}
    for k in opts.data.keys():
        if opts.data_labels.get(k) is not None:
            options.update({k: opts.data.get(k, opts.data_labels.get(k).default)})
        else:
            options.update({k: opts.data.get(k, None)})
    if "sd_lyco" in options:
        del options["sd_lyco"]
    if "sd_lora" in options:
        del options["sd_lora"]
    return options

def set_options(req: dict[str, Any]):
    updated = []
    for k, v in req.items():
        updated.append({k: opts.set(k, v)})
    opts.save()
    return {"updated": updated}

def get_options_info():
    """
    Return metadata for all application settings.
    Returns every registered option with its label, section, type, default value,
    component kind (slider, switch, dropdown, etc.), and component args (min/max/step/choices).
    Used by alternative UIs to dynamically build a settings editor.
    """
    import re
    import gradio as gr
    from modules.shared_legacy import LegacyOption
    from modules.ui_components import DropdownEditable
    component_map = {
        gr.Slider: "slider",
        gr.Checkbox: "switch",
        gr.Radio: "radio",
        gr.Dropdown: "dropdown",
        gr.Textbox: "input",
        gr.Number: "number",
        gr.ColorPicker: "color",
        gr.CheckboxGroup: "checkboxgroup",
        gr.HTML: "separator",
    }
    options_info = {}
    sections_seen = {}
    for key, info in opts.data_labels.items():
        section_id = info.section[0] if info.section else None
        section_title = info.section[1] if info.section and len(info.section) > 1 else ""
        hidden = section_id is None or "hidden" in (section_id or "").lower() or "hidden" in section_title.lower()
        if section_id and section_id not in sections_seen:
            sections_seen[section_id] = {"id": section_id, "title": section_title, "hidden": hidden}
        if hidden:
            args = {}
        else:
            try:
                args = info.component_args() if callable(info.component_args) else (info.component_args or {})
            except Exception:
                args = {}
        comp_name = component_map.get(info.component, "input")
        if info.component is DropdownEditable:
            comp_name = "dropdown"
        elif info.component is None:
            comp_name = "switch" if isinstance(info.default, bool) else "number" if isinstance(info.default, (int, float)) else "input"
        visible = args.get("visible", True) and (comp_name == "separator" or len(info.label) > 2)
        serializable_args = {}
        for arg_key in ("minimum", "maximum", "step", "choices", "precision", "multiselect"):
            if arg_key in args:
                serializable_args[arg_key] = args[arg_key]
        label = info.label
        if comp_name == "separator" and not label and isinstance(info.default, str):
            label = re.sub(r"<[^>]+>", "", info.default).strip()
        options_info[key] = {
            "label": label,
            "section_id": section_id,
            "section_title": section_title,
            "visible": visible,
            "hidden": hidden,
            "type": "boolean" if isinstance(info.default, bool) else "number" if isinstance(info.default, (int, float)) else "array" if isinstance(info.default, list) else "string",
            "component": comp_name,
            "component_args": serializable_args,
            "default": info.default,
            "is_legacy": isinstance(info, LegacyOption),
            "is_secret": getattr(info, "secret", False),
        }
    return {"options": options_info, "sections": list(sections_seen.values())}


def register_api(api):
    api.add_api_route("/sdapi/v1/options", get_options, methods=["GET"], response_model=dict, tags=["Server"])
    api.add_api_route("/sdapi/v1/options", set_options, methods=["POST"], tags=["Server"])
    api.add_api_route("/sdapi/v1/options-info", get_options_info, methods=["GET"], tags=["Server"])
