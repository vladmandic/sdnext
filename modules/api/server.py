import os
import time
from typing import Any
from fastapi import Request, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
import installer
from modules import shared
from modules.logger import log
from modules.api import models, helpers

def _get_version():
    return installer.get_version()


def post_shutdown():
    """Shut down the server process immediately."""
    log.info('Shutdown request received')
    import sys
    sys.exit(0)

def get_js(request: Request):
    file = request.query_params.get("file", None)
    if (file is None) or (len(file) == 0):
        raise HTTPException(status_code=400, detail="file parameter is required")
    ext = file.split('.')[-1]
    if ext not in ['js', 'css', 'map', 'html', 'wasm', 'ttf', 'mjs', 'json']:
        raise HTTPException(status_code=400, detail=f"invalid file extension: {ext}")
    if not os.path.exists(file):
        log.error(f"API: file not found: {file}")
        raise HTTPException(status_code=404, detail=f"file not found: {file}")
    if ext in ['js', 'mjs']:
        media_type = 'application/javascript'
    elif ext in ['map', 'json']:
        media_type = 'application/json'
    elif ext in ['css']:
        media_type = 'text/css'
    elif ext in ['html']:
        media_type = 'text/html'
    elif ext in ['wasm']:
        media_type = 'application/wasm'
    elif ext in ['ttf']:
        media_type = 'font/ttf'
    else:
        media_type = 'application/octet-stream'
    return FileResponse(file, media_type=media_type)

def get_motd():
    """Return the message of the day, including version info and optional upstream MOTD."""
    import requests
    motd = ''
    ver = _get_version()
    if ver.get('updated', None) is not None:
        motd = f"version <b>{ver['commit']} {ver['updated']}</b> <span style='color: var(--primary-500)'>{ver['url'].split('/')[-1]}</span><br>" # pylint: disable=use-maxsplit-arg
    if shared.opts.motd:
        try:
            res = requests.get('https://vladmandic.github.io/sdnext/motd', timeout=3)
            if res.status_code == 200:
                msg = (res.text or '').strip()
                log.info(f'MOTD: {msg if len(msg) > 0 else "N/A"}')
                motd += res.text
            else:
                log.error(f'MOTD: {res.status_code}')
        except Exception as err:
            log.error(f'MOTD: {err}')
    return motd

def get_version():
    """Return application version, commit hash, branch, and update timestamp."""
    return _get_version()

def get_platform():
    """Return platform details (OS, Python, torch, CUDA versions) and installed package versions."""
    from installer import get_platform as installer_get_platform
    from modules.loader import get_packages as loader_get_packages
    return { **installer_get_platform(), **loader_get_packages() }

def get_log(req: models.ReqGetLog = Depends()):
    """Return recent log lines from the in-memory buffer. Optionally clears the buffer after reading."""
    lines = log.buffer[:req.lines] if req.lines > 0 else log.buffer.copy()
    if req.clear:
        log.buffer.clear()
    return lines

def post_log(req: models.ReqPostLog):
    """Write a message to the server log at info, debug, or error level."""
    if req.message is not None:
        log.info(f'UI: {req.message}')
    if req.debug is not None:
        log.debug(f'UI: {req.debug}')
    if req.error is not None:
        log.error(f'UI: {req.error}')
    return {}


def get_config():
    """Return all current application options as a key-value dictionary."""
    from modules import secrets_manager
    options = {}
    for k in shared.opts.data.keys():
        info = shared.opts.data_labels.get(k)
        if info is not None:
            if getattr(info, 'secret', False):
                continue  # handled below
            options[k] = shared.opts.data.get(k, info.default)
        else:
            options[k] = shared.opts.data.get(k, None)
    for k, info in shared.opts.data_labels.items():
        if getattr(info, 'secret', False):
            options[k] = secrets_manager.get_mask(k, info.env_var)
    if 'sd_lyco' in options:
        del options['sd_lyco']
    if 'sd_lora' in options:
        del options['sd_lora']
    return options

def set_config(req: dict[str, Any]):
    """Update one or more application options and persist them to disk."""
    from modules import secrets_manager
    updated = []
    for k, v in req.items():
        info = shared.opts.data_labels.get(k)
        if info and getattr(info, 'secret', False):
            mask = secrets_manager.get_mask(k, info.env_var)
            if str(v) != mask:
                secrets_manager.set(k, v)
            updated.append({k: True})
        else:
            updated.append({k: shared.opts.set(k, v)})
    shared.opts.save()
    return {"updated": updated}

def get_cmd_flags():
    """Return all command-line flags and their current values."""
    return vars(shared.cmd_opts)

def get_history(req: models.ReqHistory = Depends()):
    """Return generation job history. Optionally filter by task ID."""
    if req.id is not None and len(req.id) > 0:
        res = [item for item in shared.state.state_history if item['id'] == req.id]
    else:
        res = shared.state.state_history
    res = [models.ResHistory(**item) for item in res]
    return res

def get_progress(req: models.ReqProgress = Depends()):
    """Return current generation progress, ETA, and optionally a preview of the in-progress image."""
    if shared.state.job_count == 0 and shared.state.sampling_step == 0: # truly idle
        return models.ResProgress(id=shared.state.id, progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)
    shared.state.do_set_current_image()
    current_image = None
    if shared.state.current_image and not req.skip_current_image:
        current_image = helpers.encode_pil_to_base64(shared.state.current_image)
    batch_x = max(shared.state.job_no, 0)
    batch_y = max(shared.state.job_count, 1)
    step_x = max(shared.state.sampling_step, 0)
    prev_steps = max(shared.state.sampling_steps, 1)
    while step_x > shared.state.sampling_steps:
        shared.state.sampling_steps += prev_steps
    step_y = max(shared.state.sampling_steps, 1)
    current = step_y * batch_x + step_x
    total = step_y * batch_y
    progress = min((current / total) if current > 0 and total > 0 else 0, 1)
    time_since_start = time.time() - shared.state.time_start
    eta_relative = (time_since_start / progress) - time_since_start if progress > 0 else 0
    # log.critical(f'get_progress: batch {batch_x}/{batch_y} step {step_x}/{step_y} current {current}/{total} time={time_since_start} eta={eta_relative}')
    # log.critical(shared.state)
    res = models.ResProgress(id=shared.state.id, progress=round(progress, 2), eta_relative=round(eta_relative, 2), current_image=current_image, textinfo=shared.state.textinfo, state=shared.state.dict(), )
    return res

def get_status():
    """Return server status including current task, step progress, queue depth, and uptime."""
    return shared.state.status()

def post_interrupt():
    """Interrupt the currently running generation job."""
    shared.state.interrupt()
    return {}

def post_skip():
    """Skip the current image in a batch generation and move to the next."""
    shared.state.skip()

def get_memory():
    """Return system RAM and CUDA GPU memory usage statistics."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
        ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
        ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
    except Exception as err:
        ram = { 'error': f'{err}' }
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
            reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
            active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
            inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
            warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
            cuda = {
                'system': system,
                'active': active,
                'allocated': allocated,
                'reserved': reserved,
                'inactive': inactive,
                'events': warnings,
            }
        else:
            cuda = { 'error': 'unavailable' }
    except Exception as err:
        cuda = { 'error': f'{err}' }
    return models.ResMemory(ram = ram, cuda = cuda)

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
        gr.Slider: "slider", gr.Checkbox: "switch", gr.Radio: "radio",
        gr.Dropdown: "dropdown", gr.Textbox: "input", gr.Number: "number",
        gr.ColorPicker: "color", gr.CheckboxGroup: "checkboxgroup", gr.HTML: "separator",
    }
    options_info = {}
    sections_seen = {}

    for key, info in shared.opts.data_labels.items():
        section_id = info.section[0] if info.section else None
        section_title = info.section[1] if info.section and len(info.section) > 1 else ""
        hidden = section_id is None or 'hidden' in (section_id or '').lower() or 'hidden' in section_title.lower()

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

        visible = args.get('visible', True) and (comp_name == "separator" or len(info.label) > 2)

        serializable_args = {}
        for arg_key in ('minimum', 'maximum', 'step', 'choices', 'precision', 'multiselect'):
            if arg_key in args:
                serializable_args[arg_key] = args[arg_key]

        label = info.label
        if comp_name == "separator" and not label and isinstance(info.default, str):
            label = re.sub(r'<[^>]+>', '', info.default).strip()

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
            "is_secret": getattr(info, 'secret', False),
        }

    return {"options": options_info, "sections": list(sections_seen.values())}


def get_secrets_status():
    """Return configuration status for all secret options."""
    from modules import secrets_manager
    result = {}
    for key, info in shared.opts.data_labels.items():
        if getattr(info, 'secret', False):
            result[key] = secrets_manager.get_status(key, info.env_var)
    return result


def get_server_info():
    """
    Return server identity and capabilities.

    Returns version info, active backend (Diffusers), platform/device name, supported
    generation modes (txt2img, img2img, control, video, websocket), and the currently
    loaded model name and pipeline class.
    """
    from modules import devices
    from modules.sd_models import model_data
    ver = shared.get_version()
    model_name = None
    model_type = None
    if hasattr(shared.opts, 'sd_model_checkpoint'):
        model_name = shared.opts.sd_model_checkpoint
    if model_data.sd_model is not None:
        model_type = type(model_data.sd_model).__name__
    return {
        "version": ver,
        "backend": shared.backend.name if hasattr(shared.backend, 'name') else str(shared.backend),
        "platform": devices.get_device_name() if hasattr(devices, 'get_device_name') else str(shared.device),
        "api_version": "v1",
        "capabilities": {
            "txt2img": True,
            "img2img": True,
            "control": True,
            "video": True,
            "websocket": True,
        },
        "model": {
            "name": model_name,
            "type": model_type,
        }
    }
