import time
from typing import Any, Dict
from fastapi import Depends
from modules import shared
from modules.api import models, helpers


def post_shutdown():
    shared.log.info('Shutdown request received')
    import sys
    sys.exit(0)

def get_motd():
    import requests
    motd = ''
    ver = shared.get_version()
    if ver.get('updated', None) is not None:
        motd = f"version <b>{ver['commit']} {ver['updated']}</b> <span style='color: var(--primary-500)'>{ver['url'].split('/')[-1]}</span><br>" # pylint: disable=use-maxsplit-arg
    if shared.opts.motd:
        try:
            res = requests.get('https://vladmandic.github.io/sdnext/motd', timeout=3)
            if res.status_code == 200:
                msg = (res.text or '').strip()
                shared.log.info(f'MOTD: {msg if len(msg) > 0 else "N/A"}')
                motd += res.text
            else:
                shared.log.error(f'MOTD: {res.status_code}')
        except Exception as err:
            shared.log.error(f'MOTD: {err}')
    return motd

def get_version():
    return shared.get_version()

def get_platform():
    from installer import get_platform as installer_get_platform
    from modules.loader import get_packages as loader_get_packages
    return { **installer_get_platform(), **loader_get_packages() }

def get_log(req: models.ReqGetLog = Depends()):
    lines = shared.log.buffer[:req.lines] if req.lines > 0 else shared.log.buffer.copy()
    if req.clear:
        shared.log.buffer.clear()
    return lines

def post_log(req: models.ReqPostLog):
    if req.message is not None:
        shared.log.info(f'UI: {req.message}')
    if req.debug is not None:
        shared.log.debug(f'UI: {req.debug}')
    if req.error is not None:
        shared.log.error(f'UI: {req.error}')
    return {}


def get_config():
    options = {}
    for k in shared.opts.data.keys():
        if shared.opts.data_labels.get(k) is not None:
            options.update({k: shared.opts.data.get(k, shared.opts.data_labels.get(k).default)})
        else:
            options.update({k: shared.opts.data.get(k, None)})
    if 'sd_lyco' in options:
        del options['sd_lyco']
    if 'sd_lora' in options:
        del options['sd_lora']
    return options

def set_config(req: Dict[str, Any]):
    updated = []
    for k, v in req.items():
        updated.append({ k: shared.opts.set(k, v) })
    shared.opts.save(shared.config_filename)
    return { "updated": updated }

def get_cmd_flags():
    return vars(shared.cmd_opts)

def get_history(req: models.ReqHistory = Depends()):
    if req.id is not None and len(req.id) > 0:
        res = [item for item in shared.state.state_history if item['id'] == req.id]
    else:
        res = shared.state.state_history
    res = [models.ResHistory(**item) for item in res]
    return res

def get_progress(req: models.ReqProgress = Depends()):
    if shared.state.job_count == 0: # idle state
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
    # shared.log.critical(f'get_progress: batch {batch_x}/{batch_y} step {step_x}/{step_y} current {current}/{total} time={time_since_start} eta={eta_relative}')
    # shared.log.critical(shared.state)
    res = models.ResProgress(id=shared.state.id, progress=round(progress, 2), eta_relative=round(eta_relative, 2), current_image=current_image, textinfo=shared.state.textinfo, state=shared.state.dict(), )
    return res

def get_status():
    return shared.state.status()

def post_interrupt():
    shared.state.interrupt()
    return {}

def post_skip():
    shared.state.skip()

def get_memory():
    try:
        import os
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
