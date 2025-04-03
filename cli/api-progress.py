#!/usr/bin/env python

"""
check progress of last job and shutdown system if timeout reached
"""

import os
import time
import datetime
import logging
import urllib3
import requests

class Dot(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opts = Dot({
    "timeout": 3600,
    "frequency": 1,
    "action": "sudo shutdown now",
    "url": "http://127.0.0.1:7860",
    "user": "",
    "password": "",
})

log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level = logging.INFO, format = log_format)
log = logging.getLogger("sd")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
status = None

def progress():
    auth = requests.auth.HTTPBasicAuth(opts.user, opts.password) if opts.user is not None and len(opts.user) > 0 and opts.password is not None and len(opts.password) > 0 else None
    req = requests.get(f'{opts.url}/sdapi/v1/progress?skip_current_image=true', verify=False, auth=auth, timeout=60)
    if req.status_code != 200:
        log.error({ 'url': req.url, 'request': req.status_code, 'reason': req.reason })
        return status
    else:
        res = Dot(req.json())
        log.debug({ 'url': req.url, 'request': req.status_code, 'result': res })
        return res

log.info(f'sdnext monitor started: {opts}')
while True:
    try:
        status = progress()
        # {'progress': 0.0, 'eta_relative': 0.0, 'state': {'skipped': False, 'interrupted': False, 'job': '', 'job_count': 0, 'job_timestamp': '20250316110822', 'job_no': 0, 'sampling_step': 20, 'sampling_steps': 20}, 'current_image': None, 'textinfo': None}
        state = status.get('state', {})
        job_timestamp = state.get('job_timestamp', None)
        job_progress = status.get('progress', 0)
        eta_relative = status.get('eta_relative', 0)
        job = state.get('job', '')
        job_timestamp = state.get('job_timestamp', None)
        sampling_step = state.get('sampling_step', 0)
        sampling_steps = state.get('sampling_steps', 0)
        if job_timestamp is None:
            log.warning(f'sdnext montoring cannot get last job info: {status}')
        else:
            job_timestamp = datetime.datetime.strptime(job_timestamp, "%Y%m%d%H%M%S") if job_timestamp != '0' else datetime.datetime.now()
            elapsed = datetime.datetime.now() - job_timestamp
            timeout = round(opts.timeout - elapsed.total_seconds())
            log.info(f'sdnext: last="{job_timestamp}" elapsed={elapsed} timeout={timeout} progress={job_progress} eta={eta_relative} step={sampling_step}/{sampling_steps} job="{job}"')
            if timeout < 0:
                log.warning(f'sdnext reached: timeout={opts.timeout} action={opts.action}')
                os.system(opts.action)
    except Exception as e:
        log.error(f'sdnext monitor error: {e}')
    finally:
        time.sleep(opts.frequency)
