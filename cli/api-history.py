#!/usr/bin/env python

"""
get list of all history jobs or a specific job
"""

import sys
import logging
import urllib3
import requests


url = "http://127.0.0.1:7860"
user = ""
password = ""

log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level = logging.INFO, format = log_format)
log = logging.getLogger("sd")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


log.info('state history')
sys.argv.pop(0)
task_id = sys.argv[0] if len(sys.argv) == 1 else ''
auth = requests.auth.HTTPBasicAuth(user, password) if len(user) > 0 and len(password) > 0 else None
req = requests.get(f'{url}/sdapi/v1/history?id={task_id}', verify=False, auth=auth, timeout=60)
if req.status_code != 200:
    log.error({ 'url': req.url, 'request': req.status_code, 'reason': req.reason })
    exit(1)
res = req.json()
for item in res:
    log.info(item)
