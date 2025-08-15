#!/usr/bin/env python
import os
import logging
import requests
import urllib3


sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)
options = {
    "save_images": True,
    "send_images": True,
}

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def get(endpoint: str, dct: dict = None):
    req = requests.get(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


if __name__ == "__main__":
    model = get('/sdapi/v1/checkpoint')
    log.info(f'api-checkpoint: {model}')
    model = get('/sdapi/v1/modules')
    log.info(f'api-modules: {model}')
