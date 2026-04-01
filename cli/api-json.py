#!/usr/bin/env python

# curl -vX POST http://localhost:7860/sdapi/v1/txt2img --header "Content-Type: application/json" -d @3261.json
import os
import json
import logging
import argparse
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


def post(endpoint: str, payload: dict | None = None):
    if 'sdapi' not in endpoint:
        endpoint = f'sdapi/v1/{endpoint}'
    if 'http' not in endpoint:
        endpoint = f'{sd_url}/{endpoint}'
    req = requests.post(endpoint, json = payload, timeout=300, verify=False, auth=auth())
    return { 'error': req.status_code, 'reason': req.reason, 'url': req.url } if req.status_code != 200 else req.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-json')
    parser.add_argument('endpoint', nargs=1, help='endpoint')
    parser.add_argument('json', nargs=1, help='json data or file')
    args = parser.parse_args()
    log.info(f'api-json: {args}')
    if os.path.isfile(args.json[0]):
        with open(args.json[0], 'r', encoding='ascii') as f:
            dct = json.load(f)
    else:
        dct = json.loads(args.json[0])
    res = post(endpoint=args.endpoint[0], payload=dct)
    print(res)
