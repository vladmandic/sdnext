#!/usr/bin/env python
import io
import os
import time
import base64
import logging
import requests
import urllib3
import pathvalidate
from PIL import Image


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


output_folder = 'outputs/compare'
models = [
    "sdxl-base-v10-vaefix",
    "tempest-by-vlad-0.1",
    "icbinpXL_v6",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Kontext-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
    "vladmandic/chroma-unlocked-v50",
    "vladmandic/chroma-unlocked-v50-annealed",
    "Qwen/Qwen-Image",
    "briaai/BRIA-3.2",
    "stabilityai/stable-cascade",
    "ostris/Flex.2-preview",
    "OmniGen2/OmniGen2",
    "Freepik/F-Lite",
    "Freepik/F-Lite-Texture",
    "HiDream-ai/HiDream-I1-Full",
    "nvidia/Cosmos-Predict2-2B-Text2Image",
    "nvidia/Cosmos-Predict2-14B-Text2Image",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
    "fal/AuraFlow-v0.3",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    "Alpha-VLLM/Lumina-Next-SFT-diffusers",
    "Alpha-VLLM/Lumina-Image-2.0",
    "Kwai-Kolors/Kolors-diffusers",
    "THUDM/CogView4-6B",
    "kandinsky-community/kandinsky-3",
]
styles = [
    'Fixed Astronaut',
    'Fixed Bear',
    'Fixed Steampunk City',
    'Fixed Road sign',
    'Fixed Futuristic hypercar',
    'Fixed Pirate Ship in Space',
    'Fixed Fallout girl',
    'Fixed Kneeling on Bed',
    'Fixed Girl in Sin City',
    'Fixed Girl in a city',
    'Fixed Lady in Tokyo',
    'Fixed MadMax selfie',
    'Fixed SDNext Neon',
]


def request(endpoint: str, dct: dict = None, method: str = 'POST'):
    def auth():
        if sd_username is not None and sd_password is not None:
            return requests.auth.HTTPBasicAuth(sd_username, sd_password)
        return None
    sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
    sd_username = os.environ.get('SDAPI_USR', None)
    sd_password = os.environ.get('SDAPI_PWD', None)
    method = requests.post if method.upper() == 'POST' else requests.get
    req = method(f'{sd_url}{endpoint}', json = dct, timeout=120000, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def generate(): # pylint: disable=redefined-outer-name
    for m, model in enumerate(models):
        model_name = pathvalidate.sanitize_filename(model, replacement_text='_')
        log.info(f'model: name="{model}" n={m+1}/{len(models)}')
        for s, style in enumerate(styles):
            model_name = pathvalidate.sanitize_filename(model, replacement_text='_')
            style_name = pathvalidate.sanitize_filename(style, replacement_text='_')
            fn = os.path.join(output_folder, f'{model_name}__{style_name}.jpg')
            if os.path.exists(fn):
                continue
            request(f'/sdapi/v1/checkpoint?sd_model_checkpoint={model}', method='POST')
            loaded = request('/sdapi/v1/checkpoint', method='GET')
            if not (model in loaded.get('checkpoint') or model in loaded.get('title') or model in loaded.get('name')):
                log.error(f' model: error="{model}"')
                continue
            log.info(f' style: name="{style}" n={s+1}/{len(styles)} fn="{fn}"')
            t0 = time.time()
            data = request('/sdapi/v1/txt2img', { 'styles': [style] })
            t1 = time.time()
            if 'images' in data and len(data['images']) > 0:
                b64 = data['images'][0].split(',',1)[0]
                image = Image.open(io.BytesIO(base64.b64decode(b64)))
                info = data['info']
                log.info(f' image: size={image.size} time={t1-t0:.2f} info="{len(info)}" fn="{fn}"')
                image.save(fn)


if __name__ == "__main__":
    log.info('test-all-models')
    log.info(f'output="{output_folder}" models={len(models)} styles={len(styles)}')
    log.info('start...')
    generate()
    log.info('done...')
