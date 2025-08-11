#!/usr/bin/env python
"""
- fal/AuraFlow-v0.3: SDNQ: layer_class_name=Linear layer_weight_shape=torch.Size([3072, 2, 1024]) weights_dtype=int8 unsupported
"""

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
    "briaai/BRIA-3.2",
    "Freepik/F-Lite",
    "Freepik/F-Lite-Texture",
    "ostris/Flex.2-preview",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    "fal/AuraFlow-v0.3",
    "zai-org/CogView4-6B",
    "zai-org/CogView3-Plus-3B",
    "nvidia/Cosmos-Predict2-2B-Text2Image",
    "nvidia/Cosmos-Predict2-14B-Text2Image",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Lightning",
    "Shitao/OmniGen-v1-diffusers",
    "OmniGen2/OmniGen2",
    "HiDream-ai/HiDream-I1-Full",
    "Kwai-Kolors/Kolors-diffusers",
    "vladmandic/chroma-unlocked-v50",
    "vladmandic/chroma-unlocked-v50-annealed",
    "Alpha-VLLM/Lumina-Next-SFT-diffusers",
    "Alpha-VLLM/Lumina-Image-2.0",
    "MeissonFlow/Meissonic",
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "stabilityai/stable-cascade",
]
models_tbd = [
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Kontext-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", # TODO
    "kandinsky-community/kandinsky-3", # TODO
]
styles = [
    'Fixed Astronaut',
]
styles_tbd = [
    'Fixed Bear',
    'Fixed Steampunk City',
    'Fixed Road sign',
    'Fixed Futuristic hypercar',
    'Fixed Pirate Ship in Space',
    'Fixed Fallout girl',
    'Fixed Kneeling on Bed',
    'Fixed Girl in Sin City',
    'Fixed Girl in a city',
    'Fixed Girl in Lace',
    'Fixed Lady in Tokyo',
    'Fixed MadMax selfie',
    'Fixed Party Yacht',
    'Fixed Yoga Girls',
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
            try:
                model_name = pathvalidate.sanitize_filename(model, replacement_text='_')
                style_name = pathvalidate.sanitize_filename(style, replacement_text='_')
                fn = os.path.join(output_folder, f'{model_name}__{style_name}.jpg')
                if os.path.exists(fn):
                    continue
                request(f'/sdapi/v1/checkpoint?sd_model_checkpoint={model}', method='POST')
                loaded = request('/sdapi/v1/checkpoint', method='GET')
                if not loaded or not (model in loaded.get('checkpoint') or model in loaded.get('title') or model in loaded.get('name')):
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
                    log.info(f' image: size={image.width}x{image.height} time={t1-t0:.2f} info={len(info)}')
                    image.save(fn)
                else:
                    log.error(f' model: error="{model}" style="{style}" no image')
            except Exception as e:
                log.error(f' model: error="{model}" style="{style}" exception="{e}"')

if __name__ == "__main__":
    log.info('test-all-models')
    log.info(f'output="{output_folder}" models={len(models)} styles={len(styles)}')
    log.info('start...')
    generate()
    log.info('done...')
