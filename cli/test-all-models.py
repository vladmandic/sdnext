#!/usr/bin/env python
"""
Warnings:
- fal/AuraFlow-v0.3: layer_class_name=Linear layer_weight_shape=torch.Size([3072, 2, 1024]) weights_dtype=int8 unsupported
- Kwai-Kolors/Kolors-diffusers: set_input_embeddings not autohandled for ChatGLMModel
- kandinsky-community/kandinsky-2-1: get_input_embeddings not autohandled for MultilingualCLIP
Errors:
- kandinsky-community/kandinsky-3: corrupt output
- Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers: CUDA error device-side assert triggered
Other:
- Wan-AI/Wan2.2-T2V-A14B-Diffusers: extreme memory usage
- HiDream-ai/HiDream-I1-Full: very slow at 30+s/it
"""

import io
import os
import time
import json
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
models = {
    "sdxl-base-v10-vaefix": {},
    "tempest-by-vlad-0.1": {},
    "icbinpXL_v6": {},
    "briaai/BRIA-3.2": {},
    "Freepik/F-Lite": {},
    "Freepik/F-Lite-Texture": {},
    "ostris/Flex.2-preview": {},
    "playgroundai/playground-v2-1024px-aesthetic": {},
    "playground-v2.5-1024px-aesthetic.fp16": { "sampler_name": "DPM++ 2M EDM" },
    "stabilityai/stable-diffusion-3.5-medium": {},
    "stabilityai/stable-diffusion-3.5-large": {},
    "fal/AuraFlow-v0.3": {},
    "fal/AuraFlow-v0.2": {},
    "zai-org/CogView4-6B": {},
    "zai-org/CogView3-Plus-3B": {},
    "Qwen/Qwen-Image": {},
    "vladmandic/Qwen-Lightning": {},
    "Shitao/OmniGen-v1-diffusers": {},
    "OmniGen2/OmniGen2": {},
    "Kwai-Kolors/Kolors-diffusers": {},
    "kandinsky-community/kandinsky-2-2-decoder": {},
    "kandinsky-community/kandinsky-2-1": {},
    "kandinsky-community/kandinsky-3": {},
    "Alpha-VLLM/Lumina-Next-SFT-diffusers": {},
    "Alpha-VLLM/Lumina-Image-2.0": {},
    "MeissonFlow/Meissonic": {},
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers": {},
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers": {},
    "PixArt-alpha/PixArt-XL-2-1024-MS": {},
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": {},
    "stabilityai/stable-cascade": {},
    "nvidia/Cosmos-Predict2-2B-Text2Image": {},
    "nvidia/Cosmos-Predict2-14B-Text2Image": {},
    "black-forest-labs/FLUX.1-dev": {},
    "black-forest-labs/FLUX.1-Kontext-dev": {},
    "black-forest-labs/FLUX.1-Krea-dev": {},
    "lodestones/Chroma1-HD": {},
    "vladmandic/chroma-unlocked-v50-annealed": {},
    "vladmandic/chroma-unlocked-v48": {},
    "vladmandic/chroma-unlocked-v48-detail-calibrated": {},
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {},
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {},
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": {},
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {},
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers": {},
    # "HiDream-ai/HiDream-I1-Full": {},
    # "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {},
}
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
history = []


def read_history():
    global history # pylint: disable=global-statement
    fn = os.path.join(output_folder, 'history.json')
    if not os.path.exists(fn):
        return
    with open(fn, "r", encoding='utf8') as file:
        data = file.read()
        history = json.loads(data)
    log.info(f'history: file="{fn}" records={len(history)}')


def write_history(model:str, style:str, image:str='', size:tuple=(0,0), generate:float=0, load:float=0, info:str=''):
    fn = os.path.join(output_folder, 'history.json')
    history.append({
        'model': model,
        'title': model.split('/')[-1].replace('_diffusers', '').replace('-diffusers', ''),
        'style': style,
        'image': image,
        'size': size,
        'time': generate,
        'load': load,
        'info': info,
    })
    with open(fn, "w", encoding='utf8') as file:
        data = json.dumps(history) # pylint: disable=no-member
        file.write(data)


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


def main(): # pylint: disable=redefined-outer-name
    idx_model = 0
    idx_images = 0
    t_generate0 = time.time()
    log.info(f'generate: models={len(models)} styles={len(styles)}')
    for model, args in models.items():
        t_model0 = time.time()
        idx_model += 1
        model_name = pathvalidate.sanitize_filename(model, replacement_text='_')
        log.info(f'model: n={idx_model+1}/{len(models)} name="{model}"')
        idx_style = 0
        for s, style in enumerate(styles):
            try:
                model_name = pathvalidate.sanitize_filename(model, replacement_text='_')
                style_name = pathvalidate.sanitize_filename(style, replacement_text='_')
                fn = os.path.join(output_folder, f'{model_name}__{style_name}.jpg')
                if os.path.exists(fn):
                    continue
                t_load0 = time.time()
                request(f'/sdapi/v1/checkpoint?sd_model_checkpoint={model}', method='POST')
                loaded = request('/sdapi/v1/checkpoint', method='GET')
                t_load1 = time.time()
                if not loaded or not (model in loaded.get('checkpoint') or model in loaded.get('title') or model in loaded.get('name')):
                    log.error(f' model: error="{model}"')
                    continue
                t_style0 = time.time()
                params = { 'styles': [style] }
                for k, v in args.items():
                    params[k] = v
                log.info(f' style: n={s+1}/{len(styles)} name="{style}" args={params} fn="{fn}"')
                data = request('/sdapi/v1/txt2img', params)
                t_style1 = time.time()
                if 'images' in data and len(data['images']) > 0:
                    idx_style += 1
                    idx_images += 1
                    b64 = data['images'][0].split(',',1)[0]
                    image = Image.open(io.BytesIO(base64.b64decode(b64)))
                    info = data['info']
                    log.info(f' image: size={image.width}x{image.height} time={t_style1-t_style0:.2f} info={len(info)}')
                    image.save(fn)
                    write_history(model=model, style=style, image=fn, size=image.size, generate=round(t_style1-t_style0, 3), load=round(t_load1-t_load0, 3), info=info)
                else:
                    log.error(f' model: error="{model}" style="{style}" no image')
            except Exception as e:
                if 'Connection refused' in str(e) or 'RemoteDisconnected' in str(e):
                    log.error('server offline')
                    os._exit(1)
                log.error(f' model: error="{model}" style="{style}" exception="{e}"')
        t_model1 = time.time()
        if idx_style > 0:
            log.info(f'model: name="{model}" images={idx_style} time={t_model1-t_model0:.2f}')
    t_generate1 = time.time()
    if idx_images > 0:
        log.info(f'generate: models={idx_model} images={idx_images} time={t_generate1-t_generate0:.2f}')


if __name__ == "__main__":
    log.info('test-all-models')
    log.info(f'output="{output_folder}"')
    read_history()
    main()
