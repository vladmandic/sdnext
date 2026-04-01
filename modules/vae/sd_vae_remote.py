import io
import time
import json
import torch
import requests
from PIL import Image
from safetensors.torch import _tobytes
from modules.logger import log


hf_decode_endpoints = {
    'sd': 'https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud',
    'sdxl': 'https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud',
    'f1': 'https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud',
    'hunyuanvideo': 'https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud',
}
hf_decode_endpoints['pixartalpha'] = hf_decode_endpoints['sd']
hf_decode_endpoints['pixartsigma'] = hf_decode_endpoints['sdxl']
hf_decode_endpoints['hunyuandit'] = hf_decode_endpoints['sdxl']
hf_decode_endpoints['auraflow'] = hf_decode_endpoints['sdxl']
hf_decode_endpoints['omnigen'] = hf_decode_endpoints['sdxl']
hf_decode_endpoints['h1'] = hf_decode_endpoints['f1']
hf_decode_endpoints['chroma'] = hf_decode_endpoints['f1']
hf_decode_endpoints['zimage'] = hf_decode_endpoints['f1']
hf_decode_endpoints['lumina2'] = hf_decode_endpoints['f1']

hf_encode_endpoints = {
    'sd': 'https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud',
    'sdxl': 'https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud',
    'f1': 'https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud',
    'chroma': 'https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud',
}
hf_encode_endpoints['pixartalpha'] = hf_encode_endpoints['sd']
hf_encode_endpoints['pixartsigma'] = hf_encode_endpoints['sdxl']
hf_encode_endpoints['hunyuandit'] = hf_encode_endpoints['sdxl']
hf_encode_endpoints['auraflow'] = hf_encode_endpoints['sdxl']
hf_encode_endpoints['omnigen'] = hf_encode_endpoints['sdxl']
hf_encode_endpoints['h1'] = hf_encode_endpoints['f1']
hf_encode_endpoints['zimage'] = hf_encode_endpoints['f1']
hf_encode_endpoints['lumina2'] = hf_encode_endpoints['f1']

dtypes = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


def remote_decode(latents: torch.Tensor, width: int = 0, height: int = 0, model_type: str | None = None):
    from modules import devices, shared, errors, modelloader
    tensors = []
    content = 0
    model_type = model_type or shared.sd_model_type
    url = hf_decode_endpoints.get(model_type, None)
    if url is None:
        log.error(f'Decode: type="remote" type={model_type} unsuppported')
        return tensors
    t0 = time.time()
    modelloader.hf_login()
    latent_copy = latents.detach().clone().to(device=devices.cpu, dtype=devices.dtype)
    latent_copy = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
    if model_type == 'hunyuanvideo':
        latent_copy = latent_copy.unsqueeze(0) if len(latents.shape) == 4 else latents

    for i in range(latent_copy.shape[0]):
        params = {}
        try:
            latent = latent_copy[i]
            if model_type not in ['f1', 'chroma']:
                latent = latent.unsqueeze(0)
            params = {
                "input_tensor_type": "binary",
                "shape": list(latent.shape),
                "dtype": str(latent.dtype).split(".", maxsplit=1)[-1],
            }
            headers = { "Content-Type": "tensor/binary" }
            if 'video' in model_type:
                params["partial_postprocess"] = False
                params["output_type"] = "pt"
                params["output_tensor_type"] = "binary"
                headers["Accept"] = "tensor/binary"
            elif shared.opts.remote_vae_type == 'png':
                params["image_format"] = "png"
                params["output_type"] = "pil"
                headers["Accept"] = "image/png"
            elif shared.opts.remote_vae_type == 'jpg':
                params["image_format"] = "jpg"
                params["output_type"] = "pil"
                headers["Accept"] = "image/jpeg"
            elif shared.opts.remote_vae_type == 'raw':
                params["partial_postprocess"] = False
                params["output_type"] = "pt"
                params["output_tensor_type"] = "binary"
                headers["Accept"] = "tensor/binary"
            if model_type in {'f1', 'h1', 'zimage', 'lumina2', 'chroma'} and (width > 0) and (height > 0):
                params['width'] = width
                params['height'] = height
            if shared.sd_model.vae is not None and shared.sd_model.vae.config is not None:
                params['scaling_factor'] = shared.sd_model.vae.config.get("scaling_factor", None)
                params['shift_factor'] = shared.sd_model.vae.config.get("shift_factor", None)
            response = requests.post(
                url=url,
                headers=headers,
                params=params,
                data=_tobytes(latent, "tensor"),
                timeout=300,
            )
            if not response.ok:
                log.error(f'Decode: type="remote" model={model_type} code={response.status_code} shape={latent.shape} url="{url}" args={params} headers={response.headers} response={response.json()}')
            else:
                content += len(response.content)
                if shared.opts.remote_vae_type == 'raw' or 'video' in model_type:
                    shape = json.loads(response.headers["shape"])
                    dtype = response.headers["dtype"]
                    tensor = torch.frombuffer(bytearray(response.content), dtype=dtypes[dtype]).reshape(shape)
                    tensors.append(tensor)
                elif shared.opts.remote_vae_type == 'jpg' or shared.opts.remote_vae_type == 'png':
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    tensors.append(image)
        except Exception as e:
            log.error(f'Decode: type="remote" model={model_type} {e}')
            errors.display(e, 'VAE')
    if len(tensors) > 0 and shared.opts.remote_vae_type == 'raw':
        tensors = torch.cat(tensors, dim=0)
    t1 = time.time()
    log.debug(f'Decode: type="remote" model={model_type} mode={shared.opts.remote_vae_type} args={params} bytes={content} time={t1-t0:.3f}s')
    return tensors


def remote_encode(images: list[Image.Image], model_type: str | None = None):
    from diffusers.utils import remote_utils
    from modules import devices, shared, errors, modelloader
    if not shared.opts.remote_vae_encode:
        return images
    tensors = []
    model_type = model_type or shared.sd_model_type
    url = hf_encode_endpoints.get(model_type, None)
    if url is None:
        log.error(f'Decode: type="remote" type={model_type} unsuppported')
        return images
    t0 = time.time()
    modelloader.hf_login()

    if isinstance(images, Image.Image):
        images = [images]
    for init_image in images:
        try:
            init_latent = remote_utils.remote_encode(
                endpoint=url,
                image=init_image,
                scaling_factor = shared.sd_model.vae.config.get("scaling_factor", None),
                shift_factor = shared.sd_model.vae.config.get("shift_factor", None),
            )
            tensors.append(init_latent)
        except Exception as e:
            log.error(f'Encode: type="remote" model={model_type} {e}')
            errors.display(e, 'VAE')

    if len(tensors) > 0 and torch.is_tensor(tensors[0]):
        tensors = torch.cat(tensors, dim=0)
        tensors = tensors.to(dtype=devices.dtype)
    else:
        return images
    t1 = time.time()
    log.debug(f'Encode: type="remote" model={model_type} mode={shared.opts.remote_vae_type} image={images} latent={tensors.shape} time={t1-t0:.3f}s')
    return tensors
