import os
import cv2
import torch
import numpy as np
import huggingface_hub as hf
from PIL import Image
from modules import processing, shared, devices
from modules.logger import log

RESWAPPER_REPO = 'somanchiu/reswapper'
RESWAPPER_MODELS = {
    "ReSwapper 256 0.2": "reswapper_256-1567500.pth",
    "ReSwapper 256 0.1": "reswapper_256-1399500.pth",
    "ReSwapper 128 0.2": "reswapper-429500.pth",
    "ReSwapper 128 0.1": "reswapper-1019500.pth",
}
reswapper_model = None
reswapper_name = None
debug = log.trace if os.environ.get("SD_FACE_DEBUG", None) is not None else lambda *args, **kwargs: None
dtype = devices.dtype

def get_model(model_name: str):
    global reswapper_model, reswapper_name # pylint: disable=global-statement
    if reswapper_model is None or reswapper_name != model_name:
        try:
            fn = RESWAPPER_MODELS.get(model_name)
            url = hf.hf_hub_download(repo_id=RESWAPPER_REPO, filename=fn, repo_type="model", cache_dir=shared.opts.hfcache_dir)
            from modules.face.reswapper_model import ReSwapperModel
            reswapper_model = ReSwapperModel()
            reswapper_model.load_state_dict(torch.load(url, map_location='cpu'), strict=False)
            reswapper_model = reswapper_model.to(device=devices.device, dtype=dtype)
            reswapper_model.eval()
            reswapper_name = model_name
            log.info(f'ReSwapper: model="{model_name}" url="{url}" cls={reswapper_model.__class__.__name__}')
            if reswapper_model is None:
                log.error(f'ReSwapper: model="{model_name}" fn="{fn}" url="{url}" failed to load model')
            return reswapper_model
        except Exception as e:
            log.error(f'ReSwapper: model="{model_name}" fn="{fn}" url="{url}" {e}')
    return reswapper_model


def reswapper(
    p: processing.StableDiffusionProcessing,
    app,
    source_images: list[Image.Image],
    target_images: list[Image.Image],
    model_name: str,
    original: bool,
):
    from modules.face import reswapper_utils as utils
    if source_images is None or len(source_images) == 0:
        log.warning('ReSwapper: no input images')
        return None

    processed_images = []
    if original:
        processed_images += source_images

    model = get_model(model_name)
    if model is None:
        return source_images
    model = model.to(device=devices.device)

    i = 0
    for x, image in enumerate(source_images):
        image = image.convert('RGB')
        source_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        source_faces = app.get(source_np)
        if len(source_faces) == 0:
            log.error(f"ReSwapper: image={x+1} no source faces found")
            return source_images
        if len(source_faces) != len(target_images):
            log.warning(f"ReSwapper: image={x+1} source-faces={len(source_faces)} target-images={len(target_images)}")
        for y, source_face in enumerate(source_faces):
            target_image = target_images[y] if y < len(target_images) else target_images[-1]
            target_image = target_image.convert('RGB')
            target_np = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
            target_faces = app.get(target_np)
            if len(target_faces) != 1:
                log.error(f"ReSwapper: image={x+1} source-faces={y+1} target-faces={len(target_faces)} must be exactly one")
                return source_images
            target_face = target_faces[0]
            source_str = f'score:{source_face.det_score:.2f} gender:{"female" if source_face.gender==0 else "male"} age:{source_face.age}'
            target_str = f'score:{target_face.det_score:.2f} gender:{"female" if target_face.gender==0 else "male"} age:{target_face.age}'
            log.debug(f'ReSwapper image={x+1} face={y+1} source="{source_str}" target="{target_str}"')

            source_latent = utils.getLatent(source_face)
            source_tensor = torch.from_numpy(source_latent).to(device=devices.device, dtype=dtype)

            resolution = 256 if '256' in model_name else 128
            target_np = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
            target_aligned, M = utils.norm_crop2(target_np, target_face.kps, resolution)
            target_blob = utils.getBlob(target_aligned, (resolution, resolution))
            target_tensor = torch.from_numpy(target_blob).to(device=devices.device, dtype=dtype)

            with devices.inference_context():
                swapped_tensor = model(target_tensor, source_tensor)
                swapped_tensor = swapped_tensor.float()

            swapped_face = (swapped_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
            swapped_np = utils.blend_swapped_image(swapped_face, source_np, M)
            swapped_image = Image.fromarray(cv2.cvtColor(swapped_np, cv2.COLOR_BGR2RGB))
            processed_images.append(swapped_image)
            i += 1

    p.extra_generation_params['ReSwapper'] = f'faces={i}'
    devices.torch_gc()

    return processed_images
