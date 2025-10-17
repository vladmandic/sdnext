import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from .src.core.generation import generation_loop
from .src.core.model_manager import configure_runner



device = 'cuda'
dtype = torch.bfloat16
model_dir = 'seedvr2_models'
model = 'seedvr2_ema_3b_fp16.safetensors'
resolution = 1024
seed = 100
cfg = 1.0
input_image = '/home/vlado/generative/Samples/cutie-512.png'

to_pil = ToPILImage()
runner = None
loaded_model = None


def upscale_image(model_name:str, image_path:str):
    global runner, loaded_model
    if (runner is None) or (loaded_model != model_name):
        runner = configure_runner(model_name, model_dir, device=device, dtype=dtype)
        loaded_model = model_name

    image = Image.open(image_path).convert("RGB")
    image_tensor = np.array(image)
    image_tensor = torch.from_numpy(image_tensor).to(device=device, dtype=dtype).unsqueeze(0) / 255.0

    result_tensor = generation_loop(
        runner=runner,
        images=image_tensor,
        cfg_scale=cfg,
        seed=seed,
        res_w=resolution,
        batch_size=1,
        temporal_overlap=0,
        device=device,
    )
    image = to_pil(result_tensor.squeeze().permute((2, 0, 1)))

    output_path = os.path.join('/tmp', os.path.basename(image_path))

    image.save(output_path, quality=95)
    return image


if __name__ == "__main__":
    output_image = upscale_image(model, input_image)
    print('input:', input_image)
    print('output:', output_image)
