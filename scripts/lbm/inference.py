import logging
import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor
from .lbm import LBMModel
from modules import devices


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}


@torch.no_grad()
def evaluate(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):
    """
    Evaluate the model on an image coming from the source distribution and generate a new image from the target distribution.

    Args:
        model (LBMModel): The model to evaluate.
        source_image (PIL.Image.Image): The source image to evaluate the model on.
        num_sampling_steps (int): The number of sampling steps to use for the model.

    Returns:
        PIL.Image.Image: The generated image.
    """

    ori_h_bg, ori_w_bg = source_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    source_dimensions = ASPECT_RATIOS[closest_ar_bg]

    source_image = source_image.resize(source_dimensions)

    img_pasted_tensor = ToTensor()(source_image).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.to(dtype=devices.dtype, device=devices.device),
    }

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)
    output_image.resize((ori_h_bg, ori_w_bg))

    return output_image
