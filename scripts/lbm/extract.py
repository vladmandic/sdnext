import torch
from PIL import Image
from torchvision import transforms
from modules import devices


def extract_object(birefnet, img):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = img
    input_images = transform_image(image).unsqueeze(0).to(dtype=torch.float32, device=devices.device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
    return image, mask


def resize_and_center_crop(image, target_width, target_height):
    original_width, original_height = image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image
