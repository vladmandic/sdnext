model = None
repo_id = "egeorcun/lucida"


def remove(image):
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    from modules import devices

    global model # pylint: disable=global-statement

    if model is None:
        model = AutoModelForImageSegmentation.from_pretrained(repo_id,
                                                              trust_remote_code=True,
                                                              dtype=torch.float32,
                                                             )
        model.eval()

    t = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = model.to(device=devices.device)
    with devices.inference_context():
        input_tensor = t(image).unsqueeze(0).to(devices.device)
        preds = model(input_tensor)[-1].sigmoid()
        alpha = transforms.functional.resize(preds[0], image.size[::-1]).squeeze(0)
        alpha = alpha.detach().cpu().numpy()
    model = model.to(device=devices.cpu)

    rgba = image.copy()
    rgba.putalpha(Image.fromarray((255.0 * alpha).astype("uint8")))

    if rgba is None:
        return image
    return rgba
