from copy import deepcopy
from PIL import Image
import gradio as gr
from modules import scripts_manager, processing, shared, devices, sd_models


birefnet = None
model = None
model_type = ''
repos = {
    'Simple': None,
    'Normals': 'jasperai/LBM_normals',
    'Depth': 'jasperai/LBM_depth',
    'Relighting': 'jasperai/LBM_relighting',
}

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


class Script(scripts_manager.Script):
    def title(self):
        return 'LBM: Latent Bridge Matching'

    def show(self, is_img2img):
        return is_img2img

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/gojasper/LBM">&nbsp LBM: Latent Bridge Matching</a><br>')
        with gr.Row():
            lbm_method = gr.Dropdown(label='LBM Method', choices=['Simple', 'Relighting', 'Normals', 'Depth'], value='Simple', elem_id='lbm_method')
        with gr.Row():
            lbm_composite = gr.Checkbox(label='LBM Composite', value=True, elem_id='lbm_composite')
            lbm_steps = gr.Slider(label='LBM Steps', minimum=1, maximum=20, step=1, value=1, elem_id='lbm_steps')
        with gr.Row():
            bg_image = gr.Image(label='Background image', type='pil', height=512, elem_id='lbm_bg_image')
        return [lbm_method, lbm_composite, lbm_steps, bg_image]

    def load(self, method: str):
        global birefnet, model, model_type # pylint: disable=global-statement
        import torch
        if birefnet is None:
            from transformers import AutoModelForImageSegmentation
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(dtype=torch.float32, device=devices.device)
        if model is None or model_type != method:
            repo_id = repos.get(method, None)
            model_type = method
            if repo_id is not None:
                import huggingface_hub as hf
                repo_file = hf.snapshot_download(repo_id, cache_dir=shared.opts.hfcache_dir)
                from scripts.lbm import get_model # pylint: disable=no-name-in-module
                model = get_model(
                    repo_file,
                    save_dir=None,
                    torch_dtype=devices.dtype,
                    device=devices.device,
                ).to(dtype=devices.dtype, device=devices.device)

    def run(self, p: processing.StableDiffusionProcessing, lbm_method, lbm_composite, lbm_steps, bg_image): # pylint: disable=arguments-differ, unused-argument
        fg_image = getattr(p, 'init_images', None)
        if fg_image is None or len(fg_image) == 0 or bg_image is None:
            shared.log.error('LBM: no init images')
            return None
        else:
            fg_image = fg_image[0]

        from installer import install
        install('lpips')

        from modules import images_sharpfin
        from scripts.lbm import get_model, extract_object, resize_and_center_crop # pylint: disable=no-name-in-module

        ori_h_bg, ori_w_bg = fg_image.size
        ar_bg = ori_h_bg / ori_w_bg
        closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
        dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

        shared.log.info(f'LBM: method={lbm_method} steps={lbm_steps} size={dimensions_bg[0]}x{dimensions_bg[1]}')
        self.load(lbm_method)

        if birefnet:
            birefnet.to(device=devices.device)
        if model:
            model.to(device=devices.device)

        output_image = None
        _, fg_mask = extract_object(birefnet, deepcopy(fg_image))
        fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
        fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
        bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])
        img_pasted = Image.composite(fg_image, bg_image, fg_mask)

        if lbm_method == 'Simple':
            output_image = img_pasted
        else:
            img_pasted_tensor = images_sharpfin.to_tensor(img_pasted).to(device=devices.device, dtype=devices.dtype).unsqueeze(0) * 2 - 1
            batch = { "source_image": img_pasted_tensor }
            z_source = model.vae.encode(batch[model.source_key])
            output_image = model.sample(
                z=z_source,
                num_steps=lbm_steps,
                conditioner_inputs=batch,
                max_samples=1,
            )
            output_image = (output_image[0].clamp(-1, 1).float().cpu() + 1) / 2
            output_image = images_sharpfin.to_pil(output_image)
            if lbm_composite:
                output_image = Image.composite(output_image, bg_image, fg_mask)

        if birefnet:
            birefnet.to(device=devices.cpu)
        if model:
            model.to(device=devices.cpu)

        if output_image is not None:
            output_image.resize((ori_h_bg, ori_w_bg))
            return processing.get_processed(p, [output_image])
        else:
            return processing.Processed(p, [])
