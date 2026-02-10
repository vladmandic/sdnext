import gradio as gr
from PIL import Image
from modules import scripts_manager, processing, shared, sd_models, devices, images


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.orig_vae = None
        self.vae = None

    def title(self):
        return 'PixelSmith'

    def show(self, is_img2img):
        return True

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/Thanos-DB/Pixelsmith">&nbsp PixelSmith</a><br>')
        with gr.Row():
            slider = gr.Slider(label="Slider", value=20, minimum=0, maximum=100, step=1)
        return [slider]

    def encode(self, p: processing.StableDiffusionProcessing, image: Image.Image):
        if image is None:
            return None
        from modules import images_sharpfin
        if p.width is None or p.width == 0:
            p.width = int(8 * (image.width * p.scale_by // 8))
        if p.height is None or p.height == 0:
            p.height = int(8 * (image.height * p.scale_by // 8))
        image = images.resize_image(p.resize_mode, image, p.width, p.height, upscaler_name=p.resize_name, context=p.resize_context)
        tensor = images_sharpfin.to_tensor(image).unsqueeze(0).to(device=devices.device, dtype=devices.dtype)
        tensor = 2.0 * tensor - 1.0
        with devices.inference_context():
            latent = shared.sd_model.vae.tiled_encode(tensor)
            latent = shared.sd_model.vae.config.scaling_factor * latent.latent_dist.sample()
        shared.log.info(f'PixelSmith encode: image={image} latent={latent.shape} width={p.width} height={p.height} vae={shared.sd_model.vae.__class__.__name__}')
        return latent


    def run(self, p: processing.StableDiffusionProcessing, slider: int = 20): # pylint: disable=arguments-differ
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'PixelSmith: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
        from scripts.pixelsmith import PixelSmithXLPipeline, PixelSmithVAE # pylint: disable=no-name-in-module
        self.orig_pipe = shared.sd_model
        self.orig_vae = shared.sd_model.vae
        if self.vae is None:
            self.vae = PixelSmithVAE.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=devices.dtype).to(devices.device)
        shared.sd_model = sd_models.switch_pipe(PixelSmithXLPipeline, shared.sd_model)
        shared.sd_model.vae = self.vae
        shared.sd_model.vae.enable_tiling()
        p.extra_generation_params["PixelSmith"] = f'Slider={slider}'
        p.sampler_name = 'DDIM'
        p.task_args['slider'] = slider
        # p.task_args['output_type'] = 'pil'
        if hasattr(p, 'init_images') and p.init_images is not None and len(p.init_images) > 0:
            p.task_args['image'] = self.encode(p, p.init_images[0])
            p.init_images = None
        shared.log.info(f'PixelSmith apply: slider={slider} class={shared.sd_model.__class__.__name__} vae={shared.sd_model.vae.__class__.__name__}')
        # processed = processing.process_images(p)

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, slider): # pylint: disable=unused-argument
        if self.orig_pipe is None:
            return processed
        if shared.sd_model.__class__.__name__ == 'PixelSmithXLPipeline':
            shared.sd_model = self.orig_pipe
            shared.sd_model.vae = self.orig_vae
        self.orig_pipe = None
        self.orig_vae = None
        return processed
