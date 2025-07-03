"""
Repo: <https://github.com/unity-research/IP-Adapter-Instruct>
Models: <https://huggingface.co/CiaraRowles/IP-Adapter-Instruct/tree/main>
adapter: `sd15`=0.35GB `sdxl`=2.12GB `sd3`=1.56GB
encoder: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`=3.94GB
"""
import os
import importlib
import gradio as gr
from modules import scripts_manager, processing, shared, sd_models, devices


repo = 'https://github.com/vladmandic/IP-Instruct'
repo_id = 'CiaraRowles/IP-Adapter-Instruct'
encoder = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
folder = os.path.join('repositories', 'ip_instruct')


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.lib = None

    def title(self):
        return 'IP Instruct'

    def show(self, is_img2img):
        if shared.cmd_opts.experimental:
            return not is_img2img if shared.native else False
        else:
            return False

    def install(self):
        if not os.path.exists(folder):
            from installer import clone
            clone(repo, folder)
        if self.lib is None:
            self.lib = importlib.import_module('ip_instruct.ip_adapter')


    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/unity-research/IP-Adapter-Instruct">&nbsp IP Adapter Instruct</a><br>')
        with gr.Row():
            query = gr.Textbox(lines=1, label='Query', placeholder='use the composition from the image')
        with gr.Row():
            image = gr.Image(value=None, label='Image', type='pil', width=256, height=256)
        with gr.Row():
            strength = gr.Slider(label="Strength", value=1.0, minimum=0, maximum=2.0, step=0.05)
            tokens = gr.Slider(label="Tokens", value=4, minimum=1, maximum=32, step=1)
        with gr.Row():
            instruct_guidance = gr.Slider(label="Guidance", value=6.0, minimum=1.0, maximum=15.0, step=0.05)
            image_guidance = gr.Slider(label="Guidance", value=0.5, minimum=0, maximum=1.0, step=0.05)
        return [query, image, strength, tokens, instruct_guidance, image_guidance]

    def run(self, p: processing.StableDiffusionProcessing, query, image, strength, tokens, instruct_guidance, image_guidance): # pylint: disable=arguments-differ
        supported_model_list = ['sd', 'sdxl', 'sd3']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'IP-Instruct: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        self.install()
        if self.lib is None:
            shared.log.error('IP-Instruct: failed to import library')
            return None
        self.orig_pipe = shared.sd_model
        if shared.sd_model_type == 'sdxl':
            pipe = self.lib.StableDiffusionXLPipelineExtraCFG
            cls = self.lib.IPAdapterInstructSDXL
            ckpt = "ip-adapter-instruct-sdxl.bin"
        elif shared.sd_model_type == 'sd3':
            pipe = self.lib.StableDiffusion3PipelineExtraCFG
            cls = self.lib.IPAdapter_sd3_Instruct
            ckpt = "ip-adapter-instruct-sd3.bin"
        else:
            pipe = self.lib.StableDiffusionPipelineCFG
            cls = self.lib.IPAdapterInstruct
            ckpt = "ip-adapter-instruct-sd15.bin"

        shared.sd_model = sd_models.switch_pipe(pipe, shared.sd_model)

        import huggingface_hub as hf
        ip_ckpt = hf.hf_hub_download(repo_id=repo_id, filename=ckpt, cache_dir=shared.opts.hfcache_dir)
        ip_model = cls(shared.sd_model, encoder, ip_ckpt, device=devices.device, dtypein=devices.dtype, num_tokens=tokens)
        processing.fix_seed(p)
        shared.log.debug(f'IP-Instruct: class={shared.sd_model.__class__.__name__} wrapper={ip_model.__class__.__name__} encoder={encoder} adapter={ckpt}')
        shared.log.info(f'IP-Instruct: image={image} query="{query}" strength={strength} tokens={tokens} instruct_guidance={instruct_guidance} image_guidance={image_guidance}')

        image_list = ip_model.generate(
            query = query,
            scale = strength,
            instruct_guidance_scale = instruct_guidance,
            image_guidance_scale = image_guidance,

            prompt = p.prompt,
            pil_image = image,
            num_samples = 1,
            num_inference_steps = p.steps,
            seed = p.seed,
            guidance_scale = p.cfg_scale,
            auto_scale = False,
            simple_cfg_mode = False,
        )
        processed = processing.Processed(p, images_list=image_list, seed=p.seed, subseed=p.subseed, index_of_first_image=0) # manually created processed object
        # p.extra_generation_params["IPInstruct"] = f''
        return processed

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, **kwargs): # pylint: disable=unused-argument
        if self.orig_pipe is not None:
            shared.sd_model = self.orig_pipe
        return processed
