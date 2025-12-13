import io
import os
import time
import contextlib
import gradio as gr
from PIL import Image
from modules import shared, devices, errors, scripts_manager, processing, processing_helpers, sd_models


debug = os.environ.get('SD_PULID_DEBUG', None) is not None
direct = False
registered = False
uploaded_images = []


class Script(scripts_manager.Script):
    def __init__(self):
        self.pulid = None
        self.cache = None
        self.preprocess = 0
        super().__init__()
        self.register() # pulid is script with processing override so xyz doesnt execute

    def title(self):
        return 'PuLID: ID Customization'

    def show(self, _is_img2img):
        return True

    def dependencies(self):
        from installer import installed, install, install_insightface
        if not installed('insightface', reload=False, quiet=True):
            install_insightface()
        if not installed('torchdiffeq'):
            install('torchdiffeq')

    def register(self): # register xyz grid elements
        global registered # pylint: disable=global-statement
        if registered:
            return
        registered = True
        def apply_field(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                setattr(p, field, x)
                self.run(p)
            return fun

        import sys
        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k]
        if xyz_classes and len(xyz_classes) > 0:
            xyz_classes = xyz_classes[0]
            options = [
                xyz_classes.AxisOption("[PuLID] Strength", float, apply_field("pulid_strength")),
                xyz_classes.AxisOption("[PuLID] Zero", int, apply_field("pulid_zero")),
                xyz_classes.AxisOption("[PuLID] Ortho", str, apply_field("pulid_ortho"), choices=lambda: ['off', 'v1', 'v2']),
            ]
            for option in options:
                if option not in xyz_classes.axis_options:
                    xyz_classes.axis_options.append(option)


    def decode_image(self,  b64):
        from modules.api.api import decode_base64_to_image
        return decode_base64_to_image(b64)

    def load_images(self, files):
        uploaded_images.clear()
        for file in files or []:
            try:
                if isinstance(file, str):
                    image = self.decode_image(file)
                elif isinstance(file, Image.Image):
                    image = file
                elif isinstance(file, dict) and 'name' in file:
                    image = Image.open(file['name']) # _TemporaryFileWrapper from gr.Files
                elif hasattr(file, 'name'):
                    image = Image.open(file.name) # _TemporaryFileWrapper from gr.Files
                else:
                    raise ValueError(f'IP adapter unknown input: {file}')
                uploaded_images.append(image)
            except Exception as e:
                shared.log.warning(f'IP adapter failed to load image: {e}')
        return gr.update(value=uploaded_images, visible=len(uploaded_images) > 0)

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/ToTheBeginning/PuLID">&nbsp PuLID: Pure and Lightning ID Customization</a><br>')
        with gr.Row():
            strength = gr.Slider(label = 'Strength', value = 0.8, minimum = 0, maximum = 1, step = 0.01)
            zero = gr.Slider(label = 'Zero', value = 20, minimum = 0, maximum = 80, step = 1)
        with gr.Row():
            sampler = gr.Dropdown(label="Sampler", value='dpmpp_sde', choices=['dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2s_ancestral', 'dpmpp_3m_sde', 'dpmpp_sde', 'euler', 'euler_ancestral'])
            ortho = gr.Dropdown(label="Ortho", choices=['off', 'v1', 'v2'], value='v2')
        with gr.Row():
            version = gr.Dropdown(label="Version", value='v1.1', choices=['v1.0', 'v1.1'])
        with gr.Row():
            restore = gr.Checkbox(label='Restore pipe on end', value=False)
            offload = gr.Checkbox(label='Offload face module', value=True)
        with gr.Row():
            files = gr.File(label='Input images', file_count='multiple', file_types=['image'], interactive=True, height=100)
        with gr.Row():
            gallery = gr.Gallery(show_label=False, value=[], visible=False, container=False, rows=1)
        files.change(fn=self.load_images, inputs=[files], outputs=[gallery])
        return [gallery, strength, zero, sampler, ortho, restore, offload, version]

    def run(
            self,
            p: processing.StableDiffusionProcessing,
            gallery: list = [],
            strength: float = 0.8,
            zero: int = 20,
            sampler: str = 'dpmpp_sde',
            ortho: str = 'v2',
            restore: bool = False,
            offload: bool = True,
            version: str = 'v1.1'
        ): # pylint: disable=arguments-differ, unused-argument
        images = []
        import numpy as np
        try:
            if gallery is None or (isinstance(gallery, list) and len(gallery) == 0):
                images = getattr(p, 'pulid_images', uploaded_images)
                images = [self.decode_image(image) if isinstance(image, str) else image for image in images]
            elif isinstance(gallery[0], dict):
                images = [Image.open(f['name']) for f in gallery]
            elif isinstance(gallery, str):
                images = [self.decode_image(gallery)]
            elif isinstance(gallery[0], str):
                images = [self.decode_image(f) for f in gallery]
            else:
                images = gallery
            images = [np.array(image) for image in images]
        except Exception as e:
            shared.log.error(f'PuLID: failed to load images: {e}')
            return None
        if len(images) == 0:
            shared.log.error('PuLID: no images')
            return None

        supported_model_list = ['sdxl', 'f1']
        if MODELDATA.sd_model_type not in supported_model_list:
            shared.log.error(f'PuLID: class={MODELDATA.sd_model.__class__.__name__} model={MODELDATA.sd_model_type} required={supported_model_list}')
            return None
        if self.pulid is None:
            self.dependencies()
            try:
                from scripts import pulid # pylint: disable=redefined-outer-name
                self.pulid = pulid
                from diffusers import pipelines
                pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["pulid"] = pulid.StableDiffusionXLPuLIDPipeline
                pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["pulid"] = pulid.StableDiffusionXLPuLIDPipelineImage
                pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["pulid"] = pulid.StableDiffusionXLPuLIDPipelineInpaint
            except Exception as e:
                shared.log.error(f'PuLID: failed to import library: {e}')
                return None
            if self.pulid is None:
                shared.log.error('PuLID: failed to load PuLID library')
                return None

        try:
            images = [self.pulid.resize(image, 1024) for image in images]
        except Exception as e:
            shared.log.error(f'PuLID: failed to resize images: {e}')
            return None

        if p.batch_size > 1:
            shared.log.warning('PuLID: batch size not supported')
            p.batch_size = 1

        sdp = shared.opts.cross_attention_optimization == "Scaled-Dot-Product"
        strength = getattr(p, 'pulid_strength', strength)
        zero = getattr(p, 'pulid_zero', zero)
        ortho = getattr(p, 'pulid_ortho', ortho)
        sampler = getattr(p, 'pulid_sampler', sampler)
        restore = getattr(p, 'pulid_restore', restore)
        p.pulid_restore = restore

        if MODELDATA.sd_model_type == 'sdxl' and not hasattr(MODELDATA.sd_model, 'pipe'):
            try:
                stdout = io.StringIO()
                ctx = contextlib.nullcontext() if debug else contextlib.redirect_stdout(stdout)
                with ctx:
                    MODELDATA.sd_model = self.pulid.StableDiffusionXLPuLIDPipeline(
                        pipe=MODELDATA.sd_model,
                        device=devices.device,
                        dtype=devices.dtype,
                        providers=devices.onnx,
                        offload=offload,
                        version=version,
                        sdp=sdp,
                        cache_dir=shared.opts.hfcache_dir,
                    )
                MODELDATA.sd_model.no_recurse = True
                sd_models.copy_diffuser_options(MODELDATA.sd_model, MODELDATA.sd_model.pipe)
                sd_models.move_model(MODELDATA.sd_model, devices.device) # move pipeline to device
                sd_models.set_diffuser_options(MODELDATA.sd_model, vae=None, op='model')
                # shared.sd_model.hack_unet_attn_layers(shared.sd_model.pipe.unet) # reapply attention layers
                devices.torch_gc()
            except Exception as e:
                shared.log.error(f'PuLID: failed to create pipeline: {e}')
                errors.display(e, 'PuLID')
                return None
        elif MODELDATA.sd_model_type == 'f1':
            MODELDATA.sd_model = self.pulid.apply_flux(MODELDATA.sd_model)

        if MODELDATA.sd_model_type == 'sdxl':
            processed = self.run_sdxl(p, images, strength, zero, sampler, ortho, restore, offload, version)
        elif MODELDATA.sd_model_type == 'f1':
            processed = self.run_flux(p, images, strength)
        else:
            shared.log.error(f'PuLID: class={MODELDATA.sd_model.__class__.__name__} model={MODELDATA.sd_model_type} required={supported_model_list}')
            processed = None
        return processed

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=unused-argument
        _strength, _zero, _sampler, _ortho, _gallery, restore, _offload, _version = args
        if MODELDATA.sd_model_type == "sdxl" and hasattr(MODELDATA.sd_model, 'pipe'):
            restore = getattr(p, 'pulid_restore', restore)
            if restore:
                if hasattr(MODELDATA.sd_model, 'app'):
                    MODELDATA.sd_model.app = None
                    MODELDATA.sd_model.ip_adapter = None
                    MODELDATA.sd_model.face_helper = None
                    MODELDATA.sd_model.clip_vision_model = None
                    MODELDATA.sd_model.handler_ante = None
                MODELDATA.sd_model = MODELDATA.sd_model.pipe
                devices.torch_gc(force=True, reason='pulid')
            shared.log.debug(f'PuLID complete: class={MODELDATA.sd_model.__class__.__name__} preprocess={self.preprocess:.2f} pipe={"restore" if restore else "cache"}')
        if MODELDATA.sd_model_type == "f1":
            restore = getattr(p, 'pulid_restore', restore)
            if restore:
                MODELDATA.sd_model = self.pulid.unapply_flux(MODELDATA.sd_model)
                devices.torch_gc(force=True, reason='pulid')
            shared.log.debug(f'PuLID complete: class={MODELDATA.sd_model.__class__.__name__} pipe={"restore" if restore else "cache"}')
        return processed

    def run_sdxl(self, p: processing.StableDiffusionProcessing, images: list, strength: float, zero: int, sampler: str, ortho: str, restore: bool, offload: bool, version: str):
        sampler_fn = getattr(self.pulid.sampling, f'sample_{sampler}', None)
        if sampler_fn is None:
            sampler_fn = self.pulid.sampling.sample_dpmpp_2m_sde
        MODELDATA.sd_model.sampler = sampler_fn
        shared.log.info(f'PuLID: class={MODELDATA.sd_model.__class__.__name__} version="{version}" strength={strength} zero={zero} ortho={ortho} sampler={sampler_fn} images={[i.shape for i in images]} offload={offload} restore={restore}')
        self.pulid.attention.NUM_ZERO = zero
        self.pulid.attention.ORTHO = ortho == 'v1'
        self.pulid.attention.ORTHO_v2 = ortho == 'v2'
        MODELDATA.sd_model.debug_img_list = []

        # get id embedding used for attention
        t0 = time.time()
        uncond_id_embedding, id_embedding = MODELDATA.sd_model.get_id_embedding(images)
        if offload:
            devices.torch_gc()
        t1 = time.time()
        self.preprocess = t1-t0

        p.seed = processing_helpers.get_fixed_seed(p.seed)
        if direct: # run pipeline directly
            jobid = shared.state.begin('PuLID')
            processing.fix_seed(p)
            p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
            p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
            shared.prompt_styles.apply_styles_to_extra(p)
            p.styles = []
            with devices.inference_context():
                output = MODELDATA.sd_model(
                    prompt=p.prompt,
                    negative_prompt=p.negative_prompt,
                    width=p.width,
                    height=p.height,
                    seed=p.seed,
                    num_inference_steps=p.steps,
                    guidance_scale=p.cfg_scale,
                    id_embedding=id_embedding,
                    uncond_id_embedding=uncond_id_embedding,
                    id_scale=strength,
                    )[0]
            info = processing.create_infotext(p)
            processed = processing.get_processed(p, [output], info=info)
            shared.state.end(jobid)
        else: # let processing run the pipeline
            p.task_args['id_embedding'] = id_embedding
            p.task_args['uncond_id_embedding'] = uncond_id_embedding
            p.task_args['id_scale'] = strength
            p.extra_generation_params["PuLID"] = f'Strength={strength} Zero={zero} Ortho={ortho}'
            p.extra_generation_params["Sampler"] = sampler
            if getattr(p, 'xyz', False): # xyz will run its own processing
                return None
            processed: processing.Processed = processing.process_images(p) # runs processing using main loop

        # interim = [Image.fromarray(img) for img in shared.sd_model.debug_img_list]
        # shared.log.debug(f'PuLID: time={t1-t0:.2f}')
        return processed

    def run_flux(self, p: processing.StableDiffusionProcessing, images: list, strength: float):
        image = Image.fromarray(images[0]) # takes single pil image
        p.task_args['id_image'] = image
        p.task_args['id_weight'] = strength
        shared.log.info(f'PuLID: class={MODELDATA.sd_model.__class__.__name__} strength={strength} image={image}')
        p.extra_generation_params["PuLID"] = f'Strength={strength}'

        processed: processing.Processed = processing.process_images(p) # runs processing using main loop
        return processed
