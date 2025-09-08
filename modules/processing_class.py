import os
import sys
import inspect
import hashlib
from typing import Any, Dict, List
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageOps
from modules import shared, images, scripts_manager, masking, sd_models, sd_vae, processing_helpers


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None


@dataclass(repr=False)
class StableDiffusionProcessing:
    def __init__(self,
                 sd_model_checkpoint: str = None, # # used only to set sd_model
                 sd_model=None, # pylint: disable=unused-argument # local instance of sd_model
                 # base params
                 prompt: str = "",
                 negative_prompt: str = "",
                 seed: int = -1,
                 subseed: int = -1,
                 subseed_strength: float = 0,
                 seed_resize_from_h: int = -1,
                 seed_resize_from_w: int = -1,
                 batch_size: int = 1,
                 n_iter: int = 1,
                 steps: int = 20,
                 clip_skip: int = 1,
                 width: int = 1024,
                 height: int = 1024,
                 # samplers
                 sampler_index: int = None, # pylint: disable=unused-argument # used only to set sampler_name
                 sampler_name: str = None,
                 hr_sampler_name: str = None,
                 eta: float = None,
                 # guidance
                 cfg_scale: float = 6.0,
                 cfg_end: float = 1,
                 diffusers_guidance_rescale: float = 0.0,
                 pag_scale: float = 0.0,
                 pag_adaptive: float = 0.5,
                 # styles
                 styles: List[str] = [],
                 # vae
                 tiling: bool = False,
                 vae_type: str = 'Full',
                 # other
                 hidiffusion: bool = False,
                 do_not_reload_embeddings: bool = False,
                 restore_faces: bool = False,
                 # detailer
                 detailer_enabled: bool = False,
                 detailer_prompt: str = '',
                 detailer_negative: str = '',
                 detailer_steps: int = 10,
                 detailer_strength: float = 0.3,
                 # hdr corrections
                 hdr_mode: int = 0,
                 hdr_brightness: float = 0,
                 hdr_color: float = 0,
                 hdr_sharpen: float = 0,
                 hdr_clamp: bool = False,
                 hdr_boundary: float = 4.0,
                 hdr_threshold: float = 0.95,
                 hdr_maximize: bool = False,
                 hdr_max_center: float = 0.6,
                 hdr_max_boundary: float = 1.0,
                 hdr_color_picker: str = None,
                 hdr_tint_ratio: float = 0,
                 # img2img
                 init_images: list = None,
                 denoising_strength: float = 0.3,
                 image_cfg_scale: float = None,
                 initial_noise_multiplier: float = None, # pylint: disable=unused-argument # a1111 compatibility
                 # resize
                 scale_by: float = 1,
                 selected_scale_tab: int = 0, # pylint: disable=unused-argument # a1111 compatibility
                 resize_mode: int = 0,
                 resize_name: str = 'None',
                 resize_context: str = 'None',
                 width_before:int = 0,
                 width_after:int = 0,
                 width_mask:int = 0,
                 height_before:int = 0,
                 height_after:int = 0,
                 height_mask:int = 0,
                 resize_name_before: str = 'None',
                 resize_name_after: str = 'None',
                 resize_name_mask: str = 'None',
                 resize_mode_before: int = 0,
                 resize_mode_after: int = 0,
                 resize_mode_mask: int = 0,
                 resize_context_before: str = 'None',
                 resize_context_after: str = 'None',
                 resize_context_mask: str = 'None',
                 selected_scale_tab_before: int = 0,
                 selected_scale_tab_after: int = 0,
                 selected_scale_tab_mask: int = 0,
                 scale_by_before: float = 1,
                 scale_by_after: float = 1,
                 scale_by_mask: float = 1,
                 # inpaint
                 mask: Any = None,
                 latent_mask: Any = None,
                 mask_for_overlay: Any = None,
                 mask_blur: int = 4,
                 paste_to: Any = None,
                 inpainting_fill: int = 1, # obsolete
                 inpaint_full_res: bool = False,
                 inpaint_full_res_padding: int = 0,
                 inpainting_mask_invert: int = 0,
                 overlay_images: Any = None,
                 # refiner
                 enable_hr: bool = False,
                 firstphase_width: int = 0,
                 firstphase_height: int = 0,
                 hr_scale: float = 2.0,
                 hr_force: bool = False,
                 hr_resize_mode: int = 0,
                 hr_resize_context: str = 'None',
                 hr_upscaler: str = None,
                 hr_second_pass_steps: int = 0,
                 hr_resize_x: int = 0,
                 hr_resize_y: int = 0,
                 hr_denoising_strength: float = 0.0,
                 refiner_steps: int = 5,
                 refiner_start: float = 0,
                 refiner_prompt: str = '',
                 refiner_negative: str = '',
                 hr_refiner_start: float = 0,
                 # prompt enhancer
                 enhance_prompt: bool = False,
                 # save options
                 outpath_samples=None,
                 outpath_grids=None,
                 do_not_save_samples: bool = False,
                 do_not_save_grid: bool = False,
                 # xyz flag
                 xyz: bool = False,
                 # scripts
                 script_args: list = [],
                 # overrides
                 override_settings: Dict[str, Any] = {},
                 override_settings_restore_afterwards: bool = True,
                 # metadata
                 # extra_generation_params: Dict[Any, Any] = {},
                 # task_args: Dict[str, Any] = {},
                 # ops: List[str] = [],
                 **kwargs,
                ):

        for k, v in kwargs.items():
            setattr(self, k, v)

        # extra args set by processing loop
        self.task_args = {}
        self.extra_generation_params = {}

        # state items
        self.state: str = ''
        self.ops = []
        self.skip = []
        self.color_corrections = []
        self.is_control = False
        self.is_hr_pass = False
        self.is_refiner_pass = False
        self.is_api = False
        self.scheduled_prompt = False
        self.enhance_prompt = enhance_prompt
        self.prompt_embeds = []
        self.positive_pooleds = []
        self.negative_embeds = []
        self.negative_pooleds = []
        self.prompt_attention_masks = []
        self.negative_prompt_attention_masks = []
        self.disable_extra_networks = False
        self.iteration = 0
        self.network_data = {}

        # initializers
        self.prompt = prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.clip_skip = clip_skip
        self.width = width
        self.height = height
        self.negative_prompt = negative_prompt
        self.styles = styles
        self.tiling = tiling
        self.vae_type = vae_type
        self.hidiffusion = hidiffusion
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.detailer_enabled = detailer_enabled
        self.detailer_prompt = detailer_prompt
        self.detailer_negative = detailer_negative
        self.detailer_steps = detailer_steps
        self.detailer_strength = detailer_strength
        self.restore_faces = restore_faces
        self.init_images = init_images
        self.resize_mode = resize_mode
        self.resize_name = resize_name
        self.resize_context = resize_context
        self.denoising_strength = denoising_strength
        self.image_cfg_scale = image_cfg_scale
        self.scale_by = scale_by
        self.mask = mask
        self.image_mask = mask # TODO processing: remove duplicate mask params
        self.latent_mask = latent_mask
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.overlay_images = overlay_images
        self.enable_hr = enable_hr
        self.firstphase_width = firstphase_width
        self.firstphase_height = firstphase_height
        self.hr_scale = hr_scale
        self.hr_force = hr_force
        self.hr_resize_mode = hr_resize_mode
        self.hr_resize_context = hr_resize_context
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_upscale_to_x = hr_resize_x
        self.hr_upscale_to_y = hr_resize_y
        self.hr_denoising_strength = hr_denoising_strength
        self.refiner_steps = refiner_steps
        self.refiner_start = refiner_start
        self.refiner_prompt = refiner_prompt
        self.refiner_negative = refiner_negative
        self.hr_refiner_start = hr_refiner_start
        self.outpath_samples = outpath_samples
        self.outpath_grids = outpath_grids
        self.do_not_save_samples = do_not_save_samples
        self.do_not_save_grid = do_not_save_grid
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.eta = eta
        self.cfg_scale = cfg_scale
        self.cfg_end = cfg_end
        self.diffusers_guidance_rescale = diffusers_guidance_rescale
        self.pag_scale = pag_scale
        self.pag_adaptive = pag_adaptive
        self.selected_scale_tab = selected_scale_tab
        self.mask_for_overlay = mask_for_overlay
        self.paste_to = paste_to
        self.init_latent = None
        self.width_before = width_before
        self.width_after = width_after
        self.width_mask = width_mask
        self.height_before = height_before
        self.height_after = height_after
        self.height_mask = height_mask
        self.resize_name_before = resize_name_before
        self.resize_name_after = resize_name_after
        self.resize_name_mask = resize_name_mask
        self.resize_mode_before = resize_mode_before
        self.resize_mode_after = resize_mode_after
        self.resize_mode_mask = resize_mode_mask
        self.resize_context_before = resize_context_before
        self.resize_context_after = resize_context_after
        self.resize_context_mask = resize_context_mask
        self.selected_scale_tab_before = selected_scale_tab_before
        self.selected_scale_tab_after = selected_scale_tab_after
        self.selected_scale_tab_mask = selected_scale_tab_mask
        self.scale_by_before = scale_by_before
        self.scale_by_after = scale_by_after
        self.scale_by_mask = scale_by_mask

        # special handled items
        if firstphase_width != 0 or firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = firstphase_width
            self.height = firstphase_height
        self.sampler_name = sampler_name or processing_helpers.get_sampler_name(sampler_index, img=True)
        self.hr_sampler_name: str = hr_sampler_name if hr_sampler_name != 'Same as primary' else self.sampler_name
        self.inpaint_full_res = inpaint_full_res if isinstance(inpaint_full_res, bool) else self.inpaint_full_res
        self.inpaint_full_res = inpaint_full_res != 0 if isinstance(inpaint_full_res, int) else self.inpaint_full_res
        try:
            self.override_settings = {k: v for k, v in (override_settings or {}).items() if k not in shared.restricted_opts}
        except Exception as e:
            shared.log.error(f'Override: {override_settings} {e}')
            self.override_settings = {}

        # null items initialized later
        self.prompts = None
        self.negative_prompts = None
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None

        # a1111 compatibility items
        self.seed_enable_extras: bool = True
        self.is_using_inpainting_conditioning = False # a111 compatibility
        self.batch_index = 0
        self.refiner_switch_at = 0
        self.hr_prompt = ''
        self.all_hr_prompts = []
        self.hr_negative_prompt = ''
        self.all_hr_negative_prompts = []
        self.comments = {}
        self.sampler = None
        self.nmask = None
        self.initial_noise_multiplier = initial_noise_multiplier or shared.opts.initial_noise_multiplier
        self.image_conditioning = None
        self.prompt_for_display: str = None

        # scripts
        self.scripts_value: scripts_manager.ScriptRunner = field(default=None, init=False)
        self.script_args_value: list = field(default=None, init=False)
        self.scripts_setup_complete: bool = field(default=False, init=False)
        self.script_args = script_args
        self.per_script_args = {}

        # ip adapter
        self.ip_adapter_names = []
        self.ip_adapter_scales = [0.0]
        self.ip_adapter_images = []
        self.ip_adapter_starts = [0.0]
        self.ip_adapter_ends = [1.0]
        self.ip_adapter_crops = []

        # hdr
        self.hdr_mode=hdr_mode
        self.hdr_brightness=hdr_brightness
        self.hdr_color=hdr_color
        self.hdr_sharpen=hdr_sharpen
        self.hdr_clamp=hdr_clamp
        self.hdr_boundary=hdr_boundary
        self.hdr_threshold=hdr_threshold
        self.hdr_maximize=hdr_maximize
        self.hdr_max_center=hdr_max_center
        self.hdr_max_boundary=hdr_max_boundary
        self.hdr_color_picker=hdr_color_picker
        self.hdr_tint_ratio=hdr_tint_ratio

        # globals
        self.embedder = None
        self.override = None
        self.scheduled_prompt: bool = False
        self.prompt_embeds = []
        self.positive_pooleds = []
        self.negative_embeds = []
        self.negative_pooleds = []
        self.prompt_attention_masks = []
        self.negative_prompt_attention_mask = []
        self.xyz = xyz
        self.abort = False

        # set model
        if sd_model_checkpoint is not None and len(sd_model_checkpoint) > 0:
            from modules import sd_checkpoint
            if sd_checkpoint.select_checkpoint(op='model', sd_model_checkpoint=sd_model_checkpoint) is None:
                shared.log.error(f'Processing: model="{sd_model_checkpoint}" not found')
                self.abort = True
            else:
                shared.opts.sd_model_checkpoint = sd_model_checkpoint
                sd_models.reload_model_weights()

    def __str__(self):
        return f'{self.__class__.__name__}: {self.__dict__}'

    @property
    def sd_model(self):
        return shared.sd_model

    @property
    def scripts(self):
        return self.scripts_value

    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    def setup_scripts(self):
        self.scripts_setup_complete = True
        self.scripts.setup_scripts()

    def comment(self, text):
        self.comments[text] = 1

    def init(self, all_prompts=None, all_seeds=None, all_subseeds=None):
        pass

    def close(self):
        self.sampler = None
        self.scripts = None


class StableDiffusionProcessingVideo(StableDiffusionProcessing):
    def __init__(self, **kwargs):
        self.prompt_template: str = None
        self.frames: int = 1
        self.scheduler_shift: float = 0.0
        self.vae_tile_frames: int = 0
        debug(f'Process init: mode={self.__class__.__name__} kwargs={kwargs}') # pylint: disable=protected-access
        super().__init__(**kwargs)

class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    def __init__(self, **kwargs):
        debug(f'Process init: mode={self.__class__.__name__} kwargs={kwargs}') # pylint: disable=protected-access
        super().__init__(**kwargs)

    def init(self, all_prompts=None, all_seeds=None, all_subseeds=None):
        shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        self.width = self.width or 1024
        self.height = self.height or 1024
        if all_prompts is not None:
            self.all_prompts = all_prompts
        if all_seeds is not None:
            self.all_seeds = all_seeds
        if all_subseeds is not None:
            self.all_subseeds = all_subseeds

    def init_hr(self, scale = None, upscaler = None, force = False): # pylint: disable=unused-argument
        scale = scale or self.hr_scale
        upscaler = upscaler or self.hr_upscaler
        if self.hr_resize_x == 0 and self.hr_resize_y == 0:
            self.hr_upscale_to_x = int(self.width * scale)
            self.hr_upscale_to_y = int(self.height * scale)
        else:
            if self.hr_resize_y == 0:
                self.hr_upscale_to_x = self.hr_resize_x
                self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
            elif self.hr_resize_x == 0:
                self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                self.hr_upscale_to_y = self.hr_resize_y
            elif self.hr_resize_x > 0 and self.hr_resize_y > 0:
                self.hr_upscale_to_x = self.hr_resize_x
                self.hr_upscale_to_y = self.hr_resize_y
        shared.log.debug(f'Init hires: upscaler="{self.hr_upscaler}" sampler="{self.hr_sampler_name}" resize={self.hr_resize_x}x{self.hr_resize_y} upscale={self.hr_upscale_to_x}x{self.hr_upscale_to_y}')


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    def __init__(self, **kwargs):
        debug(f'Process init: mode={self.__class__.__name__} kwargs={kwargs}') # pylint: disable=protected-access
        super().__init__(**kwargs)

    def init(self, all_prompts=None, all_seeds=None, all_subseeds=None):
        if self.init_images is not None and len(self.init_images) > 0:
            vae_scale_factor = sd_vae.get_vae_scale_factor()
            if self.width is None or self.width == 0:
                self.width = int(vae_scale_factor * (self.init_images[0].width * self.scale_by // vae_scale_factor))
            if self.height is None or self.height == 0:
                self.height = int(vae_scale_factor * (self.init_images[0].height * self.scale_by // vae_scale_factor))
        if getattr(self, 'image_mask', None) is not None:
            shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.INPAINTING)
        elif getattr(self, 'init_images', None) is not None:
            shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)

        if all_prompts is not None:
            self.all_prompts = all_prompts
        if all_seeds is not None:
            self.all_seeds = all_seeds
        if all_subseeds is not None:
            self.all_subseeds = all_subseeds
        if self.image_mask is not None:
            self.ops.append('inpaint')
        elif self.init_images is not None and len(self.init_images) > 0:
            self.ops.append('img2img')
        crop_region = None

        if self.image_mask is not None:
            if type(self.image_mask) == list:
                self.image_mask = self.image_mask[0]
            if 'control' in self.ops:
                self.image_mask = masking.run_mask(input_image=self.init_images, input_mask=self.image_mask, return_type='Grayscale', invert=self.inpainting_mask_invert==1) # blur/padding are handled in masking module
            else:
                self.image_mask = masking.run_mask(input_image=self.init_images, input_mask=self.image_mask, return_type='Grayscale', invert=self.inpainting_mask_invert==1, mask_blur=self.mask_blur, mask_padding=self.inpaint_full_res_padding) # old img2img
            if self.inpaint_full_res: # mask only inpaint
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region
                crop_mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(resize_mode=2, im=crop_mask, width=self.width, height=self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else: # full image inpaint
                self.image_mask = images.resize_image(resize_mode=self.resize_mode, im=self.image_mask, width=self.width, height=self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)
            self.overlay_images = []

        add_color_corrections = shared.opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        processed_images = []
        if self.init_images is None:
            return
        if not isinstance(self.init_images, list):
            self.init_images = [self.init_images]
        for img in self.init_images:
            if img is None:
                continue
            self.init_img_hash = getattr(self, 'init_img_hash', hashlib.sha256(img.tobytes()).hexdigest()[0:8]) # pylint: disable=attribute-defined-outside-init
            self.init_img_width = getattr(self, 'init_img_width', img.width) # pylint: disable=attribute-defined-outside-init
            self.init_img_height = getattr(self, 'init_img_height', img.height) # pylint: disable=attribute-defined-outside-init
            if shared.opts.save_init_img:
                images.save_image(img, path=shared.opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, suffix="-init-image")
            image = images.flatten(img, shared.opts.img2img_background_color)
            if crop_region is None and self.resize_mode > 0:
                image = images.resize_image(self.resize_mode, image, self.width, self.height, upscaler_name=self.resize_name, context=self.resize_context)
                self.width = image.width
                self.height = image.height
            if self.image_mask is not None and shared.opts.mask_apply_overlay:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_to_paste = image.convert("RGBA").convert("RGBa")
                image_to_mask = ImageOps.invert(self.mask_for_overlay.convert('L')) if self.mask_for_overlay is not None else None
                image_to_mask = image_to_mask.resize((image.width, image.height), Image.Resampling.BILINEAR) if image_to_mask is not None else None
                image_masked.paste(image_to_paste, mask=image_to_mask)
                image_masked = image_masked.convert('RGBA')
                self.overlay_images.append(image_masked)
            if crop_region is not None: # crop_region is not None if we are doing inpaint full res
                image = image.crop(crop_region)
                if image.width != self.width or image.height != self.height:
                    image = images.resize_image(3, image, self.width, self.height, self.resize_name)
            # if self.image_mask is not None and self.inpainting_fill != 1:
            #     image = masking.fill(image, latent_mask)
            if add_color_corrections:
                self.color_corrections.append(processing_helpers.setup_color_correction(image))
            processed_images.append(image)
        self.init_images = processed_images
        # self.batch_size = len(self.init_images)
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size
        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size


class StableDiffusionProcessingControl(StableDiffusionProcessingImg2Img):
    def __init__(self, **kwargs):
        debug(f'Process init: mode={self.__class__.__name__} kwargs={kwargs}') # pylint: disable=protected-access
        super().__init__(**kwargs)

    def init_hr(self, scale = None, upscaler = None, force = False):
        scale = scale or self.scale_by
        upscaler = upscaler or self.resize_name
        use_scale = self.hr_resize_x == 0 or self.hr_resize_y == 0
        if upscaler == 'None' or (use_scale and scale == 1.0):
            return
        self.is_hr_pass = True
        self.hr_force = force
        self.hr_upscaler = upscaler
        if use_scale:
            vae_scale_factor = sd_vae.get_vae_scale_factor()
            self.hr_upscale_to_x, self.hr_upscale_to_y = vae_scale_factor * int(self.width * scale / vae_scale_factor), vae_scale_factor * int(self.height * scale / vae_scale_factor)
        else:
            self.hr_upscale_to_x, self.hr_upscale_to_y = self.hr_resize_x, self.hr_resize_y


def switch_class(p: StableDiffusionProcessing, new_class: type, dct: dict = None):
    signature = inspect.signature(type(new_class).__init__, follow_wrapped=True)
    possible = list(signature.parameters)
    kwargs = {}
    for k, v in p.__dict__.copy().items():
        if k in possible:
            kwargs[k] = v
    if dct is not None:
        for k, v in dct.items():
            if k in possible:
                kwargs[k] = v
    if new_class == StableDiffusionProcessingTxt2Img:
        sd_models.clean_diffuser_pipe(shared.sd_model)
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    debug(f"Switching class: {p.__class__.__name__} -> {new_class.__name__} fn={fn}") # pylint: disable=protected-access
    p.__class__ = new_class
    p.__init__(**kwargs)
    for k, v in p.__dict__.items():
        if hasattr(p, k):
            setattr(p, k, v)
    if dct is not None: # post init set additional values
        for k, v in dct.items():
            if hasattr(p, k):
                valtype = type(getattr(p, k, None))
                if valtype in [int, float, str]:
                    setattr(p, k, valtype(v))
                else:
                    setattr(p, k, v)
    return p
