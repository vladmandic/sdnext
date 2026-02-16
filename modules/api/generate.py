from threading import Lock
from fastapi.responses import JSONResponse
from modules import errors, shared, scripts_manager, ui
from modules.api import models, script, helpers
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.paths import resolve_output_path


errors.install()


class APIGenerate:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []
        self.default_script_arg_control = []

    def sanitize_args(self, args: dict):
        args = vars(args)
        args.pop('include_init_images', None) # this is meant to be done by "exclude": True in model
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        args.pop('face', None)
        args.pop('face_id', None)
        args.pop('save_images', None)
        args.pop('control_units', None)
        args.pop('init_control', None)
        return args

    def sanitize_b64(self, request):
        def sanitize_str(args: list):
            for idx in range(0, len(args)):
                if isinstance(args[idx], str) and len(args[idx]) >= 1000:
                    args[idx] = f"<str {len(args[idx])}>"

        if hasattr(request, "alwayson_scripts") and request.alwayson_scripts:
            for script_name in request.alwayson_scripts.keys():
                script_obj = request.alwayson_scripts[script_name]
                if script_obj and "args" in script_obj and script_obj["args"]:
                    sanitize_str(script_obj["args"])
        if hasattr(request, "script_args") and request.script_args:
            sanitize_str(request.script_args)

    def prepare_face_module(self, request):
        if getattr(request, "face", None) is not None and (not request.alwayson_scripts or "face" not in request.alwayson_scripts.keys()):
            request.script_name = "face"
            request.script_args = [
                request.face.mode,
                request.face.source_images,
                request.face.ip_model,
                request.face.ip_override_sampler,
                request.face.ip_cache_model,
                request.face.ip_strength,
                request.face.ip_structure,
                request.face.id_strength,
                request.face.id_conditioning,
                request.face.id_cache,
                request.face.pm_trigger,
                request.face.pm_strength,
                request.face.pm_start,
                request.face.fs_cache
            ]
            del request.face

    def prepare_ip_adapter(self, request, p):
        if hasattr(request, "ip_adapter") and request.ip_adapter:
            p.ip_adapter_names = []
            p.ip_adapter_scales = []
            p.ip_adapter_crops = []
            p.ip_adapter_starts = []
            p.ip_adapter_ends = []
            p.ip_adapter_images = []
            for ipadapter in request.ip_adapter:
                if not ipadapter.images or len(ipadapter.images) == 0:
                    continue
                p.ip_adapter_names.append(ipadapter.adapter)
                p.ip_adapter_scales.append(ipadapter.scale)
                p.ip_adapter_crops.append(ipadapter.crop)
                p.ip_adapter_starts.append(ipadapter.start)
                p.ip_adapter_ends.append(ipadapter.end)
                p.ip_adapter_images.append([helpers.decode_base64_to_image(x) for x in ipadapter.images])
                p.ip_adapter_masks = []
                if ipadapter.masks:
                    p.ip_adapter_masks.append([helpers.decode_base64_to_image(x) for x in ipadapter.masks])
            del request.ip_adapter

    def prepare_control_units(self, request):
        """Build control Unit objects from API request control_units field."""
        if not hasattr(request, "control_units") or not request.control_units:
            return None, None
        import gradio as gr
        from modules.control import unit as control_unit

        units = []
        input_images = []
        with gr.Blocks():
            for cu in request.control_units:
                if not cu.enabled or not cu.image:
                    continue
                u = control_unit.Unit(
                    index=len(units),
                    enabled=True,
                    strength=cu.strength,
                    unit_type=cu.unit_type,
                    start=cu.start,
                    end=cu.end,
                )
                u.model_name = cu.model if cu.model != "None" else None
                u.update_choices()
                u.mode = cu.mode if cu.mode != "default" and cu.mode in u.choices else u.choices[0]
                u.process_name = cu.processor if cu.processor != "None" else None
                u.guess = cu.guess
                u.factor = cu.factor
                u.attention = cu.attention
                u.fidelity = cu.fidelity
                u.query_weight = cu.query_weight
                u.adain_weight = cu.adain_weight
                img = helpers.decode_base64_to_image(cu.image)
                u.override = img
                u.process.override = img
                units.append(u)
                input_images.append(img)

        if not units:
            return None, None
        return units, input_images

    def _run_control(self, request, units, input_images, mode="txt2img"):
        from modules.control.run import control_run
        from modules.processing_helpers import get_sampler_index

        sampler_name = helpers.validate_sampler_name(getattr(request, "sampler_name", "Default") or "Default")
        sampler_index = get_sampler_index(sampler_name)
        send_images = getattr(request, "send_images", True)

        inits = []
        mask = None
        if mode == "img2img":
            if hasattr(request, "init_images") and request.init_images:
                inits = [helpers.decode_base64_to_image(x) for x in request.init_images]
            if hasattr(request, "mask") and request.mask:
                mask = helpers.decode_base64_to_image(request.mask)

        with self.queue_lock:
            unit_type = units[0].type if units else "controlnet"
            jobid = shared.state.begin(f'API-CTRL-{mode.upper()}', api=True)
            gen = control_run(
                units=units,
                inputs=input_images,
                inits=inits,
                mask=mask,
                unit_type=unit_type,
                is_generator=False,
                prompt=getattr(request, "prompt", ""),
                negative_prompt=getattr(request, "negative_prompt", ""),
                steps=getattr(request, "steps", 20),
                sampler_index=sampler_index,
                seed=getattr(request, "seed", -1),
                subseed=getattr(request, "subseed", -1),
                subseed_strength=getattr(request, "subseed_strength", 0),
                cfg_scale=getattr(request, "cfg_scale", 6.0),
                clip_skip=getattr(request, "clip_skip", 1.0),
                denoising_strength=getattr(request, "denoising_strength", 0.3),
                batch_count=getattr(request, "n_iter", 1),
                batch_size=getattr(request, "batch_size", 1),
                width_before=getattr(request, "width", 512),
                height_before=getattr(request, "height", 512),
                enable_hr=getattr(request, "enable_hr", False),
                hr_scale=getattr(request, "hr_scale", 1.0),
                hr_second_pass_steps=getattr(request, "hr_second_pass_steps", 20),
                hr_denoising_strength=getattr(request, "hr_denoising_strength", 0.0),
                hr_upscaler=getattr(request, "hr_upscaler", None),
                hr_force=getattr(request, "hr_force", False),
                hr_resize_x=getattr(request, "hr_resize_x", 0),
                hr_resize_y=getattr(request, "hr_resize_y", 0),
                hr_resize_mode=getattr(request, "hr_resize_mode", 0),
                refiner_steps=getattr(request, "refiner_steps", 5),
                refiner_start=getattr(request, "refiner_start", 0.0),
                refiner_prompt=getattr(request, "refiner_prompt", ""),
                refiner_negative=getattr(request, "refiner_negative", ""),
                vae_type=getattr(request, "vae_type", "Full"),
                tiling=getattr(request, "tiling", False),
                hidiffusion=getattr(request, "hidiffusion", False),
                guidance_scale=getattr(request, "cfg_scale", 6.0),
                diffusers_guidance_rescale=getattr(request, "diffusers_guidance_rescale", 0.7),
                pag_scale=getattr(request, "diffusers_pag_scale", 0.0),
                pag_adaptive=getattr(request, "diffusers_pag_adaptive", 0.5),
                cfg_end=getattr(request, "cfg_end", 1.0),
                hdr_mode=getattr(request, "hdr_mode", 0),
                hdr_brightness=getattr(request, "hdr_brightness", 0),
                hdr_color=getattr(request, "hdr_color", 0),
                hdr_sharpen=getattr(request, "hdr_sharpen", 0),
                hdr_clamp=getattr(request, "hdr_clamp", False),
                hdr_boundary=getattr(request, "hdr_boundary", 4.0),
                hdr_threshold=getattr(request, "hdr_threshold", 0.95),
                hdr_maximize=getattr(request, "hdr_maximize", False),
                hdr_max_center=getattr(request, "hdr_max_center", 0.6),
                hdr_max_boundary=getattr(request, "hdr_max_boundary", 1.0),
                detailer_enabled=getattr(request, "detailer_enabled", False),
                detailer_prompt=getattr(request, "detailer_prompt", ""),
                detailer_negative=getattr(request, "detailer_negative", ""),
                detailer_steps=getattr(request, "detailer_steps", 10),
                detailer_strength=getattr(request, "detailer_strength", 0.3),
                detailer_resolution=getattr(request, "detailer_resolution", 1024),
            )
            # control_run is a generator; with is_generator=False, exhaust and extract return value
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value
            shared.state.end(jobid)

        # result = (output_images, blended_image, html_txt, output_filename)
        output_images = result[0] if result else []
        if output_images:
            b64images = list(map(helpers.encode_pil_to_base64, output_images)) if send_images else []
        else:
            b64images = []
        # Encode preprocessed composite image from control pipeline return value
        processed_b64 = None
        blended = result[1] if result and len(result) > 1 else None
        if blended is not None:
            from PIL import Image
            if isinstance(blended, Image.Image):
                processed_b64 = [helpers.encode_pil_to_base64(blended)]
            elif isinstance(blended, list):
                processed_b64 = [helpers.encode_pil_to_base64(img) for img in blended if isinstance(img, Image.Image)]
        self.sanitize_b64(request)
        return models.ResTxt2Img(images=b64images, parameters=vars(request), info='', processed_images=processed_b64)

    def post_text2img(self, txt2imgreq: models.ReqTxt2Img):
        """Generate images from a text prompt. Supports IP-Adapter, ControlNet units, FaceID, and script overrides."""
        self.prepare_face_module(txt2imgreq)
        control_units, control_images = self.prepare_control_units(txt2imgreq)
        if control_units:
            return self._run_control(txt2imgreq, control_units, control_images, mode="txt2img")
        script_runner = scripts_manager.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui(None)
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(txt2imgreq.script_name, script_runner)
        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": helpers.validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        args = self.sanitize_args(populate)
        send_images = args.pop('send_images', True)
        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            self.prepare_ip_adapter(txt2imgreq, p)
            if getattr(txt2imgreq, 'init_control', None):
                p.init_control = [helpers.decode_base64_to_image(x) for x in txt2imgreq.init_control]
            p.scripts = script_runner
            p.outpath_grids = resolve_output_path(shared.opts.outdir_grids, shared.opts.outdir_txt2img_grids)
            p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_txt2img_samples)
            for key, value in getattr(txt2imgreq, "extra", {}).items():
                setattr(p, key, value)
            jobid = shared.state.begin('API-TXT', api=True)
            script_args = script.init_script_args(p, txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)
            p.script_args = tuple(script_args) # Need to pass args as tuple here
            if selectable_scripts is not None:
                processed = scripts_manager.scripts_txt2img.run(p, *script_args) # Need to pass args as list here
            else:
                processed = process_images(p)
            processed = scripts_manager.scripts_txt2img.after(p, processed, *script_args)
            p.close()
            shared.state.end(jobid)
        if processed is None or processed.images is None or len(processed.images) == 0:
            b64images = []
        else:
            b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        self.sanitize_b64(txt2imgreq)
        info = processed.js() if processed else ''
        return models.ResTxt2Img(images=b64images, parameters=vars(txt2imgreq), info=info)

    def post_img2img(self, img2imgreq: models.ReqImg2Img):
        """Generate images from input images with optional inpainting mask. Supports IP-Adapter, ControlNet units, FaceID, and script overrides."""
        self.prepare_face_module(img2imgreq)
        control_units, control_images = self.prepare_control_units(img2imgreq)
        if control_units:
            return self._run_control(img2imgreq, control_units, control_images, mode="img2img")
        init_images = img2imgreq.init_images
        if init_images is None:
            return JSONResponse(status_code=400, content={"error": "Init image is none"})
        mask = img2imgreq.mask
        if mask:
            mask = helpers.decode_base64_to_image(mask)
        script_runner = scripts_manager.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui(None)
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(img2imgreq.script_name, script_runner)
        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": helpers.validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        args = self.sanitize_args(populate)
        send_images = args.pop('send_images', True)
        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            self.prepare_ip_adapter(img2imgreq, p)
            if getattr(img2imgreq, 'init_control', None):
                p.init_control = [helpers.decode_base64_to_image(x) for x in img2imgreq.init_control]
            p.init_images = [helpers.decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = resolve_output_path(shared.opts.outdir_grids, shared.opts.outdir_img2img_grids)
            p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_img2img_samples)
            for key, value in getattr(img2imgreq, "extra", {}).items():
                setattr(p, key, value)
            jobid = shared.state.begin('API-IMG', api=True)
            script_args = script.init_script_args(p, img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)
            p.script_args = tuple(script_args) # Need to pass args as tuple here
            if selectable_scripts is not None:
                processed = scripts_manager.scripts_img2img.run(p, *script_args) # Need to pass args as list here
            else:
                processed = process_images(p)
            processed = scripts_manager.scripts_img2img.after(p, processed, *script_args)
            p.close()
            shared.state.end(jobid)
        if processed is None or processed.images is None or len(processed.images) == 0:
            b64images = []
        else:
            b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None
        self.sanitize_b64(img2imgreq)
        info = processed.js() if processed else ''
        return models.ResImg2Img(images=b64images, parameters=vars(img2imgreq), info=info)
