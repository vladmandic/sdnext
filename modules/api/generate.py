from threading import Lock
from fastapi.responses import JSONResponse
from modules import errors, shared, scripts_manager, ui
from modules.api import models, script, helpers
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images


errors.install()


class APIGenerate():
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

    def post_text2img(self, txt2imgreq: models.ReqTxt2Img):
        self.prepare_face_module(txt2imgreq)
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
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
            p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
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
        self.prepare_face_module(img2imgreq)
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
            p.init_images = [helpers.decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_img2img_grids
            p.outpath_samples = shared.opts.outdir_img2img_samples
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
