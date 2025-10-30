from typing import Optional, List
from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from modules import errors, shared, processing_helpers
from modules.api import models, helpers
from modules.control import run


errors.install()


class ItemControl(BaseModel):
    process: str = Field(title="Preprocessor", default="", description="")
    model: str = Field(title="Control Model", default="", description="")
    strength: float = Field(title="Control model strength", default=1.0, description="")
    start: float = Field(title="Control model start", default=0.0, description="")
    end: float = Field(title="Control model end", default=1.0, description="")
    override: str = Field(title="Override image", default=None, description="")


ReqControl = models.create_model_from_signature(
    func = run.control_run,
    model_name = "StableDiffusionProcessingControl",
    additional_fields = [
        {"key": "sampler_name", "type": str, "default": "Default"},
        {"key": "script_name", "type": Optional[str], "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[List[models.ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "face", "type": Optional[models.ItemFace], "default": None, "exclude": True},
        {"key": "control", "type": Optional[List[ItemControl]], "default": [], "exclude": True},
        # {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
    ]
)
if not hasattr(ReqControl, "__config__"):
    ReqControl.__config__ = models.DummyConfig


class ResControl(BaseModel):
    images: List[str] = Field(default=None, title="Images", description="")
    processed: List[str] = Field(default=None, title="Processed", description="")
    params: dict = Field(default={}, title="Settings", description="")
    info: str = Field(default="", title="Info", description="")


class APIControl():
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg = []
        self.units = []

    def sanitize_args(self, args: dict):
        args = vars(args)
        args.pop('sampler_name', None)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        args.pop('face', None)
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
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

    def prepare_face_module(self, req):
        if hasattr(req, "face") and req.face and not req.script_name and (not req.alwayson_scripts or "face" not in req.alwayson_scripts.keys()):
            req.script_name = "face"
            req.script_args = [
                req.face.mode,
                req.face.source_images,
                req.face.ip_model,
                req.face.ip_override_sampler,
                req.face.ip_cache_model,
                req.face.ip_strength,
                req.face.ip_structure,
                req.face.id_strength,
                req.face.id_conditioning,
                req.face.id_cache,
                req.face.pm_trigger,
                req.face.pm_strength,
                req.face.pm_start,
                req.face.fs_cache
            ]
            del req.face

    def prepare_ip_adapter(self, request):
        if hasattr(request, "ip_adapter") and request.ip_adapter:
            args = { 'ip_adapter_names': [], 'ip_adapter_scales': [], 'ip_adapter_crops': [], 'ip_adapter_starts': [], 'ip_adapter_ends': [], 'ip_adapter_images': [], 'ip_adapter_masks': [] }
            for ipadapter in request.ip_adapter:
                if not ipadapter.images or len(ipadapter.images) == 0:
                    continue
                args['ip_adapter_names'].append(ipadapter.adapter)
                args['ip_adapter_scales'].append(ipadapter.scale)
                args['ip_adapter_starts'].append(ipadapter.start)
                args['ip_adapter_ends'].append(ipadapter.end)
                args['ip_adapter_crops'].append(ipadapter.crop)
                args['ip_adapter_images'].append([helpers.decode_base64_to_image(x) for x in ipadapter.images])
                if ipadapter.masks:
                    args['ip_adapter_masks'].append([helpers.decode_base64_to_image(x) for x in ipadapter.masks])

            del request.ip_adapter
            return args
        else:
            return {}

    def prepare_control(self, req):
        from modules.control.unit import Unit, unit_types
        req.units = []
        if req.unit_type is None:
            req.unit_type = 'controlnet'
        if req.unit_type not in unit_types:
            shared.log.error(f'Control uknown unit type: type={req.unit_type} available={unit_types}')
            return
        for i in range(len(req.control)):
            u = req.control[i]
            if (len(self.units) > i) and (self.units[i].process_id == u.process) and (self.units[i].model_id == u.model):
                unit = self.units[i]
                unit.enabled = True
                unit.strength = u.strength
                unit.start = u.start
                unit.end = u.end
            else:
                unit = Unit(
                    enabled = True,
                    unit_type = req.unit_type,
                    model_id = u.model,
                    process_id = u.process,
                    strength = u.strength,
                    start = u.start,
                    end = u.end,
                )
            if u.override is not None:
                unit.override = helpers.decode_base64_to_image(u.override)
            req.units.append(unit)
        self.units = req.units
        del req.control

    def post_control(self, req: ReqControl):
        self.prepare_face_module(req)
        requested = req.control
        self.prepare_control(req)

        # prepare args
        args = req.copy(update={  # Override __init__ params
            "sampler_index": processing_helpers.get_sampler_index(req.sampler_name),
            "is_generator": True,
            "inputs": [helpers.decode_base64_to_image(x) for x in req.inputs] if req.inputs else None,
            "inits": [helpers.decode_base64_to_image(x) for x in req.inits] if req.inits else None,
            "mask": helpers.decode_base64_to_image(req.mask) if req.mask else None,
        })
        args = self.sanitize_args(args)
        send_images = args.pop('send_images', True)

        # run
        with self.queue_lock:
            jobid = shared.state.begin('API-CTL', api=True)
            output_images = []
            output_processed = []
            output_info = ''
            # TODO control: support scripts via api
            # init script args, call scripts.script_control.run, call scripts.script_control.after
            run.control_set({ 'do_not_save_grid': not req.save_images, 'do_not_save_samples': not req.save_images, **self.prepare_ip_adapter(req) })
            run.control_set(getattr(req, "extra", {}))
            res = run.control_run(**args)
            for item in res:
                if len(item) > 0 and (isinstance(item[0], list) or item[0] is None): # output_images
                    output_images += item[0] if item[0] is not None else []
                    output_processed += [item[1]] if item[1] is not None else []
                    output_info += item[2] if len(item) > 2 and item[2] is not None else ''
                elif isinstance(item, str):
                    output_info += item
                else:
                    pass
            shared.state.end(jobid)

        # return
        b64images = list(map(helpers.encode_pil_to_base64, output_images)) if send_images else []
        b64processed = list(map(helpers.encode_pil_to_base64, output_processed)) if send_images else []
        self.sanitize_b64(req)
        req.units = requested
        return ResControl(images=b64images, processed=b64processed, params=vars(req), info=output_info)
