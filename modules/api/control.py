from typing import Optional
from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from modules import errors, shared, processing_helpers
from modules.logger import log
from modules.api import models, helpers
from modules.control import run


errors.install()


class ItemControl(BaseModel):
    process: str = Field(title="Preprocessor", default="", description="Preprocessor name to apply to the input image")
    model: str = Field(title="Control Model", default="", description="Control model name (ControlNet, T2I Adapter, etc.)")
    strength: float = Field(title="Control model strength", default=1.0, description="How strongly the control model influences generation (0.0-2.0)")
    start: float = Field(title="Control model start", default=0.0, description="Step fraction at which control begins (0.0-1.0)")
    end: float = Field(title="Control model end", default=1.0, description="Step fraction at which control ends (0.0-1.0)")
    override: str = Field(title="Override image", default=None, description="Base64 encoded override image, bypasses preprocessor")
    unit_type: Optional[str] = Field(default=None, description="Per-unit type override (controlnet, t2i adapter, xs, lite, reference)")
    mode: str = Field(default="default", description="Control mode for Union/ProMax models")
    guess: bool = Field(default=False, description="Guess mode (ControlNet only)")
    factor: float = Field(default=1.0, description="Adapter factor (T2I only)")
    attention: str = Field(default="Attention", description="Reference attention type")
    fidelity: float = Field(default=0.5, description="Reference fidelity")
    query_weight: float = Field(default=1.0, description="Reference query weight")
    adain_weight: float = Field(default=1.0, description="Reference adain weight")
    process_params: Optional[dict] = Field(default=None, description="Custom preprocessor parameter overrides")
    image: Optional[str] = Field(default=None, description="Alias for override — base64 control image")


class ItemXYZ(BaseModel):
    x_type: str = Field(title="X axis values", default='')
    x_values: str = Field(title="X axis values", default='')
    y_type: str = Field(title="Y axis values", default='')
    y_values: str = Field(title="Y axis values", default='')
    z_type: str = Field(title="Z axis values", default='')
    z_values: str = Field(title="Z axis values", default='')
    draw_legend: bool = Field(title="Draw legend", default=True)
    include_grid: bool = Field(title="Include grid", default=True)
    include_subgrids: bool = Field(title="Include subgrids", default=False)
    include_images: bool = Field(title="Include images", default=False)
    include_time: bool = Field(title="Include time", default=False)
    include_text: bool = Field(title="Include text", default=False)


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
        {"key": "ip_adapter", "type": Optional[list[models.ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "face", "type": Optional[models.ItemFace], "default": None, "exclude": True},
        {"key": "control", "type": Optional[list[ItemControl]], "default": [], "exclude": True},
        {"key": "xyz", "type": Optional[ItemXYZ], "default": None, "exclude": True},
        {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
        {"key": "init_control", "type": Optional[list], "default": None, "exclude": True},
    ]
)
if not hasattr(ReqControl, "__config__"):
    ReqControl.__config__ = models.DummyConfig


class ResControl(BaseModel):
    images: list[str] = Field(default=None, title="Images", description="Generated output images in base64 format")
    processed: list[str] = Field(default=None, title="Processed", description="Preprocessed control images in base64 format")
    params: dict = Field(default={}, title="Settings", description="Request parameters echoed back")
    info: str = Field(default="", title="Info", description="Generation info string")


class APIControl:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg = []
        self.units = []

    def sanitize_args(self, args: dict):
        args = vars(args)
        args.pop('sampler_name', None)
        args.pop('alwayson_scripts', None)
        args.pop('face', None)
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
        args.pop('save_images', None)
        args.pop('init_control', None)
        args.pop('mask_blur', None)
        args.pop('inpaint_full_res', None)
        args.pop('inpaint_full_res_padding', None)
        args.pop('inpainting_mask_invert', None)
        args['override_script_name'] = args.pop('script_name', None)
        args['override_script_args'] = args.pop('script_args', None)
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
        if hasattr(request, 'override_script_args') and request.override_script_args:
            request.pop('override_script_args', None)

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

    def prepare_xyz_grid(self, req):
        if hasattr(req, "xyz") and req.xyz:
            req.script_name = "xyz grid"
            req.script_args = [
                req.xyz.x_type, req.xyz.x_values, '',
                req.xyz.y_type, req.xyz.y_values, '',
                req.xyz.z_type, req.xyz.z_values, '',
                False, # csv_mode
                req.xyz.draw_legend,
                False, # no_fixed_seeds
                req.xyz.include_grid, req.xyz.include_subgrids, req.xyz.include_images,
                req.xyz.include_time, req.xyz.include_text,
            ]
            del req.xyz

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
        default_type = req.unit_type if req.unit_type is not None else 'controlnet'
        # Set top-level unit_type from first unit (control_run filters units by this)
        if req.control:
            first_type = req.control[0].unit_type if req.control[0].unit_type is not None else default_type
            req.unit_type = first_type
        for i in range(len(req.control)):
            u = req.control[i]
            ut = u.unit_type if u.unit_type is not None else default_type
            if ut not in unit_types:
                shared.log.error(f'Control unknown unit type: type={ut} available={unit_types}')
                continue
            if (len(self.units) > i) and (self.units[i].process_id == u.process) and (self.units[i].model_id == u.model) and (self.units[i].type == ut):
                unit = self.units[i]
                unit.enabled = True
                unit.strength = u.strength
                unit.start = u.start
                unit.end = u.end
            else:
                unit = Unit(
                    enabled = True,
                    unit_type = ut,
                    model_id = u.model,
                    process_id = u.process,
                    strength = u.strength,
                    start = u.start,
                    end = u.end,
                )
            # Extended per-unit properties
            unit.guess = u.guess
            unit.factor = u.factor
            unit.attention = u.attention
            unit.fidelity = u.fidelity
            unit.query_weight = u.query_weight
            unit.adain_weight = u.adain_weight
            unit.process_params = u.process_params or {}
            unit.update_choices(u.model)
            # Keep mode as string — run.py expects str and converts to index
            if u.mode != "default" and u.mode in unit.choices:
                unit.mode = u.mode
            else:
                unit.mode = unit.choices[0] if unit.choices else "default"
            # Always clear process.override so init_units() re-sets it from unit.override
            # This ensures the preprocessor runs fresh every generation (not reusing stale cached results)
            if unit.process is not None:
                unit.process.override = None
            # Override image: use 'override' field, fall back to 'image' alias
            override_b64 = u.override if u.override is not None else u.image
            if override_b64 is not None:
                unit.override = helpers.decode_base64_to_image(override_b64)
            req.units.append(unit)
        self.units = req.units
        del req.control

    def post_control(self, req: ReqControl):
        """Run the control pipeline (ControlNet, T2I Adapter, XS, Lite, Reference) with one or more control units."""
        requested = req.control
        self.prepare_face_module(req)
        self.prepare_control(req)
        self.prepare_xyz_grid(req)

        # Merge init_control images into inits
        init_control = getattr(req, "init_control", None)
        decoded_inits = [helpers.decode_base64_to_image(x) for x in req.inits] if req.inits else None
        if init_control:
            extra_inits = [helpers.decode_base64_to_image(x) for x in init_control]
            decoded_inits = (decoded_inits or []) + extra_inits

        # Extract excluded fields before copy (Pydantic exclude=True drops them from copy)
        extra = getattr(req, "extra", {}) or {}

        # prepare args
        args = req.copy(update={
            "sampler_index": processing_helpers.get_sampler_index(req.sampler_name),
            "is_generator": True,
            "inputs": [helpers.decode_base64_to_image(x) for x in req.inputs] if req.inputs else None,
            "inits": decoded_inits,
            "mask": helpers.decode_base64_to_image(req.mask) if req.mask else None,
        })

        args = self.sanitize_args(args)
        args['extra'] = extra
        send_images = args.pop('send_images', True)

        # run
        with self.queue_lock:
            jobid = shared.state.begin('API-CTL', api=True)
            output_images = []
            output_processed = []
            output_info = ''
            extra_p_args = {
                'do_not_save_grid': not req.save_images,
                'do_not_save_samples': not req.save_images,
                **self.prepare_ip_adapter(req),
            }
            # Forward inpainting fields
            for field in ('mask_blur', 'inpaint_full_res', 'inpaint_full_res_padding', 'inpainting_mask_invert'):
                val = getattr(req, field, None)
                if val is not None:
                    extra_p_args[field] = val
            run.control_set(extra_p_args)
            # run
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
