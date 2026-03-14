from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64
from modules import errors, shared, postprocessing
from modules.api import models, helpers


processor = None # cached instance of processor
errors.install()


class ReqPreprocess(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")
    model: str = Field(title="Model", description="The model to use for preprocessing")
    params: dict | None = Field(default={}, title="Settings", description="Preprocessor settings")

class ResPreprocess(BaseModel):
    model: str = Field(default='', title="Model", description="The processor model used")
    image: str = Field(default='', title="Image", description="The processed image in base64 format")

class ReqMask(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")
    type: str = Field(title="Mask type", description="Type of masking image to return")
    mask: str | None = Field(title="Mask", description="If optional maks image is not provided auto-masking will be performed")
    model: str | None = Field(title="Model", description="The model to use for preprocessing")
    params: dict | None = Field(default={}, title="Settings", description="Preprocessor settings")

class ReqFace(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")
    model: str | None = Field(title="Model", description="The model to use for detection")

class ResFace(BaseModel):
    classes: list[int] = Field(title="Class", description="The class of detected item")
    labels: list[str] = Field(title="Label", description="The label of detected item")
    boxes: list[list[int]] = Field(title="Box", description="The bounding box of detected item")
    images: list[str] = Field(title="Image", description="The base64 encoded images of detected faces")
    scores: list[float] = Field(title="Scores", description="The scores of the detected faces")

class ResMask(BaseModel):
    mask: str = Field(default='', title="Image", description="The processed image in base64 format")

class ItemPreprocess(BaseModel):
    name: str = Field(title="Name", description="Preprocessor name")
    group: str = Field(default="Other", title="Group", description="Category group")
    params: dict = Field(title="Params", description="Configurable parameters for this preprocessor")

class ItemMask(BaseModel):
    models: list[str] = Field(title="Models", description="Available segmentation model names")
    colormaps: list[str] = Field(title="Color maps", description="Available color map options for mask visualization")
    params: dict = Field(title="Params", description="Current masking parameters")
    types: list[str] = Field(title="Types", description="Available mask return types")


class APIProcess:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock

    def get_preprocess(self):
        """List available image preprocessors with their configurable parameters."""
        from modules.control import processors
        items = []
        for k, v in processors.config.items():
            items.append(ItemPreprocess(name=k, group=v.get('group', 'Other'), params=v.get('params', {})))
        return items

    def post_preprocess(self, req: ReqPreprocess):
        """Run an image preprocessor (e.g., canny, depth, pose) on the input image and return the processed result."""
        global processor # pylint: disable=global-statement
        from modules.control import processors
        processors_list = list(processors.config)
        if req.model not in processors_list:
            return JSONResponse(status_code=400, content={"error": f"Processor model not found: id={req.model}"})
        image = decode_base64_to_image(req.image)
        if processor is None or processor.processor_id != req.model:
            with self.queue_lock:
                processor = processors.Processor(req.model)
        for k, v in req.params.items():
            if k not in processors.config[processor.processor_id]['params']:
                return JSONResponse(status_code=400, content={"error": f"Processor invalid parameter: id={req.model} {k}={v}"})
        jobid = shared.state.begin('API-PRE', api=True)
        processed = processor(image, local_config=req.params)
        image = encode_pil_to_base64(processed)
        shared.state.end(jobid, api=False)
        return ResPreprocess(model=processor.processor_id, image=image)

    def get_mask(self):
        """List available masking models, color maps, parameters, and mask types."""
        from modules import masking
        return ItemMask(models=list(masking.MODELS), colormaps=masking.COLORMAP, params=vars(masking.opts), types=masking.TYPES)

    def post_mask(self, req: ReqMask):
        """Generate a segmentation mask for the input image. Auto-masks if no mask is provided."""
        from modules import masking
        if req.model:
            if req.model not in masking.MODELS:
                return JSONResponse(status_code=400, content={"error": f"Mask model not found: id={req.model}"})
            else:
                masking.init_model(req.model)
        if req.type not in masking.TYPES:
            return JSONResponse(status_code=400, content={"error": f"Mask type not found: id={req.type}"})
        image = decode_base64_to_image(req.image)
        mask = decode_base64_to_image(req.mask) if req.mask else None
        for k, v in req.params.items():
            if not hasattr(masking.opts, k):
                return JSONResponse(status_code=400, content={"error": f"Mask invalid parameter: {k}={v}"})
            else:
                setattr(masking.opts, k, v)
        jobid = shared.state.begin('API-MASK', api=True)
        with self.queue_lock:
            processed = masking.run_mask(input_image=image, input_mask=mask, return_type=req.type)
        shared.state.end(jobid, api=False)
        if processed is None:
            return JSONResponse(status_code=400, content={"error": "Mask is none"})
        image = encode_pil_to_base64(processed)
        return ResMask(mask=image)

    def post_detect(self, req: ReqFace):
        """Detect faces/objects in an image using YOLO. Returns bounding boxes, labels, scores, and cropped images."""
        from modules.shared import yolo # pylint: disable=no-name-in-module
        image = decode_base64_to_image(req.image)
        jobid = shared.state.begin('API-FACE', api=True)
        images = []
        scores = []
        classes = []
        boxes = []
        labels = []
        with self.queue_lock:
            items = yolo.predict(req.model, image)
            for item in items:
                images.append(encode_pil_to_base64(item.item))
                scores.append(item.score)
                classes.append(item.cls)
                labels.append(item.label)
                boxes.append(item.box)
        shared.state.end(jobid, api=False)
        return ResFace(classes=classes, labels=labels, scores=scores, boxes=boxes, images=images)

    def post_prompt_enhance(self, req: models.ReqPromptEnhance):
        """Enhance a prompt using an LLM. Supports text, image-conditioned, and video prompt enhancement modes."""
        from modules import processing_helpers
        seed = req.seed or -1
        seed = processing_helpers.get_fixed_seed(seed)
        prompt = ''
        if req.type in ('text', 'image'):
            from modules.scripts_manager import scripts_txt2img
            default_model = 'google/gemma-3-4b-it' if req.type == 'image' else 'google/gemma-3-1b-it'
            model = default_model if req.model is None or len(req.model) < 4 else req.model
            instance = [s for s in scripts_txt2img.scripts if 'prompt_enhance.py' in s.filename][0]
            prompt = instance.enhance(
                model=model,
                prompt=req.prompt,
                system=req.system_prompt,
                prefix=req.prefix,
                suffix=req.suffix,
                sample=req.do_sample,
                tokens=req.max_tokens,
                temperature=req.temperature,
                penalty=req.repetition_penalty,
                top_k=req.top_k,
                top_p=req.top_p,
                thinking=req.thinking,
                keep_thinking=req.keep_thinking,
                use_vision=req.use_vision,
                prefill=req.prefill or '',
                keep_prefill=req.keep_prefill,
                image=decode_base64_to_image(req.image) if req.image else None,
                seed=seed,
                nsfw=req.nsfw,
            )
        elif req.type == 'video':
            from modules.ui_video_vlm import enhance_prompt
            model = 'Google Gemma 3 4B' if req.model is None or len(req.model) < 4 else req.model
            prompt = enhance_prompt(
                enable=True,
                image=decode_base64_to_image(req.image),
                prompt=req.prompt,
                model=model,
                system_prompt=req.system_prompt,
                nsfw=req.nsfw,
            )
        else:
            raise HTTPException(status_code=400, detail="prompt enhancement: invalid type")
        res = models.ResPromptEnhance(prompt=prompt, seed=seed)
        return res

    def get_prompt_enhance_models(self):
        """
        List available prompt enhancement models.

        Returns model repository IDs with capability flags indicating vision
        (image-conditioned enhancement) and thinking (reasoning mode) support.
        """
        from scripts.prompt_enhance import Options, is_vision_model, is_thinking_model
        result = []
        for repo in Options.models.keys():
            result.append({
                "name": repo,
                "vision": is_vision_model(repo),
                "thinking": is_thinking_model(repo),
            })
        return result

    def set_upscalers(self, req: dict):
        reqDict = vars(req)
        reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
        reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
        return reqDict

    def extras_single_image_api(self, req: models.ReqProcessImage):
        """Upscale or postprocess a single image using the configured upscaler pipeline."""
        reqDict = self.set_upscalers(req)
        reqDict['image'] = helpers.decode_base64_to_image(reqDict['image'])
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessImage(image=helpers.encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ReqProcessBatch):
        """Upscale or postprocess a batch of images using the configured upscaler pipeline."""
        reqDict = self.set_upscalers(req)
        image_list = reqDict.pop('imageList', [])
        image_folder = [helpers.decode_base64_to_image(x.data) for x in image_list]
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessBatch(images=list(map(helpers.encode_pil_to_base64, result[0])), html_info=result[1])
