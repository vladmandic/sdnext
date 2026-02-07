import inspect
from typing import Any, Optional, Dict, List, Type, Callable, Union
from pydantic import BaseModel, Field, create_model # pylint: disable=no-name-in-module
from inflection import underscore
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
import modules.shared as shared

API_NOT_ALLOWED = [
    "self",
    "kwargs",
    "sd_model",
    "outpath_samples",
    "outpath_grids",
]

class ModelDef(BaseModel):
    field: str
    field_alias: str
    field_type: Any
    field_value: Any
    field_exclude: bool = False


class DummyConfig:
    dummy_value = None


if not hasattr(BaseModel, "__config__"):
    BaseModel.__config__ = DummyConfig


class PydanticModelGenerator:
    def __init__(
        self,
        model_name: str = None,
        class_instance = None,
        additional_fields = None,
        exclude_fields: List = [],
    ):
        def field_type_generator(_k, v):
            field_type = v.annotation
            return Optional[field_type]

        def merge_class_params(class_):
            all_classes = list(filter(lambda x: x is not object, inspect.getmro(class_)))
            parameters = {}
            for classes in all_classes:
                parameters = {**parameters, **inspect.signature(classes.__init__).parameters}
            return parameters


        self._model_name = model_name
        self._class_data = merge_class_params(class_instance)

        self._model_def = [
            ModelDef(
                field=underscore(k),
                field_alias=k,
                field_type=field_type_generator(k, v),
                field_value=v.default
            )
            for (k,v) in self._class_data.items() if k not in API_NOT_ALLOWED
        ]

        for fld in additional_fields:
            self._model_def.append(ModelDef(
                field=underscore(fld["key"]),
                field_alias=fld["key"],
                field_type=fld["type"],
                field_value=fld["default"],
                field_exclude=fld["exclude"] if "exclude" in fld else False))
        for fld in exclude_fields:
            self._model_def = [x for x in self._model_def if x.field != fld]

    def generate_model(self):
        model_fields = { d.field: (d.field_type, Field(default=d.field_value, alias=d.field_alias, exclude=d.field_exclude)) for d in self._model_def }
        DynamicModel = create_model(self._model_name, **model_fields)
        try:
            DynamicModel.__config__.allow_population_by_field_name = True
            DynamicModel.__config__.allow_mutation = True
        except Exception:
            pass
        return DynamicModel

### item classes

class ItemSampler(BaseModel):
    name: str = Field(title="Name")
    options: dict

class ItemVae(BaseModel):
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")

class ItemUpscaler(BaseModel):
    name: str = Field(title="Name")
    model_name: Optional[str] = Field(title="Model Name")
    model_path: Optional[str] = Field(title="Path")
    model_url: Optional[str] = Field(title="URL")
    scale: Optional[float] = Field(title="Scale")

class ItemModel(BaseModel):
    title: str = Field(title="Title")
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")
    type: str = Field(title="Model type")
    sha256: Optional[str] = Field(title="SHA256 hash")
    hash: Optional[str] = Field(title="Short hash")
    config: Optional[str] = Field(title="Config file")

class ItemHypernetwork(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")

class ItemDetailer(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")

class ItemGAN(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")
    scale: Optional[int] = Field(title="Scale")

class ItemStyle(BaseModel):
    name: str = Field(title="Name")
    prompt: Optional[str] = Field(title="Prompt")
    negative_prompt: Optional[str] = Field(title="Negative Prompt")
    extra: Optional[str] = Field(title="Extra")
    filename: Optional[str] = Field(title="Filename")
    preview: Optional[str] = Field(title="Preview")

class ItemExtraNetwork(BaseModel):
    name: str = Field(title="Name")
    type: str = Field(title="Type")
    title: Optional[str] = Field(title="Title")
    fullname: Optional[str] = Field(title="Fullname")
    filename: Optional[str] = Field(title="Filename")
    hash: Optional[str] = Field(title="Hash")
    preview: Optional[str] = Field(title="Preview image URL")

class ItemArtist(BaseModel):
    name: str = Field(title="Name")
    score: float = Field(title="Score")
    category: str = Field(title="Category")

class ItemEmbedding(BaseModel):
    step: Optional[int] = Field(title="Step", description="The number of steps that were used to train this embedding, if available")
    sd_checkpoint: Optional[str] = Field(title="SD Checkpoint", description="The hash of the checkpoint this embedding was trained on, if available")
    sd_checkpoint_name: Optional[str] = Field(title="SD Checkpoint Name", description="The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead")
    shape: int = Field(title="Shape", description="The length of each individual vector in the embedding")
    vectors: int = Field(title="Vectors", description="The number of vectors in the embedding")

class ItemIPAdapter(BaseModel):
    adapter: str = Field(title="Adapter", default="Base", description="IP adapter name")
    images: List[str] = Field(title="Image", default=[], description="IP adapter input images")
    masks: Optional[List[str]] = Field(title="Mask", default=[], description="IP adapter mask images")
    scale: float = Field(title="Scale", default=0.5, ge=0, le=1, description="IP adapter scale")
    start: float = Field(title="Start", default=0.0, ge=0, le=1, description="IP adapter start step")
    end: float = Field(title="End", default=1.0, gt=0, le=1, description="IP adapter end step")
    crop: bool = Field(title="Crop", default=False, description="IP adapter crop face from input")

class ItemFace(BaseModel):
    mode: str = Field(title="Mode", default="FaceID", description="The mode to use (available values: FaceID, FaceSwap, PhotoMaker, InstantID).")
    source_images: list[str] = Field(title="Source Images", description="Source face images, must be base64 encoded containing the image's data.")
    ip_model: str = Field(title="IPAdapter Model", default="FaceID Base", description="The IPAdapter model to use.")
    ip_override_sampler: bool = Field(title="IPAdapter Override Sampler", default=True, description="Should the sampler be overriden?")
    ip_cache_model: bool = Field(title="IPAdapter Cache", default=True, description="Should the IPAdapter model be cached?")
    ip_strength: float = Field(title="IPAdapter Strength", default=1, ge=0, le=2, description="IPAdapter strength of the source images, must be between 0.0 and 2.0.")
    ip_structure: float = Field(title="IPAdapter Structure", default=1, ge=0, le=1, description="IPAdapter structure to use, must be between 0.0 and 1.0.")
    id_strength: float = Field(title="InstantID Strength", default=1, ge=0, le=2, description="InstantID Strength of the source images, must be between 0.0 and 2.0.")
    id_conditioning: float = Field(title="InstantID Condition", default=0.5, ge=0, le=2, description="InstantID control amount, must be between 0.0 and 2.0.")
    id_cache: bool = Field(title="InstantID Cache", default=True, description="Should the InstantID model be cached?")
    pm_trigger: str = Field(title="PhotoMaker Trigger", default="person", description="PhotoMaker trigger word to use.")
    pm_strength: float = Field(title="PhotoMaker Strength", default=1, ge=0, le=2, description="PhotoMaker strength to use, must be between 0.0 and 2.0.")
    pm_start: float = Field(title="PhotoMaker Start", default=0.5, ge=0, le=1, description="PhotoMaker start value, must be between 0.0 and 1.0.")
    fs_cache: bool = Field(title="FaceSwap Cache", default=True, description="Should the FaceSwap model be cached?")

class ScriptArg(BaseModel):
    label: str = Field(default=None, title="Label", description="Name of the argument in UI")
    value: Optional[Any] = Field(default=None, title="Value", description="Default value of the argument")
    minimum: Optional[Any] = Field(default=None, title="Minimum", description="Minimum allowed value for the argumentin UI")
    maximum: Optional[Any] = Field(default=None, title="Minimum", description="Maximum allowed value for the argumentin UI")
    step: Optional[Any] = Field(default=None, title="Minimum", description="Step for changing value of the argumentin UI")
    choices: Optional[Any] = Field(default=None, title="Choices", description="Possible values for the argument")

class ItemScript(BaseModel):
    name: str = Field(default=None, title="Name", description="Script name")
    is_alwayson: bool = Field(default=None, title="IsAlwayson", description="Flag specifying whether this script is an alwayson script")
    is_img2img: bool = Field(default=None, title="IsImg2img", description="Flag specifying whether this script is an img2img script")
    args: List[ScriptArg] = Field(title="Arguments", description="List of script's arguments")

class ItemExtension(BaseModel):
    name: str = Field(title="Name", description="Extension name")
    remote: str = Field(title="Remote", description="Extension Repository URL")
    branch: str = Field(default="uknnown", title="Branch", description="Extension Repository Branch")
    commit_hash: str = Field(title="Commit Hash", description="Extension Repository Commit Hash")
    version: str = Field(title="Version", description="Extension Version")
    commit_date: Union[str, int] = Field(title="Commit Date", description="Extension Repository Commit Date")
    enabled: bool = Field(title="Enabled", description="Flag specifying whether this extension is enabled")

class ItemScheduler(BaseModel):
    name: str = Field(title="Name", description="Scheduler name")
    cls: str = Field(title="Class", description="Scheduler class name")
    options: Dict[str, Any] = Field(title="Options", description="Dictionary of scheduler options")

### request/response classes

ReqTxt2Img = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": Union[int, str], "default": 0},
        {"key": "sampler_name", "type": str, "default": "Default"},
        {"key": "hr_sampler_name", "type": str, "default": "Same as primary"},
        {"key": "script_name", "type": Optional[str], "default": ""},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[List[ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "face", "type": Optional[ItemFace], "default": None, "exclude": True},
        {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
    ]
).generate_model()
if not hasattr(ReqTxt2Img, "__config__"):
    ReqTxt2Img.__config__ = DummyConfig
StableDiffusionTxt2ImgProcessingAPI = ReqTxt2Img

class ResTxt2Img(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated images in base64 format.")
    parameters: dict
    info: str

ReqImg2Img = PydanticModelGenerator(
    "StableDiffusionProcessingImg2Img",
    StableDiffusionProcessingImg2Img,
    [
        {"key": "sampler_index", "type": Union[int, str], "default": 0},
        {"key": "sampler_name", "type": str, "default": "UniPC"},
        {"key": "hr_sampler_name", "type": str, "default": "Same as primary"},
        {"key": "init_images", "type": list, "default": None},
        {"key": "denoising_strength", "type": float, "default": 0.5},
        {"key": "mask", "type": Optional[str], "default": None},
        {"key": "include_init_images", "type": bool, "default": False, "exclude": True},
        {"key": "script_name", "type": Optional[str], "default": ""},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[List[ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "face_id", "type": Optional[ItemFace], "default": None, "exclude": True},
        {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
    ]
).generate_model()
if not hasattr(ReqImg2Img, "__config__"):
    ReqImg2Img.__config__ = DummyConfig
StableDiffusionImg2ImgProcessingAPI = ReqImg2Img

class ResImg2Img(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated images in base64 format.")
    parameters: dict
    info: str

class FileData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")

class ReqProcess(BaseModel):
    resize_mode: float = Field(default=0, title="Resize Mode", description="Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.")
    show_extras_results: bool = Field(default=True, title="Show results", description="Should the backend return the generated image?")
    gfpgan_visibility: float = Field(default=0, title="GFPGAN Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of GFPGAN, values should be between 0 and 1.")
    codeformer_visibility: float = Field(default=0, title="CodeFormer Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of CodeFormer, values should be between 0 and 1.")
    codeformer_weight: float = Field(default=0, title="CodeFormer Weight", ge=0, le=1, allow_inf_nan=False, description="Sets the weight of CodeFormer, values should be between 0 and 1.")
    upscaling_resize: float = Field(default=2, title="Upscaling Factor", ge=1, le=8, description="By how much to upscale the image, only used when resize_mode=0.")
    upscaling_resize_w: int = Field(default=512, title="Target Width", ge=1, description="Target width for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_resize_h: int = Field(default=512, title="Target Height", ge=1, description="Target height for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_crop: bool = Field(default=True, title="Crop to fit", description="Should the upscaler crop the image to fit in the chosen size?")
    upscaler_1: str = Field(default="None", title="Main upscaler", description=f"The name of the main upscaler to use, it has to be one of this list: {' , '.join([x.name for x in shared.sd_upscalers])}")
    upscaler_2: str = Field(default="None", title="Refine upscaler", description=f"The name of the secondary upscaler to use, it has to be one of this list: {' , '.join([x.name for x in shared.sd_upscalers])}")
    extras_upscaler_2_visibility: float = Field(default=0, title="Refine upscaler visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of secondary upscaler, values should be between 0 and 1.")

class ResProcess(BaseModel):
    html_info: str = Field(title="HTML info", description="A series of HTML tags containing the process info.")


class ReqPromptEnhance(BaseModel):
    prompt: str = Field(title="Prompt", description="Prompt to enhance")
    type: str = Field(title="Type", default='text', description="Type of enhancement to perform")
    model: Optional[str] = Field(title="Model", default=None, description="Model to use for enhancement")
    system_prompt: Optional[str] = Field(title="System prompt", default=None, description="Model system prompt")
    image: Optional[str] = Field(title="Image", default=None, description="Image to work on, must be a Base64 string containing the image's data.")
    seed: int = Field(title="Seed", default=-1, description="Seed used to generate the prompt")
    nsfw: bool = Field(title="NSFW", default=True, description="Should NSFW content be allowed?")

class ResPromptEnhance(BaseModel):
    prompt: str = Field(title="Prompt", description="Enhanced prompt")
    seed: int = Field(title="Seed", description="Seed used to generate the prompt")

class ReqProcessImage(ReqProcess):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

class ResProcessImage(ResProcess):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

class ReqProcessBatch(ReqProcess):
    imageList: List[FileData] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

class ResProcessBatch(ResProcess):
    images: List[str] = Field(title="Images", description="The generated images in base64 format.")

class ReqImageInfo(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")

class ResImageInfo(BaseModel):
    info: str = Field(title="Image info", description="A string with the parameters used to generate the image")
    items: dict = Field(title="Items", description="A dictionary containing all the other fields the image had")
    parameters: dict = Field(title="Parameters", description="A dictionary with parsed generation info fields")

class ReqGetLog(BaseModel):
    lines: int = Field(default=100, title="Lines", description="How many lines to return")
    clear: bool = Field(default=False, title="Clear", description="Should the log be cleared after returning the lines?")


class ReqPostLog(BaseModel):
    message: Optional[str] = Field(default=None, title="Message", description="The info message to log")
    debug: Optional[str] = Field(default=None, title="Debug message", description="The debug message to log")
    error: Optional[str] = Field(default=None, title="Error message", description="The error message to log")

class ReqHistory(BaseModel):
    id: Union[int, str, None] = Field(default=None, title="Task ID", description="Task ID")

class ReqProgress(BaseModel):
    skip_current_image: bool = Field(default=False, title="Skip current image", description="Skip current image serialization")

class ResProgress(BaseModel):
    id: Union[int, str, None] = Field(title="TaskID", description="Task ID")
    progress: float = Field(title="Progress", description="The progress with a range of 0 to 1")
    eta_relative: float = Field(title="ETA in secs")
    state: dict = Field(title="State", description="The current state snapshot")
    current_image: Optional[str] = Field(default=None, title="Current image", description="The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.")
    textinfo: Optional[str] = Field(default=None, title="Info text", description="Info text used by WebUI.")

class ResHistory(BaseModel):
    id: Union[int, str, None] = Field(title="ID", description="Task ID")
    job: str = Field(title="Job", description="Job name")
    op: str = Field(title="Operation", description="Job state")
    timestamp: Union[float, None] = Field(title="Timestamp", description="Job timestamp")
    duration: Union[float, None] = Field(title="Duration", description="Job duration")
    outputs: List[str] = Field(title="Outputs", description="List of filenames")

class ResStatus(BaseModel):
    status: str = Field(title="Status", description="Current status")
    task: str = Field(title="Task", description="Current job")
    timestamp: Optional[str] = Field(title="Timestamp", description="Timestamp of the current job")
    current: str = Field(title="Task", description="Current job")
    id: Union[int, str, None] = Field(title="ID", description="ID of the current task")
    job: int = Field(title="Job", description="Current job")
    jobs: int = Field(title="Jobs", description="Total jobs")
    total: int = Field(title="Total Jobs", description="Total jobs")
    step: int = Field(title="Step", description="Current step")
    steps: int = Field(title="Steps", description="Total steps")
    queued: int = Field(title="Queued", description="Number of queued tasks")
    uptime: int = Field(title="Uptime", description="Uptime of the server")
    elapsed: Optional[float] = Field(default=None, title="Elapsed time")
    eta: Optional[float] = Field(default=None, title="ETA in secs")
    progress: Optional[float] = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")

class ReqInterrogate(BaseModel):
    """Request model for OpenCLIP/BLIP image interrogation.

    Analyze image using CLIP model via OpenCLIP to generate prompts.
    For anime-style tagging, use /sdapi/v1/tagger with WaifuDiffusion or DeepBooru.
    """
    image: str = Field(default="", title="Image", description="Image to interrogate. Must be a Base64 encoded string containing the image data (PNG/JPEG).")
    model: str = Field(default="ViT-L-14/openai", title="Model", description="OpenCLIP model to use. Get available models from GET /sdapi/v1/interrogate.")
    clip_model: str = Field(default="ViT-L-14/openai", title="CLIP Model", description="CLIP model used for image-text similarity matching. Larger models (ViT-L, ViT-H) are more accurate but slower and use more VRAM.")
    blip_model: str = Field(default="blip-large", title="Caption Model", description="BLIP model used to generate the initial image caption. The caption model describes the image content which CLIP then enriches with style and flavor terms.")
    mode: str = Field(default="best", title="Mode", description="Interrogation mode. Fast: Quick caption with minimal flavor terms. Classic: Standard interrogation with balanced quality and speed. Best: Most thorough analysis, slowest but highest quality. Negative: Generate terms to use as negative prompt.")
    analyze: bool = Field(default=False, title="Analyze", description="If True, returns detailed image analysis breakdown (medium, artist, movement, trending, flavor) in addition to caption.")
    # Advanced settings (optional per-request overrides)
    min_length: Optional[int] = Field(default=None, title="Min Length", description="Minimum number of tokens in the generated caption.")
    max_length: Optional[int] = Field(default=None, title="Max Length", description="Maximum number of tokens in the generated caption.")
    chunk_size: Optional[int] = Field(default=None, title="Chunk Size", description="Batch size for processing description candidates (flavors). Higher values speed up interrogation but increase VRAM usage.")
    min_flavors: Optional[int] = Field(default=None, title="Min Flavors", description="Minimum number of descriptive tags (flavors) to keep in the final prompt.")
    max_flavors: Optional[int] = Field(default=None, title="Max Flavors", description="Maximum number of descriptive tags (flavors) to keep in the final prompt.")
    flavor_count: Optional[int] = Field(default=None, title="Intermediates", description="Size of the intermediate candidate pool when matching image features to descriptive tags. Higher values may improve quality but are slower.")
    num_beams: Optional[int] = Field(default=None, title="Num Beams", description="Number of beams for beam search during caption generation. Higher values search more possibilities but are slower.")

InterrogateRequest = ReqInterrogate # alias for backwards compatibility

class ResInterrogate(BaseModel):
    """Response model for image interrogation results."""
    caption: Optional[str] = Field(default=None, title="Caption", description="Generated caption/prompt describing the image content and style.")
    medium: Optional[str] = Field(default=None, title="Medium", description="Detected artistic medium (e.g., 'oil painting', 'digital art', 'photograph'). Only returned when analyze=True.")
    artist: Optional[str] = Field(default=None, title="Artist", description="Detected similar artist style (e.g., 'by greg rutkowski'). Only returned when analyze=True.")
    movement: Optional[str] = Field(default=None, title="Movement", description="Detected art movement (e.g., 'art nouveau', 'impressionism'). Only returned when analyze=True.")
    trending: Optional[str] = Field(default=None, title="Trending", description="Trending/platform tags (e.g., 'trending on artstation'). Only returned when analyze=True.")
    flavor: Optional[str] = Field(default=None, title="Flavor", description="Additional descriptive elements (e.g., 'cinematic lighting', 'highly detailed'). Only returned when analyze=True.")

class ReqVQA(BaseModel):
    """Request model for Vision-Language Model (VLM) captioning.

    Analyze image using vision language model to generate captions,
    answer questions, or perform specialized tasks like object detection.
    """
    image: str = Field(default="", title="Image", description="Image to caption. Must be a Base64 encoded string containing the image data.")
    model: str = Field(default="Alibaba Qwen 2.5 VL 3B", title="Model", description="Select which model to use for Visual Language tasks. Use GET /sdapi/v1/vqa/models for full list. Models which support thinking mode are indicated in capabilities.")
    question: str = Field(default="describe the image", title="Question/Task", description="Changes which task the model will perform. Regular text prompts can be used when the task is set to 'Use Prompt'. Common tasks: 'Short Caption', 'Normal Caption', 'Long Caption'. Florence-2 supports: '<OD>' (object detection), '<OCR>' (text recognition). Moondream supports: 'Point at [object]', 'Detect all [objects]'.")
    prompt: Optional[str] = Field(default=None, title="Prompt", description="Custom prompt text. Required when question is 'Use Prompt'. For 'Point at...' tasks, specify what to find (e.g., 'the red car'). For 'Detect all...' tasks, specify what to detect (e.g., 'faces').")
    system: str = Field(default="You are image captioning expert, creative, unbiased and uncensored.", title="System Prompt", description="System prompt controls behavior of the LLM. Processed first and persists throughout conversation. Has highest priority weighting and is always appended at the beginning of the sequence. Use for: Response formatting rules, role definition, style.")
    include_annotated: bool = Field(default=False, title="Include Annotated Image", description="If True and the task produces detection results (object detection, point detection, gaze), returns annotated image with bounding boxes/points drawn. Only applicable for detection tasks on models like Florence-2 and Moondream.")
    # LLM generation parameters (optional overrides)
    max_tokens: Optional[int] = Field(default=None, title="Max Tokens", description="Maximum number of tokens the model can generate in its response. The model is not aware of this limit during generation; it simply sets the hard limit for the length and will forcefully cut off the response when reached.")
    temperature: Optional[float] = Field(default=None, title="Temperature", description="Controls randomness in token selection. Lower values (e.g., 0.1) make outputs more focused and deterministic, always choosing high-probability tokens. Higher values (e.g., 0.9) increase creativity and diversity by allowing less probable tokens. Set to 0 for fully deterministic output.")
    top_k: Optional[int] = Field(default=None, title="Top-K", description="Limits token selection to the K most likely candidates at each step. Lower values (e.g., 40) make outputs more focused and predictable, while higher values allow more diverse choices. Set to 0 to disable.")
    top_p: Optional[float] = Field(default=None, title="Top-P", description="Selects tokens from the smallest set whose cumulative probability exceeds P (e.g., 0.9). Dynamically adapts the number of candidates based on model confidence; fewer options when certain, more when uncertain. Set to 1 to disable.")
    num_beams: Optional[int] = Field(default=None, title="Num Beams", description="Maintains multiple candidate paths simultaneously and selects the overall best sequence. More thorough but much slower and less creative than random sampling. Generally not recommended; most modern VLMs perform better with sampling methods. Set to 1 to disable.")
    do_sample: Optional[bool] = Field(default=None, title="Use Samplers", description="Enable to use sampling (randomly selecting tokens based on sampling methods like Top-K or Top-P) or disable to use greedy decoding (selecting the most probable token at each step). Enabling makes outputs more diverse and creative but less deterministic.")
    thinking_mode: Optional[bool] = Field(default=None, title="Thinking Mode", description="Enables thinking/reasoning, allowing the model to take more time to generate responses. Can lead to more thoughtful and detailed answers but increases response time. Only works with models that support this feature.")
    prefill: Optional[str] = Field(default=None, title="Prefill Text", description="Pre-fills the start of the model's response to guide its output format or content by forcing it to continue the prefill text. Prefill is filtered out and does not appear in the final response unless keep_prefill is True. Leave empty to let the model generate from scratch.")
    keep_thinking: Optional[bool] = Field(default=None, title="Keep Thinking Trace", description="Include the model's reasoning process in the final output. Useful for understanding how the model arrived at its answer. Only works with models that support thinking mode.")
    keep_prefill: Optional[bool] = Field(default=None, title="Keep Prefill", description="Include the prefill text at the beginning of the final output. If disabled, the prefill text used to guide the model is removed from the result.")

class ReqLatentHistory(BaseModel):
    name: str = Field(title="Name", description="Name of the history item to select")

class ResVQA(BaseModel):
    """Response model for VLM captioning results."""
    answer: Optional[str] = Field(default=None, title="Answer", description="Generated caption, answer, or analysis from the VLM. Format depends on the question/task type.")
    annotated_image: Optional[str] = Field(default=None, title="Annotated Image", description="Base64 encoded PNG image with detection results drawn (bounding boxes, points). Only returned when include_annotated=True and the task produces detection results.")

class ItemVLMModel(BaseModel):
    """VLM model information."""
    name: str = Field(title="Name", description="Display name of the model")
    repo: str = Field(title="Repository", description="HuggingFace repository ID")
    prompts: List[str] = Field(title="Prompts", description="Available prompts/tasks for this model")
    capabilities: List[str] = Field(title="Capabilities", description="Model capabilities: caption, vqa, detection, ocr, thinking")

class ResVLMPrompts(BaseModel):
    """Available VLM prompts grouped by category."""
    common: Optional[List[str]] = Field(default=None, title="Common", description="Prompts available for all models: Use Prompt, Short/Normal/Long Caption")
    florence: Optional[List[str]] = Field(default=None, title="Florence", description="Florence-2 specific: Phrase Grounding, Object Detection, OCR, etc.")
    moondream: Optional[List[str]] = Field(default=None, title="Moondream", description="Moondream specific: Point at..., Detect all..., Detect Gaze")
    moondream2_only: Optional[List[str]] = Field(default=None, title="Moondream 2 Only", description="Moondream 2 specific prompts (gaze detection)")
    available: Optional[List[str]] = Field(default=None, title="Available", description="When filtered by model, the available prompts for that model")

class ItemTaggerModel(BaseModel):
    """Tagger model information."""
    name: str = Field(title="Name", description="Model name")
    type: str = Field(title="Type", description="Model type: waifudiffusion or deepbooru")

class ReqTagger(BaseModel):
    """Request model for image tagging."""
    image: str = Field(default="", title="Image", description="Image to tag. Must be a Base64 encoded string.")
    model: str = Field(default="wd-eva02-large-tagger-v3", title="Model", description="Model to use for image tagging. WaifuDiffusion models (wd-*): Modern taggers with separate general and character thresholds. DeepBooru: Legacy tagger, uses only general threshold.")
    threshold: float = Field(default=0.5, title="Threshold", description="Confidence threshold for general tags (e.g., objects, actions, settings). Only tags with confidence above this threshold are included in the output. Higher values are more selective (fewer tags), lower values include more tags.", ge=0.0, le=1.0)
    character_threshold: float = Field(default=0.85, title="Character Threshold", description="Confidence threshold for character-specific tags (e.g., character names, specific traits). Only tags with confidence above this threshold are included. Higher values are more selective, lower values include more potential matches. Not supported by DeepBooru models.", ge=0.0, le=1.0)
    max_tags: int = Field(default=74, title="Max Tags", description="Maximum number of tags to include in the output. Limits the result length when an image has many detected features. Tags are sorted by confidence, so the most relevant ones are kept.", ge=1, le=512)
    include_rating: bool = Field(default=False, title="Include Rating", description="Include content rating tags in the output (e.g., safe, questionable, explicit). Useful for filtering or categorizing images by their content rating.")
    sort_alpha: bool = Field(default=False, title="Sort Alphabetically", description="Sort tags alphabetically instead of by confidence score. When disabled, tags are sorted by confidence (highest first). Alphabetical sorting makes it easier to find specific tags.")
    use_spaces: bool = Field(default=False, title="Use Spaces", description="Replace underscores with spaces in tag output. Some prompt systems prefer spaces between words (e.g., 'long hair') while others use underscores (e.g., 'long_hair').")
    escape_brackets: bool = Field(default=True, title="Escape Brackets", description="Escape parentheses and brackets in tags with backslashes. Required when tags contain characters that have special meaning in prompt syntax, such as ( ) [ ]. Enable this when using the output directly in prompts.")
    exclude_tags: str = Field(default="", title="Exclude Tags", description="Comma-separated list of tags to exclude from the output. Useful for filtering out unwanted or redundant tags that appear frequently.")
    show_scores: bool = Field(default=False, title="Show Scores", description="Display confidence scores alongside each tag. Shows how certain the model is about each tag (0.0 to 1.0). Useful for understanding which tags are most reliable.")

class ResTagger(BaseModel):
    """Response model for image tagging results."""
    tags: str = Field(title="Tags", description="Comma-separated list of detected tags")
    scores: Optional[dict] = Field(default=None, title="Scores", description="Tag confidence scores (when show_scores=True)")

class ResTrain(BaseModel):
    info: str = Field(title="Train info", description="Response string from train embedding task.")

class ResCreate(BaseModel):
    info: str = Field(title="Create info", description="Response string from create embedding task.")

class ResPreprocess(BaseModel):
    info: str = Field(title="Preprocess info", description="Response string from preprocessing task.")

fields = {}
for key, metadata in shared.opts.data_labels.items():
    value = shared.opts.data.get(key) or shared.opts.data_labels[key].default
    optType = shared.opts.typemap.get(type(metadata.default), type(value))

    if metadata is not None:
        fields.update({key: (Optional[optType], Field(
            default=metadata.default, description=metadata.label))})
    else:
        fields.update({key: (Optional[optType], Field())})

OptionsModel = create_model("Options", **fields)

flags = {}
_options = vars(shared.parser)['_option_string_actions']
for key in _options:
    if _options[key].dest != 'help':
        flag = _options[key]
        _type = Optional[str]
        if _options[key].default is not None:
            _type = type(_options[key].default)
        flags.update({flag.dest: (_type, Field(default=flag.default, description=flag.help))})

FlagsModel = create_model("Flags", **flags)

class ResEmbeddings(BaseModel):
    loaded: list = Field(default=None, title="loaded", description="List of loaded embeddings")
    skipped: list = Field(default=None, title="skipped", description="List of skipped embeddings")

class ResMemory(BaseModel):
    ram: dict = Field(title="RAM", description="System memory stats")
    cuda: dict = Field(title="CUDA", description="nVidia CUDA memory stats")

class ResScripts(BaseModel):
    txt2img: list = Field(default=None, title="Txt2img", description="Titles of scripts (txt2img)")
    img2img: list = Field(default=None, title="Img2img", description="Titles of scripts (img2img)")
    control: list = Field(default=None, title="Control", description="Titles of scripts (control)")

class ResGPU(BaseModel): # definition of http response
    name: str = Field(title="GPU Name")
    data: dict = Field(title="Name/Value data")
    chart: list[float, float] = Field(title="Exactly two items to place on chart")

# helper function

def create_model_from_signature(func: Callable, model_name: str, base_model: Type[BaseModel] = BaseModel, additional_fields: List = [], exclude_fields: List[str] = []) -> type[BaseModel]:
    from PIL import Image

    class Config:
        extra = 'allow'

    args, _, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
    config = Config if varkw else None # Allow extra params if there is a **kwargs parameter in the function signature
    defaults = defaults or []
    args = args or []
    for arg in exclude_fields:
        if arg in args:
            args.remove(arg)
    non_default_args = len(args) - len(defaults)
    defaults = (...,) * non_default_args + defaults
    keyword_only_params = {param: kwonlydefaults.get(param, Any) for param in kwonlyargs}
    for k, v in annotations.items():
        if v == List[Image.Image]:
            annotations[k] = List[str]
        elif v == Image.Image:
            annotations[k] = str
        elif str(v) == 'typing.List[modules.control.unit.Unit]':
            annotations[k] = List[str]
    model_fields = {param: (annotations.get(param, Any), default) for param, default in zip(args, defaults)}

    for fld in additional_fields:
        model_def = ModelDef(
            field=underscore(fld["key"]),
            field_alias=fld["key"],
            field_type=fld["type"],
            field_value=fld["default"],
            field_exclude=fld["exclude"] if "exclude" in fld else False)
        model_fields[model_def.field] = (model_def.field_type, Field(default=model_def.field_value, alias=model_def.field_alias, exclude=model_def.field_exclude))

    for fld in exclude_fields:
        if fld in model_fields:
            del model_fields[fld]

    model = create_model(
        model_name,
        **model_fields,
        **keyword_only_params,
        __base__=base_model,
        __config__=config,
    )
    try:
        model.__config__.allow_population_by_field_name = True
        model.__config__.allow_mutation = True
    except Exception:
        pass
    return model
