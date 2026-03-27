import re
import inspect
from typing import Any, Optional, Union
from collections.abc import Callable
from pydantic import BaseModel, Field, create_model
from pydantic import VERSION
PYDANTIC_V2 = VERSION.startswith("2.")

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None

try:
    from pydantic import BaseConfig
except ImportError:
    BaseConfig = object
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


class DummyConfig(BaseConfig):
    dummy_value = None


if not hasattr(BaseModel, "__config__"):
    BaseModel.__config__ = DummyConfig


class PydanticConfig(BaseConfig):
    arbitrary_types_allowed = True
    orm_mode = True
    allow_population_by_field_name = True


def underscore(name: str) -> str: # Convert CamelCase or PascalCase string to underscore_case (snake_case).
    # use instead of inflection.underscore
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    s2 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
    return s2.lower()

class PydanticModelGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        class_instance: type | None = None,
        additional_fields: list[dict[str, Any]] | None = None,
        exclude_fields: list | None = None,
    ):
        if exclude_fields is None:
            exclude_fields = []
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
        model_fields: dict[str, Any] = { d.field: (d.field_type, Field(default=d.field_value, alias=d.field_alias, exclude=d.field_exclude)) for d in self._model_def }
        if PYDANTIC_V2:
            config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True, populate_by_name=True)
        else:
            config = PydanticConfig
        DynamicModel = create_model(self._model_name, __config__=config, **model_fields)
        return DynamicModel

### item classes

class ItemSampler(BaseModel):
    name: str = Field(title="Name")
    options: dict = Field(title="Options")

class ItemVae(BaseModel):
    model_name: str = Field(title="Model Name", description="VAE model display name")
    filename: str = Field(title="Filename", description="Path to the VAE file")

class ItemUpscaler(BaseModel):
    name: str = Field(title="Name", description="Upscaler display name")
    model_name: str | None = Field(title="Model Name", description="Underlying model name")
    model_path: str | None = Field(title="Path", description="Path to the model file")
    model_url: str | None = Field(title="URL", description="Download URL for the model")
    scale: float | None = Field(title="Scale", description="Default upscale factor")

class ItemModel(BaseModel):
    title: str = Field(title="Title", description="Full model title including hash")
    model_name: str = Field(title="Model Name", description="Model display name")
    filename: str = Field(title="Filename", description="Path to the model file")
    type: str = Field(title="Model type", description="Model architecture type (e.g., SD, SDXL, Flux)")
    sha256: str | None = Field(title="SHA256 hash", description="Full SHA256 hash of the model file")
    hash: str | None = Field(title="Short hash", description="Short hash identifier for the model")
    config: str | None = Field(title="Config file", description="Path to the model configuration file")

class ItemDetailer(BaseModel):
    name: str = Field(title="Name", description="Detailer model name")
    path: str | None = Field(title="Path", description="Path to the detailer model file")

class ItemStyle(BaseModel):
    name: str = Field(title="Name", description="Style name")
    prompt: str | None = Field(title="Prompt", description="Prompt template text")
    negative_prompt: str | None = Field(title="Negative Prompt", description="Negative prompt template text")
    extra: str | None = Field(title="Extra", description="Additional style data")
    filename: str | None = Field(title="Filename", description="Path to the styles file")
    preview: str | None = Field(title="Preview", description="URL to the style preview image")

class ItemUNet(BaseModel):
    name: str = Field(title="Name", description="UNet/DiT name")
    filename: str | None = Field(title="Filename", description="Path to the UNet/DiT file")

class ItemExtraNetwork(BaseModel):
    name: str = Field(title="Name", description="Network short name")
    type: str = Field(title="Type", description="Network type (lora, checkpoint, embedding, etc.)")
    title: str | None = Field(title="Title", description="Display title")
    fullname: str | None = Field(title="Fullname", description="Fully qualified network name")
    filename: str | None = Field(title="Filename", description="Path to the network file")
    hash: str | None = Field(title="Hash", description="Short hash identifier")
    preview: str | None = Field(title="Preview image URL", description="URL to the preview thumbnail")
    version: str | None = Field(default=None, title="Model version or class", description="Model version string or architecture class")
    tags: str | None = Field(default=None, title="Tags", description="Pipe-separated tag list")

class ItemExtraNetworkDetail(BaseModel):
    name: str = Field(title="Name", description="Network short name")
    type: str = Field(title="Type", description="Network type (lora, checkpoint, embedding, etc.)")
    title: str | None = Field(default=None, title="Title", description="Display title")
    filename: str | None = Field(default=None, title="Filename", description="Path to the network file")
    hash: str | None = Field(default=None, title="Hash", description="Short hash identifier")
    alias: str | None = Field(default=None, title="Alias", description="Short alias for prompt use")
    size: int | None = Field(default=None, title="File size in bytes", description="File size in bytes on disk")
    mtime: str | None = Field(default=None, title="Last modified ISO timestamp", description="Last modification time in ISO 8601 format")
    version: str | None = Field(default=None, title="Model version or class", description="Model version string or architecture class")
    tags: str | None = Field(default=None, title="Tags", description="Pipe-separated tag list")
    description: str | None = Field(default=None, title="Description", description="Human-readable model description")
    info: dict | None = Field(default=None, title="Sidecar JSON info", description="Metadata from sidecar JSON file")

class ItemExtraNetworkFull(BaseModel):
    name: str = Field(title="Name", description="Network short name")
    type: str = Field(title="Type", description="Network type (lora, checkpoint, embedding, etc.)")
    title: str | None = Field(default=None, title="Title", description="Display title")
    fullname: str | None = Field(default=None, title="Fullname", description="Fully qualified network name")
    filename: str | None = Field(default=None, title="Filename", description="Path to the network file")
    hash: str | None = Field(default=None, title="Hash", description="Short hash identifier")
    preview: str | None = Field(default=None, title="Preview image URL", description="URL to the preview thumbnail")
    alias: str | None = Field(default=None, title="Alias", description="Short alias for prompt use")
    size: int | None = Field(default=None, title="File size in bytes", description="File size in bytes on disk")
    mtime: str | None = Field(default=None, title="Last modified ISO timestamp", description="Last modification time in ISO 8601 format")
    version: str | None = Field(default=None, title="Model version or class", description="Model version string or architecture class")
    tags: str | None = Field(default=None, title="Tags", description="Pipe-separated tag list")
    description: str | None = Field(default=None, title="Description", description="Human-readable model description")
    info: dict | None = Field(default=None, title="Sidecar JSON info", description="Metadata from sidecar JSON file")

class ResExtraNetworkDetails(BaseModel):
    items: list[ItemExtraNetworkFull] = Field(title="Items", description="List of extra network items with full detail")
    total: int = Field(title="Total", description="Total number of matching items before pagination")
    offset: int = Field(title="Offset", description="Number of items skipped")
    limit: int = Field(title="Limit", description="Maximum items returned per page")

class ItemEmbedding(BaseModel):
    step: int | None = Field(title="Step", description="The number of steps that were used to train this embedding, if available")
    sd_checkpoint: str | None = Field(title="SD Checkpoint", description="The hash of the checkpoint this embedding was trained on, if available")
    sd_checkpoint_name: str | None = Field(title="SD Checkpoint Name", description="The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead")
    shape: int = Field(title="Shape", description="The length of each individual vector in the embedding")
    vectors: int = Field(title="Vectors", description="The number of vectors in the embedding")

class ItemIPAdapter(BaseModel):
    adapter: str = Field(title="Adapter", default="Base", description="IP adapter name")
    images: list[str] = Field(title="Image", default=[], description="IP adapter input images")
    masks: list[str] | None = Field(title="Mask", default=[], description="IP adapter mask images")
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

class ItemControlUnit(BaseModel):
    process: str = Field(title="Preprocessor", default="", description="Preprocessor name (e.g. 'Canny', 'OpenPose')")
    model: str = Field(title="Control model", default="", description="Control model filename or path")
    strength: float = Field(title="Strength", default=1.0, description="Control influence strength (0.0-2.0)")
    start: float = Field(title="Start", default=0.0, description="Step fraction at which control begins (0.0-1.0)")
    end: float = Field(title="End", default=1.0, description="Step fraction at which control ends (0.0-1.0)")
    override: str | None = Field(title="Override image", default=None, description="Base64-encoded pre-processed control image")
    unit_type: str | None = Field(title="Unit type", default=None, description="Control unit type: 'controlnet', 't2i adapter', 'xs', 'lite', 'reference', or 'ip'")
    image: str | None = Field(title="Image", default=None, description="Base64-encoded control input image")

class ScriptArg(BaseModel):
    label: str = Field(default=None, title="Label", description="Name of the argument in UI")
    value: Any | None = Field(default=None, title="Value", description="Default value of the argument")
    minimum: Any | None = Field(default=None, title="Minimum", description="Minimum allowed value for the argument in UI")
    maximum: Any | None = Field(default=None, title="Maximum", description="Maximum allowed value for the argument in UI")
    step: Any | None = Field(default=None, title="Step", description="Step for changing value of the argument in UI")
    choices: Any | None = Field(default=None, title="Choices", description="Possible values for the argument")

class ItemScript(BaseModel):
    name: str = Field(default=None, title="Name", description="Script name")
    is_alwayson: bool = Field(default=None, title="IsAlwayson", description="Flag specifying whether this script is an alwayson script")
    is_img2img: bool = Field(default=None, title="IsImg2img", description="Flag specifying whether this script is an img2img script")
    args: list[ScriptArg] = Field(title="Arguments", description="List of script's arguments")

class ItemExtension(BaseModel):
    name: str = Field(title="Name", description="Extension name")
    remote: str = Field(title="Remote", description="Extension Repository URL")
    branch: str = Field(default="unknown", title="Branch", description="Extension Repository Branch")
    commit_hash: str = Field(title="Commit Hash", description="Extension Repository Commit Hash")
    version: str = Field(title="Version", description="Extension Version")
    commit_date: str | int = Field(title="Commit Date", description="Extension Repository Commit Date")
    enabled: bool = Field(title="Enabled", description="Flag specifying whether this extension is enabled")

class ItemScheduler(BaseModel):
    name: str = Field(title="Name", description="Scheduler name")
    cls: str = Field(title="Class", description="Scheduler class name")
    options: dict[str, Any] = Field(title="Options", description="Dictionary of scheduler options")

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
        {"key": "ip_adapter", "type": Optional[list[ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "control_units", "type": Optional[list[ItemControlUnit]], "default": None, "exclude": True},
        {"key": "init_control", "type": Optional[list], "default": None, "exclude": True},
        {"key": "face", "type": Optional[ItemFace], "default": None, "exclude": True},
        {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
    ]
).generate_model()
if not hasattr(ReqTxt2Img, "__config__"):
    ReqTxt2Img.__config__ = DummyConfig
StableDiffusionTxt2ImgProcessingAPI = ReqTxt2Img

class ResTxt2Img(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated images in base64 format.")
    parameters: dict = Field(title="Parameters", description="The request parameters echoed back.")
    info: str = Field(title="Info", description="Generation info string with all parameters used.")

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
        {"key": "ip_adapter", "type": Optional[list[ItemIPAdapter]], "default": None, "exclude": True},
        {"key": "control_units", "type": Optional[list[ItemControlUnit]], "default": None, "exclude": True},
        {"key": "init_control", "type": Optional[list], "default": None, "exclude": True},
        {"key": "face_id", "type": Optional[ItemFace], "default": None, "exclude": True},
        {"key": "extra", "type": Optional[dict], "default": {}, "exclude": True},
    ]
).generate_model()
if not hasattr(ReqImg2Img, "__config__"):
    ReqImg2Img.__config__ = DummyConfig
StableDiffusionImg2ImgProcessingAPI = ReqImg2Img

class ResImg2Img(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated images in base64 format.")
    parameters: dict = Field(title="Parameters", description="The request parameters echoed back.")
    info: str = Field(title="Info", description="Generation info string with all parameters used.")

class FileData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")

class ReqProcess(BaseModel):
    resize_mode: float = Field(default=0, title="Resize Mode", description="Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.")
    show_extras_results: bool = Field(default=True, title="Show results", description="Should the backend return the generated image?")
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
    type: str = Field(title="Type", default='text', description="Type of enhancement: text, image, video")
    model: str | None = Field(title="Model", default=None, description="Model to use for enhancement")
    system_prompt: str | None = Field(title="System prompt", default=None, description="Model system prompt")
    image: str | None = Field(title="Image", default=None, description="Image to work on, must be a Base64 string containing the image's data.")
    seed: int = Field(title="Seed", default=-1, description="Seed used to generate the prompt")
    nsfw: bool = Field(title="NSFW", default=True, description="Should NSFW content be allowed?")
    prefix: Optional[str] = Field(title="Prefix", default=None, description="Text prepended to enhanced prompt")
    suffix: Optional[str] = Field(title="Suffix", default=None, description="Text appended to enhanced prompt")
    do_sample: Optional[bool] = Field(title="Sample", default=None, description="Enable sampling")
    max_tokens: Optional[int] = Field(title="Max tokens", default=None, description="Max generation tokens")
    temperature: Optional[float] = Field(title="Temperature", default=None, description="Controls randomness in token selection (0=deterministic, higher=more creative)")
    repetition_penalty: Optional[float] = Field(title="Repetition penalty", default=None, description="Penalizes repeated tokens to reduce repetition (1.0=no penalty)")
    top_k: Optional[int] = Field(title="Top K", default=None, description="Limits token selection to the K most likely candidates")
    top_p: Optional[float] = Field(title="Top P", default=None, description="Nucleus sampling threshold (0-1)")
    thinking: bool = Field(title="Thinking", default=False, description="Enable thinking/reasoning mode")
    keep_thinking: bool = Field(title="Keep thinking", default=False, description="Keep thinking tokens in output")
    use_vision: bool = Field(title="Use vision", default=True, description="Use vision if model supports it")
    prefill: Optional[str] = Field(title="Prefill", default=None, description="Text to prefill the model response with")
    keep_prefill: bool = Field(title="Keep prefill", default=False, description="Keep prefill text in the output")

class ResPromptEnhance(BaseModel):
    prompt: str = Field(title="Prompt", description="Enhanced prompt")
    seed: int = Field(title="Seed", description="Seed used to generate the prompt")

class ReqProcessImage(ReqProcess):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

class ResProcessImage(ResProcess):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

class ReqProcessBatch(ReqProcess):
    imageList: list[FileData] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

class ResProcessBatch(ResProcess):
    images: list[str] = Field(title="Images", description="The generated images in base64 format.")

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
    message: str | None = Field(default=None, title="Message", description="The info message to log")
    debug: str | None = Field(default=None, title="Debug message", description="The debug message to log")
    error: str | None = Field(default=None, title="Error message", description="The error message to log")

class ReqHistory(BaseModel):
    id: int | str | None = Field(default=None, title="Task ID", description="Task ID")

class ReqProgress(BaseModel):
    skip_current_image: bool = Field(default=False, title="Skip current image", description="Skip current image serialization")

class ResProgress(BaseModel):
    id: int | str | None = Field(title="TaskID", description="Task ID")
    progress: float = Field(title="Progress", description="The progress with a range of 0 to 1")
    eta_relative: float = Field(title="ETA in secs")
    state: dict = Field(title="State", description="The current state snapshot")
    current_image: str | None = Field(default=None, title="Current image", description="The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.")
    textinfo: str | None = Field(default=None, title="Info text", description="Info text used by WebUI.")

class ResHistory(BaseModel):
    id: int | str | None = Field(title="ID", description="Task ID")
    job: str = Field(title="Job", description="Job name")
    op: str = Field(title="Operation", description="Job state")
    timestamp: float | None = Field(title="Timestamp", description="Job timestamp")
    duration: float | None = Field(title="Duration", description="Job duration")
    outputs: list[str] = Field(title="Outputs", description="List of filenames")

class ResStatus(BaseModel):
    status: str = Field(title="Status", description="Current status")
    task: str = Field(title="Task", description="Current job")
    timestamp: str | None = Field(title="Timestamp", description="Timestamp of the current job")
    current: str = Field(title="Task", description="Current job")
    id: int | str | None = Field(title="ID", description="ID of the current task")
    job: int = Field(title="Job", description="Current job")
    jobs: int = Field(title="Jobs", description="Total jobs")
    total: int = Field(title="Total Jobs", description="Total jobs")
    step: int = Field(title="Step", description="Current step")
    steps: int = Field(title="Steps", description="Total steps")
    queued: int = Field(title="Queued", description="Number of queued tasks")
    uptime: int = Field(title="Uptime", description="Uptime of the server")
    elapsed: float | None = Field(default=None, title="Elapsed time")
    eta: float | None = Field(default=None, title="ETA in secs")
    progress: float | None = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")

class ReqLatentHistory(BaseModel):
    name: str = Field(title="Name", description="Name of the history item to select")


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

if PYDANTIC_V2:
    pydantic_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True, populate_by_name=True)
else:
    pydantic_config = PydanticConfig
OptionsModel = create_model("Options", __config__=pydantic_config, **fields)

flags = {}
_options = vars(shared.parser)['_option_string_actions']
for key in _options:
    if _options[key].dest != 'help':
        flag = _options[key]
        _type = Optional[str]
        if _options[key].default is not None:
            _type = type(_options[key].default)
        flags.update({flag.dest: (_type, Field(default=flag.default, description=flag.help))})

if PYDANTIC_V2:
    pydantic_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True, populate_by_name=True)
else:
    pydantic_config = PydanticConfig
FlagsModel = create_model("Flags", __config__=pydantic_config, **flags)

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
    name: str = Field(title="GPU Name", description="GPU device name")
    data: dict = Field(title="Name/Value data", description="Key-value pairs of GPU metrics (utilization, temperature, clocks, memory, etc.)")
    chart: list[float, float] = Field(title="Exactly two items to place on chart", description="Two numeric values for chart display (e.g., GPU utilization %, VRAM usage %)")

class ItemLoadedModel(BaseModel):
    name: str = Field(title="Model Name", description="Model or component name")
    category: str = Field(title="Category", description="Model category (pipeline, component, controlnet, lora, ipadapter, upscaler, detailer, caption, enhance)")
    device: Optional[str] = Field(default=None, title="Device", description="Device where the model is loaded (e.g., cuda:0, cpu)")
    size_bytes: Optional[int] = Field(default=None, title="Size (bytes)", description="Total parameter memory footprint in bytes")
    dtype: Optional[str] = Field(default=None, title="Dtype", description="Effective data type (e.g., float16, nf4)")
    extra: Optional[dict] = Field(default=None, title="Extra metadata", description="Additional metadata (role, class, quantization method, etc.)")

# helper function

def create_model_from_signature(func: Callable, model_name: str, base_model: type[BaseModel] = BaseModel, additional_fields: list | None = None, exclude_fields: list[str] | None = None) -> type[BaseModel]:
    from PIL import Image

    if exclude_fields is None:
        exclude_fields = []
    if additional_fields is None:
        additional_fields = []
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
    kw_defaults = kwonlydefaults or {}
    keyword_only_params = {param: (annotations.get(param, Any), kw_defaults.get(param, ...)) for param in kwonlyargs}
    for k, v in annotations.items():
        if v == list[Image.Image]:
            annotations[k] = list[str]
        elif v == Image.Image:
            annotations[k] = str
        elif str(v) == 'typing.List[modules.control.unit.Unit]':
            annotations[k] = list[str]
    model_fields = {param: (annotations.get(param, Any), default) for param, default in zip(args, defaults, strict=False)}

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

    if PYDANTIC_V2:
        config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True, populate_by_name=True, extra='allow' if varkw else 'ignore')
        create_model_args = {'__base__': base_model, '__config__': config}
    else:
        class CustomConfig(PydanticConfig):
            extra = 'allow' if varkw else 'ignore'
        config = CustomConfig
        if base_model == BaseModel:
            create_model_args = {'__config__': config}
        else:
            create_model_args = {'__base__': base_model}

    model = create_model(
        model_name,
        **create_model_args,
        **model_fields,
        **keyword_only_params,
    )
    return model
