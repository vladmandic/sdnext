from typing import Optional, Annotated, Union
from pydantic import BaseModel, Field


# --- Job request types (discriminated union) ---

class GenerateParams(BaseModel):
    type: str = Field("generate", pattern="^generate$")
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    seed: int = -1
    batch_size: int = 1
    batch_count: int = 1
    sampler_name: str = "Default"
    denoising_strength: float = 0.5
    inputs: Optional[list[str]] = None
    inits: Optional[list[str]] = None
    mask: Optional[str] = None
    control: Optional[list[dict]] = None
    ip_adapter: Optional[list[dict]] = None
    extra: Optional[dict] = None
    save_images: bool = True
    # Generation parameters forwarded as-is to the control pipeline
    clip_skip: int = 1
    cfg_end: float = 1.0
    script_name: Optional[str] = None
    script_args: list = Field(default_factory=list)
    alwayson_scripts: dict = Field(default_factory=dict)
    override_settings: dict = Field(default_factory=dict)


class UpscaleParams(BaseModel):
    type: str = Field("upscale", pattern="^upscale$")
    image: str = Field(description="Base64 encoded image")
    upscaler: str = "None"
    scale: float = 2.0


class CaptionParams(BaseModel):
    type: str = Field("caption", pattern="^caption$")
    image: str = Field(description="Base64 encoded image")
    backend: str = "vlm"
    model: Optional[str] = None


class EnhanceParams(BaseModel):
    type: str = Field("enhance", pattern="^enhance$")
    prompt: str = ""
    model: Optional[str] = None


class DetectParams(BaseModel):
    type: str = Field("detect", pattern="^detect$")
    image: str = ""
    model: Optional[str] = None


class PreprocessParams(BaseModel):
    type: str = Field("preprocess", pattern="^preprocess$")
    image: str = ""
    model: str = ""
    params: Optional[dict] = None


JobRequest = Annotated[
    Union[GenerateParams, UpscaleParams, CaptionParams, EnhanceParams, DetectParams, PreprocessParams],
    Field(discriminator="type"),
]


# --- Job response types ---

class ImageRef(BaseModel):
    index: int
    url: str
    width: int
    height: int
    format: str
    size: int


class JobResult(BaseModel):
    images: list[ImageRef] = Field(default_factory=list)
    processed: list[ImageRef] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)


class JobResponse(BaseModel):
    id: str
    type: str
    status: str
    progress: float = 0
    step: int = 0
    steps: int = 0
    eta: Optional[float] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[JobResult] = None


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    offset: int
    limit: int


class ErrorResponse(BaseModel):
    error: str
    code: int
    detail: Optional[str] = None


# --- Simple response types for Swagger docs ---

class StatusResponse(BaseModel):
    id: str
    status: str


class MessageResponse(BaseModel):
    messages: list[str]


class VideoEngine(BaseModel):
    engine: str
    models: list[str]


class VideoModel(BaseModel):
    name: str
    repo: str
    url: str


class VideoLoadResponse(MessageResponse):
    engine: str
    model: str


class FramePackLoadResponse(MessageResponse):
    variant: str


# --- V2 API models ---

class ItemExtraNetworkV2(BaseModel):
    name: str = Field(title="Name", description="Network short name")
    type: str = Field(title="Type", description="Network type: lora, model, embedding, etc.")
    title: Optional[str] = Field(default=None, title="Title", description="Display title")
    fullname: Optional[str] = Field(default=None, title="Full name", description="Fully qualified name")
    filename: Optional[str] = Field(default=None, title="Filename", description="Path to file")
    hash: Optional[str] = Field(default=None, title="Hash", description="Short hash")
    preview: Optional[str] = Field(default=None, title="Preview", description="Preview thumbnail URL")
    version: Optional[str] = Field(default=None, title="Version", description="Base model version")
    tags: list[str] = Field(default_factory=list, title="Tags", description="Tag list")
    size: Optional[int] = Field(default=None, title="Size", description="File size in bytes")
    mtime: Optional[str] = Field(default=None, title="Modified", description="ISO 8601 modification time")

class ResExtraNetworksV2(BaseModel):
    items: list[ItemExtraNetworkV2] = Field(title="Items", description="List of extra network items")
    total: int = Field(title="Total", description="Total matching items before pagination")
    offset: int = Field(title="Offset", description="Number of items skipped")
    limit: int = Field(title="Limit", description="Maximum items returned per page")

class ItemModelV2(BaseModel):
    title: str = Field(title="Title")
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")
    type: str = Field(title="Type")
    hash: Optional[str] = Field(default=None, title="Hash")
    sha256: Optional[str] = Field(default=None, title="SHA256")
    size: Optional[int] = Field(default=None, title="Size", description="File size in bytes")
    mtime: Optional[str] = Field(default=None, title="Modified", description="ISO 8601 modification time")
    version: Optional[str] = Field(default=None, title="Version")
    subfolder: Optional[str] = Field(default=None, title="Subfolder")

class ResModelsV2(BaseModel):
    items: list[ItemModelV2] = Field(title="Items")
    total: int = Field(title="Total")
    offset: int = Field(title="Offset")
    limit: int = Field(title="Limit")

class ItemSamplerV2(BaseModel):
    name: str = Field(title="Name")
    group: str = Field(title="Group", description="Standard, FlowMatch, or Res4Lyf")
    compatible: Optional[bool] = Field(default=None, title="Compatible", description="null if no model loaded")
    options: dict = Field(default_factory=dict, title="Options")

class ItemHistoryV2(BaseModel):
    id: Optional[Union[int, str]] = Field(default=None, title="ID")
    job: str = Field(title="Job")
    op: str = Field(title="Operation")
    timestamp: Optional[float] = Field(default=None, title="Timestamp")
    duration: Optional[float] = Field(default=None, title="Duration")
    outputs: list[str] = Field(default_factory=list, title="Outputs")

class ResHistoryV2(BaseModel):
    items: list[ItemHistoryV2] = Field(title="Items")
    total: int = Field(title="Total")
    offset: int = Field(title="Offset")
    limit: int = Field(title="Limit")

class ResCheckpointV2(BaseModel):
    loaded: bool = Field(title="Loaded")
    type: Optional[str] = Field(default=None, title="Type")
    class_name: Optional[str] = Field(default=None, title="Class Name")
    checkpoint: Optional[str] = Field(default=None, title="Checkpoint")
    title: Optional[str] = Field(default=None, title="Title")
    name: Optional[str] = Field(default=None, title="Name")
    filename: Optional[str] = Field(default=None, title="Filename")
    hash: Optional[str] = Field(default=None, title="Hash")

class ReqSetCheckpointV2(BaseModel):
    sd_model_checkpoint: str = Field(title="Checkpoint")
    dtype: Optional[str] = Field(default=None, title="Dtype")
    force: bool = Field(default=False, title="Force")

class ResSetCheckpointV2(BaseModel):
    ok: bool = Field(title="OK")
    checkpoint: Optional[ResCheckpointV2] = Field(default=None, title="Checkpoint")

class ItemScriptV2(BaseModel):
    name: str = Field(title="Name")
    is_alwayson: bool = Field(title="Is Always-on")
    contexts: list[str] = Field(default_factory=list, title="Contexts", description="txt2img, img2img, control")
    args: list = Field(default_factory=list, title="Arguments")

class ResScriptsV2(BaseModel):
    scripts: list[ItemScriptV2] = Field(title="Scripts")
