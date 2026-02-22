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
