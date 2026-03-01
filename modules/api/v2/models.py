from typing import Any, Optional, Annotated, Union
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
    mask_blur: int = 0
    inpaint_full_res: bool = False
    inpaint_full_res_padding: int = 32
    inpainting_mask_invert: int = 0
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


class VideoModelEnriched(BaseModel):
    name: str
    repo: str = ''
    url: str = ''
    cached: bool = False
    loaded: bool = False
    mode: str = 't2v'


class VideoEngine(BaseModel):
    engine: str
    models: list[str]
    model_details: list[VideoModelEnriched] = Field(default_factory=list)


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

class ItemVaeV2(BaseModel):
    name: str = Field(title="Name", description="VAE model display name")
    filename: str = Field(title="Filename", description="Path to the VAE file")

class ItemUpscalerV2(BaseModel):
    name: str = Field(title="Name", description="Upscaler display name")
    group: str = Field(title="Group", description="Upscaler family name")
    model_name: Optional[str] = Field(default=None, title="Model Name", description="Underlying model name")
    model_path: Optional[str] = Field(default=None, title="Path", description="Path to the model file")
    scale: Optional[float] = Field(default=None, title="Scale", description="Default upscale factor")

class ItemEmbeddingV2(BaseModel):
    name: str = Field(title="Name", description="Embedding trigger word")
    filename: Optional[str] = Field(default=None, title="Filename", description="Path to the embedding file")
    step: Optional[int] = Field(default=None, title="Step", description="Training step count")
    shape: Optional[int] = Field(default=None, title="Shape", description="Embedding vector dimension")
    vectors: Optional[int] = Field(default=None, title="Vectors", description="Number of vectors")
    sd_checkpoint: Optional[str] = Field(default=None, title="SD Checkpoint", description="Training checkpoint hash")
    sd_checkpoint_name: Optional[str] = Field(default=None, title="SD Checkpoint Name", description="Training checkpoint name")

class ResEmbeddingsV2(BaseModel):
    loaded: list[ItemEmbeddingV2] = Field(default_factory=list, title="Loaded", description="Successfully loaded embeddings")
    skipped: list[ItemEmbeddingV2] = Field(default_factory=list, title="Skipped", description="Embeddings skipped due to incompatibility")

class ItemPromptStyleV2(BaseModel):
    name: str = Field(title="Name", description="Style name")
    prompt: Optional[str] = Field(default=None, title="Prompt", description="Prompt template text")
    negative_prompt: Optional[str] = Field(default=None, title="Negative Prompt", description="Negative prompt template text")
    extra: Optional[str] = Field(default=None, title="Extra", description="Additional style data")
    description: Optional[str] = Field(default=None, title="Description", description="Human-readable style description")
    wildcards: Optional[str] = Field(default=None, title="Wildcards", description="Wildcard references used by this style")
    filename: Optional[str] = Field(default=None, title="Filename", description="Path to the styles file")
    preview: Optional[str] = Field(default=None, title="Preview", description="URL to the style preview image")
    mtime: Optional[str] = Field(default=None, title="Modified", description="ISO 8601 modification time")

class ResRefreshNetworksV2(BaseModel):
    ok: bool = Field(title="OK", description="Whether the refresh completed successfully")
    total: int = Field(default=0, title="Total", description="Total extra-network items after refresh")


# --- Options / Settings models (v2) ---

class OptionUpdateItemV2(BaseModel):
    key: str = Field(title="Key", description="Option key name")
    changed: bool = Field(title="Changed", description="Whether the value actually changed")

class ResSetOptionsV2(BaseModel):
    ok: bool = Field(title="OK", description="Whether the update completed successfully")
    updated: list[OptionUpdateItemV2] = Field(default_factory=list, title="Updated", description="Per-key update results")

class OptionComponentArgsV2(BaseModel):
    minimum: Optional[float] = Field(default=None, title="Minimum")
    maximum: Optional[float] = Field(default=None, title="Maximum")
    step: Optional[float] = Field(default=None, title="Step")
    choices: Optional[list[str]] = Field(default=None, title="Choices")
    precision: Optional[int] = Field(default=None, title="Precision")
    multiselect: Optional[bool] = Field(default=None, title="Multiselect")

class ItemOptionInfoV2(BaseModel):
    label: str = Field(title="Label")
    section_id: Optional[str] = Field(default=None, title="Section ID")
    section_title: str = Field(title="Section Title")
    visible: bool = Field(title="Visible")
    hidden: bool = Field(title="Hidden")
    type: str = Field(title="Type", description="boolean, number, string, or array")
    component: str = Field(title="Component", description="UI component type")
    component_args: OptionComponentArgsV2 = Field(default_factory=OptionComponentArgsV2, title="Component Args")
    default: Optional[Any] = Field(default=None, title="Default")
    is_legacy: bool = Field(title="Is Legacy")
    is_secret: bool = Field(title="Is Secret")

class ItemSectionInfoV2(BaseModel):
    id: str = Field(title="ID")
    title: str = Field(title="Title")
    hidden: bool = Field(title="Hidden")

class ResOptionsInfoV2(BaseModel):
    options: dict[str, ItemOptionInfoV2] = Field(default_factory=dict, title="Options")
    sections: list[ItemSectionInfoV2] = Field(default_factory=list, title="Sections")

class ItemSecretStatusV2(BaseModel):
    configured: bool = Field(title="Configured")
    source: str = Field(title="Source", description="env, file, or none")
    masked: str = Field(title="Masked", description="Masked value preview")


# --- Control / Preprocessing models (v2) ---

class ItemPreprocessorV2(BaseModel):
    name: str = Field(title="Name", description="Preprocessor name")
    group: str = Field(default="Other", title="Group", description="Category group")
    params: dict = Field(default_factory=dict, title="Params", description="Configurable parameters with default values")

class ReqPreprocessV2(BaseModel):
    image: str = Field(title="Image", description="Base64 encoded image or upload ref")
    model: str = Field(title="Model", description="Preprocessor model name")
    params: dict = Field(default_factory=dict, title="Params", description="Preprocessor settings overrides")

class ResPreprocessV2(BaseModel):
    ok: bool = Field(title="OK", description="Whether preprocessing completed successfully")
    model: str = Field(default="", title="Model", description="Processor model actually used")
    image: str = Field(default="", title="Image", description="Processed image in base64 format")

class ItemScriptV2(BaseModel):
    name: str = Field(title="Name")
    is_alwayson: bool = Field(title="Is Always-on")
    contexts: list[str] = Field(default_factory=list, title="Contexts", description="txt2img, img2img, control")
    args: list = Field(default_factory=list, title="Arguments")

class ResScriptsV2(BaseModel):
    scripts: list[ItemScriptV2] = Field(title="Scripts")


# --- Server info models (v2) ---

class VersionInfoV2(BaseModel):
    app: str = ""
    updated: str = ""
    commit: str = ""
    branch: str = ""
    url: str = ""

class ServerCapabilities(BaseModel):
    txt2img: bool = True
    img2img: bool = True
    control: bool = True
    video: bool = True
    websocket: bool = True

class ServerModelInfo(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None

class ResServerInfoV2(BaseModel):
    version: VersionInfoV2
    backend: str
    platform: str
    api_version: str = "v2"
    capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    model: ServerModelInfo = Field(default_factory=ServerModelInfo)


# --- Memory models (v2) ---

class MemoryUsage(BaseModel):
    free: Optional[int] = None
    used: Optional[int] = None
    total: Optional[int] = None

class MemoryPeakUsage(BaseModel):
    current: Optional[int] = None
    peak: Optional[int] = None

class MemoryWarnings(BaseModel):
    retries: int = 0
    oom: int = 0

class RamMemoryV2(BaseModel):
    free: Optional[int] = None
    used: Optional[int] = None
    total: Optional[int] = None
    error: Optional[str] = None

class CudaMemoryV2(BaseModel):
    system: Optional[MemoryUsage] = None
    active: Optional[MemoryPeakUsage] = None
    allocated: Optional[MemoryPeakUsage] = None
    reserved: Optional[MemoryPeakUsage] = None
    inactive: Optional[MemoryPeakUsage] = None
    events: Optional[MemoryWarnings] = None
    error: Optional[str] = None

class ResMemoryV2(BaseModel):
    ram: RamMemoryV2
    cuda: CudaMemoryV2


# --- System info & GPU models (v2) ---

class ResSystemInfoV2(BaseModel):
    version: dict[str, str] = Field(default_factory=dict)
    uptime: str = ""
    timestamp: str = ""
    platform: dict[str, str] = Field(default_factory=dict)
    torch: str = ""
    gpu: dict[str, str] = Field(default_factory=dict)
    device: dict[str, str] = Field(default_factory=dict)
    libs: dict[str, str] = Field(default_factory=dict)
    backend: str = ""
    pipeline: str = ""
    cross_attention: str = ""
    flags: list[str] = Field(default_factory=list)


class GpuMetrics(BaseModel):
    load_gpu: Optional[float] = None
    load_vram: Optional[float] = None
    temperature: Optional[float] = None
    fan_speed: Optional[float] = None
    power_current: Optional[float] = None
    power_limit: Optional[float] = None
    vram_used: Optional[int] = None
    vram_total: Optional[int] = None

class ResGpuV2(BaseModel):
    name: str
    metrics: GpuMetrics = Field(default_factory=GpuMetrics)
    details: dict[str, str] = Field(default_factory=dict)
    chart_vram_pct: Optional[float] = None
    chart_gpu_pct: Optional[float] = None


# --- Caption models (v2) ---

class ReqOpenClipV2(BaseModel):
    image: str = Field(default="", title="Image", description="Base64 encoded image (PNG/JPEG)")
    model: str = Field(default="ViT-L-14/openai", title="Model", description="OpenCLIP model identifier")
    clip_model: str = Field(default="ViT-L-14/openai", title="CLIP Model", description="CLIP model for image-text similarity")
    blip_model: str = Field(default="blip-large", title="Caption Model", description="BLIP model for initial caption")
    mode: str = Field(default="best", title="Mode", description="Caption mode: best, fast, classic, caption, negative")
    analyze: bool = Field(default=False, title="Analyze", description="Return detailed breakdown (medium, artist, movement, trending, flavor)")
    max_length: Optional[int] = Field(default=None, title="Max Length")
    chunk_size: Optional[int] = Field(default=None, title="Chunk Size")
    min_flavors: Optional[int] = Field(default=None, title="Min Flavors")
    max_flavors: Optional[int] = Field(default=None, title="Max Flavors")
    flavor_count: Optional[int] = Field(default=None, title="Intermediates")
    num_beams: Optional[int] = Field(default=None, title="Num Beams")

class ResOpenClipV2(BaseModel):
    ok: bool = Field(title="OK")
    caption: Optional[str] = Field(default=None, title="Caption")
    medium: Optional[str] = Field(default=None, title="Medium")
    artist: Optional[str] = Field(default=None, title="Artist")
    movement: Optional[str] = Field(default=None, title="Movement")
    trending: Optional[str] = Field(default=None, title="Trending")
    flavor: Optional[str] = Field(default=None, title="Flavor")

class ReqVqaV2(BaseModel):
    image: str = Field(default="", title="Image", description="Base64 encoded image")
    model: str = Field(default="Alibaba Qwen 2.5 VL 3B", title="Model", description="VLM model name")
    question: str = Field(default="describe the image", title="Question/Task")
    prompt: Optional[str] = Field(default=None, title="Prompt", description="Custom prompt text when question is 'Use Prompt'")
    system: str = Field(default="You are image captioning expert, creative, unbiased and uncensored.", title="System Prompt")
    include_annotated: bool = Field(default=False, title="Include Annotated Image")
    max_tokens: Optional[int] = Field(default=None, title="Max Tokens")
    temperature: Optional[float] = Field(default=None, title="Temperature")
    top_k: Optional[int] = Field(default=None, title="Top-K")
    top_p: Optional[float] = Field(default=None, title="Top-P")
    num_beams: Optional[int] = Field(default=None, title="Num Beams")
    do_sample: Optional[bool] = Field(default=None, title="Use Samplers")
    thinking_mode: Optional[bool] = Field(default=None, title="Thinking Mode")
    prefill: Optional[str] = Field(default=None, title="Prefill Text")
    keep_thinking: Optional[bool] = Field(default=None, title="Keep Thinking Trace")
    keep_prefill: Optional[bool] = Field(default=None, title="Keep Prefill")

class ResVqaV2(BaseModel):
    ok: bool = Field(title="OK")
    answer: Optional[str] = Field(default=None, title="Answer")
    annotated_image: Optional[str] = Field(default=None, title="Annotated Image")

class ReqTaggerV2(BaseModel):
    image: str = Field(default="", title="Image", description="Base64 encoded image")
    model: str = Field(default="wd-eva02-large-tagger-v3", title="Model", description="Tagger model name")
    threshold: float = Field(default=0.5, title="Threshold", description="Confidence threshold for general tags", ge=0.0, le=1.0)
    character_threshold: float = Field(default=0.85, title="Character Threshold", description="Confidence threshold for character tags (WD only)", ge=0.0, le=1.0)
    max_tags: int = Field(default=74, title="Max Tags", ge=1, le=512)
    include_rating: bool = Field(default=False, title="Include Rating")
    sort_alpha: bool = Field(default=False, title="Sort Alphabetically")
    use_spaces: bool = Field(default=False, title="Use Spaces")
    escape_brackets: bool = Field(default=True, title="Escape Brackets")
    exclude_tags: str = Field(default="", title="Exclude Tags")
    show_scores: bool = Field(default=False, title="Show Scores")

class ResTaggerV2(BaseModel):
    ok: bool = Field(title="OK")
    tags: str = Field(title="Tags")
    scores: Optional[dict] = Field(default=None, title="Scores")

class ItemVlmModelV2(BaseModel):
    name: str = Field(title="Name")
    group: str = Field(default="Other", title="Group", description="Architecture family")
    repo: str = Field(title="Repository")
    prompts: list[str] = Field(title="Prompts")
    capabilities: list[str] = Field(title="Capabilities")
    cached: bool = Field(default=False, title="Cached", description="Model is available in local HF cache")

class ItemTaggerModelV2(BaseModel):
    name: str = Field(title="Name")
    type: str = Field(title="Type")


# --- Log models (v2) ---

class ResLogV2(BaseModel):
    lines: list[str] = Field(title="Lines", description="Log lines from the in-memory buffer")
    total: int = Field(title="Total", description="Number of lines returned")

class ResLogClearV2(BaseModel):
    ok: bool = Field(title="OK")


# --- PNG Info models (v2) ---

class ReqPngInfoV2(BaseModel):
    image: str = Field(title="Image", description="Base64 encoded PNG image")

class ResPngInfoV2(BaseModel):
    ok: bool = Field(title="OK")
    info: str = Field(default="", title="Info", description="Raw generation parameters string")
    items: dict[str, Any] = Field(default_factory=dict, title="Items", description="All metadata fields from the image")
    parameters: dict[str, Any] = Field(default_factory=dict, title="Parameters", description="Parsed generation parameters")


# --- Extension models (v2) ---

class ItemExtensionV2(BaseModel):
    name: str = Field(title="Name")
    remote: Optional[str] = Field(default=None, title="Remote")
    branch: str = Field(default="unknown", title="Branch")
    commit_hash: Optional[str] = Field(default=None, title="Commit Hash")
    version: Optional[str] = Field(default=None, title="Version")
    commit_date: Optional[str] = Field(default=None, title="Commit Date")
    enabled: bool = Field(default=True, title="Enabled")


# --- Detailer models (v2) ---

class ItemDetailerV2(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(default=None, title="Path")


# --- Prompt Enhance models (v2) ---

class ItemPromptEnhanceModelV2(BaseModel):
    name: str = Field(title="Name")
    group: str = Field(default="Other", title="Group", description="Architecture family")
    vision: bool = Field(title="Vision", description="Supports image input")
    thinking: bool = Field(title="Thinking", description="Supports reasoning mode")
    cached: bool = Field(default=False, title="Cached", description="Model is available in local HF cache")

class ReqPromptEnhanceV2(BaseModel):
    prompt: str = Field(title="Prompt", description="Prompt to enhance")
    type: str = Field(title="Type", default='text', description="Type of enhancement: text, image, video")
    model: Optional[str] = Field(title="Model", default=None)
    system_prompt: Optional[str] = Field(title="System prompt", default=None)
    image: Optional[str] = Field(title="Image", default=None, description="Base64 encoded image")
    seed: int = Field(title="Seed", default=-1)
    nsfw: bool = Field(title="NSFW", default=True)
    prefix: Optional[str] = Field(title="Prefix", default=None)
    suffix: Optional[str] = Field(title="Suffix", default=None)
    do_sample: Optional[bool] = Field(title="Sample", default=None)
    max_tokens: Optional[int] = Field(title="Max tokens", default=None)
    temperature: Optional[float] = Field(title="Temperature", default=None)
    repetition_penalty: Optional[float] = Field(title="Repetition penalty", default=None)
    top_k: Optional[int] = Field(title="Top K", default=None)
    top_p: Optional[float] = Field(title="Top P", default=None)
    thinking: bool = Field(title="Thinking", default=False)
    keep_thinking: bool = Field(title="Keep thinking", default=False)
    use_vision: bool = Field(title="Use vision", default=True)
    prefill: Optional[str] = Field(title="Prefill", default=None)
    keep_prefill: bool = Field(title="Keep prefill", default=False)

class ResPromptEnhanceV2(BaseModel):
    ok: bool = Field(title="OK")
    prompt: str = Field(title="Prompt", description="Enhanced prompt")
    seed: int = Field(title="Seed", description="Seed used")
