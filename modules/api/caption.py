"""Caption API — self-contained module for all image captioning endpoints.

Provides three specialized backends and one unified dispatch endpoint:

**Direct endpoints** (backend-specific request/response models):
- POST /sdapi/v1/openclip  — OpenCLIP/BLIP captioning for SD prompt generation
- POST /sdapi/v1/tagger    — WaifuDiffusion and DeepBooru anime/illustration tagging
- POST /sdapi/v1/vqa       — Vision-Language Models (Qwen, Gemma, Florence, Moondream, etc.)

**Dispatch endpoint** (discriminated union routed by ``backend`` field):
- POST /sdapi/v1/caption   — Routes to any backend via ``backend: "openclip" | "tagger" | "vlm"``

**Discovery endpoints** (GET, no request body):
- GET /sdapi/v1/openclip       — List available OpenCLIP models
- GET /sdapi/v1/vqa/models     — List available VLM models with capabilities
- GET /sdapi/v1/vqa/prompts    — List available VLM prompts/tasks (optionally filtered by model)
- GET /sdapi/v1/tagger/models  — List available tagger models

The direct endpoints use backend-specific Pydantic models (ReqCaption/ResCaption,
ReqTagger/ResTagger, ReqVQA/ResVQA) for precise request validation and typed responses.
The dispatch endpoint uses a discriminated union (ReqCaptionDispatch) and a superset
response model (ResCaptionDispatch) that includes fields from all backends.

Core processing logic is shared between direct and dispatch handlers via internal
``_do_openclip``, ``_do_tagger``, and ``_do_vqa`` functions to avoid duplication.
"""

from typing import Optional, List, Union, Literal, Annotated
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
from modules import shared
from modules.api import helpers


# =============================================================================
# Pydantic Models — Direct Endpoints
# =============================================================================

class ReqCaption(BaseModel):
    """Request model for OpenCLIP/BLIP image captioning.

    Analyze image using CLIP model via OpenCLIP to generate prompts.
    For anime-style tagging, use /sdapi/v1/tagger with WaifuDiffusion or DeepBooru.
    """
    image: str = Field(default="", title="Image", description="Image to caption. Must be a Base64 encoded string containing the image data (PNG/JPEG).")
    model: str = Field(default="ViT-L-14/openai", title="Model", description="OpenCLIP model to use. Get available models from GET /sdapi/v1/openclip.")
    clip_model: str = Field(default="ViT-L-14/openai", title="CLIP Model", description="CLIP model used for image-text similarity matching. Larger models (ViT-L, ViT-H) are more accurate but slower and use more VRAM.")
    blip_model: str = Field(default="blip-large", title="Caption Model", description="BLIP model used to generate the initial image caption. The caption model describes the image content which CLIP then enriches with style and flavor terms.")
    mode: str = Field(default="best", title="Mode", description="Caption mode. 'best': Most thorough analysis, slowest but highest quality. 'fast': Quick caption with minimal flavor terms. 'classic': Standard captioning with balanced quality and speed. 'caption': BLIP caption only, no CLIP flavor matching. 'negative': Generate terms suitable for use as a negative prompt.")
    analyze: bool = Field(default=False, title="Analyze", description="If True, returns detailed image analysis breakdown (medium, artist, movement, trending, flavor) in addition to caption.")
    # Advanced settings (optional per-request overrides)
    max_length: Optional[int] = Field(default=None, title="Max Length", description="Maximum number of tokens in the generated caption.")
    chunk_size: Optional[int] = Field(default=None, title="Chunk Size", description="Batch size for processing description candidates (flavors). Higher values speed up captioning but increase VRAM usage.")
    min_flavors: Optional[int] = Field(default=None, title="Min Flavors", description="Minimum number of descriptive tags (flavors) to keep in the final prompt.")
    max_flavors: Optional[int] = Field(default=None, title="Max Flavors", description="Maximum number of descriptive tags (flavors) to keep in the final prompt.")
    flavor_count: Optional[int] = Field(default=None, title="Intermediates", description="Size of the intermediate candidate pool when matching image features to descriptive tags. Higher values may improve quality but are slower.")
    num_beams: Optional[int] = Field(default=None, title="Num Beams", description="Number of beams for beam search during caption generation. Higher values search more possibilities but are slower.")

class ResCaption(BaseModel):
    """Response model for image captioning results."""
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
    question: str = Field(default="describe the image", title="Question/Task", description="Task for the model to perform. Common tasks: 'Short Caption', 'Normal Caption', 'Long Caption'. Set to 'Use Prompt' to pass custom text via the prompt field. Florence-2 tasks: 'Object Detection', 'OCR (Read Text)', 'Phrase Grounding', 'Dense Region Caption', 'Region Proposal', 'OCR with Regions'. PromptGen tasks: 'Analyze', 'Generate Tags', 'Mixed Caption'. Moondream tasks: 'Point at...', 'Detect all...', 'Detect Gaze' (Moondream 2 only). Use GET /sdapi/v1/vqa/prompts?model=<name> to list tasks available for a specific model.")
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

class ResVQA(BaseModel):
    """Response model for VLM captioning results."""
    answer: Optional[str] = Field(default=None, title="Answer", description="Generated caption, answer, or analysis from the VLM. Format depends on the question/task type.")
    annotated_image: Optional[str] = Field(default=None, title="Annotated Image", description="Base64 encoded PNG image with detection results drawn (bounding boxes, points). Only returned when include_annotated=True and the task produces detection results.")

class ItemVLMModel(BaseModel):
    """VLM model information."""
    name: str = Field(title="Name", description="Display name of the model")
    repo: str = Field(title="Repository", description="HuggingFace repository ID")
    prompts: List[str] = Field(title="Prompts", description="Available prompts/tasks for this model")
    capabilities: List[str] = Field(title="Capabilities", description="Model capabilities. Possible values: 'caption' (image captioning), 'vqa' (visual question answering), 'detection' (object/point detection), 'ocr' (text recognition), 'thinking' (reasoning mode support).")

class ResVLMPrompts(BaseModel):
    """Available VLM prompts grouped by category.

    When called without ``model`` parameter, returns all prompt categories.
    When called with ``model``, returns only the ``available`` field with prompts for that model.
    """
    common: Optional[List[str]] = Field(default=None, title="Common", description="Prompts available for all models: Use Prompt, Short/Normal/Long Caption.")
    florence: Optional[List[str]] = Field(default=None, title="Florence", description="Florence-2 base model tasks: Phrase Grounding, Object Detection, Dense Region Caption, Region Proposal, OCR (Read Text), OCR with Regions.")
    promptgen: Optional[List[str]] = Field(default=None, title="PromptGen", description="MiaoshouAI PromptGen fine-tune tasks: Analyze, Generate Tags, Mixed Caption, Mixed Caption+. Only available on PromptGen models.")
    moondream: Optional[List[str]] = Field(default=None, title="Moondream", description="Moondream 2 and 3 tasks: Point at..., Detect all...")
    moondream2_only: Optional[List[str]] = Field(default=None, title="Moondream 2 Only", description="Moondream 2 exclusive tasks: Detect Gaze. Not available in Moondream 3.")
    available: Optional[List[str]] = Field(default=None, title="Available", description="Populated only when filtering by model. Contains the combined list of prompts available for the specified model.")

class ItemTaggerModel(BaseModel):
    """Tagger model information."""
    name: str = Field(title="Name", description="Model name")
    type: str = Field(title="Type", description="Model type: waifudiffusion or deepbooru")

class ReqTagger(BaseModel):
    """Request model for image tagging."""
    image: str = Field(default="", title="Image", description="Image to tag. Must be a Base64 encoded string.")
    model: str = Field(default="wd-eva02-large-tagger-v3", title="Model", description="Model to use for image tagging. WaifuDiffusion models (wd-*): Modern ONNX-based taggers with separate general and character thresholds. DeepBooru ('deepbooru' or 'deepdanbooru'): Legacy tagger, uses only the general threshold. Use GET /sdapi/v1/tagger/models for the full list.")
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


# =============================================================================
# Pydantic Models — Caption Dispatch (Discriminated Union)
# =============================================================================
# These models support the unified /sdapi/v1/caption dispatch endpoint.
# Users can also access backends directly via /openclip, /tagger, /vqa.

class ReqCaptionOpenCLIP(BaseModel):
    """OpenCLIP/BLIP caption request for the dispatch endpoint.

    Generate Stable Diffusion prompts using CLIP for image-text matching and BLIP for captioning.
    Best for: General image captioning, prompt generation, style analysis.
    """
    backend: Literal["openclip"] = Field(default="openclip", description="Backend selector. Use 'openclip' for CLIP/BLIP captioning.")
    image: str = Field(default="", title="Image", description="Image to caption. Must be a Base64 encoded string containing the image data (PNG/JPEG).")
    model: str = Field(default="ViT-L-14/openai", title="Model", description="OpenCLIP model to use. Get available models from GET /sdapi/v1/openclip.")
    clip_model: str = Field(default="ViT-L-14/openai", title="CLIP Model", description="CLIP model used for image-text similarity matching. Larger models (ViT-L, ViT-H) are more accurate but slower.")
    blip_model: str = Field(default="blip-large", title="Caption Model", description="BLIP model used to generate the initial image caption.")
    mode: str = Field(default="best", title="Mode", description="Caption mode: 'best' (highest quality, slowest), 'fast' (quick, fewer flavors), 'classic' (balanced), 'caption' (BLIP only, no CLIP matching), 'negative' (for negative prompts).")
    analyze: bool = Field(default=False, title="Analyze", description="If True, returns detailed breakdown (medium, artist, movement, trending, flavor).")
    max_length: Optional[int] = Field(default=None, title="Max Length", description="Maximum tokens in generated caption.")
    chunk_size: Optional[int] = Field(default=None, title="Chunk Size", description="Batch size for processing flavors.")
    min_flavors: Optional[int] = Field(default=None, title="Min Flavors", description="Minimum descriptive tags to keep.")
    max_flavors: Optional[int] = Field(default=None, title="Max Flavors", description="Maximum descriptive tags to keep.")
    flavor_count: Optional[int] = Field(default=None, title="Intermediates", description="Size of intermediate candidate pool.")
    num_beams: Optional[int] = Field(default=None, title="Num Beams", description="Beams for beam search during caption generation.")


class ReqCaptionTagger(BaseModel):
    """Tagger request for the dispatch endpoint.

    Generate anime/illustration tags using WaifuDiffusion or DeepBooru models.
    Best for: Anime images, booru-style tagging, character identification.
    """
    backend: Literal["tagger"] = Field(..., description="Backend selector. Use 'tagger' for WaifuDiffusion/DeepBooru tagging.")
    image: str = Field(default="", title="Image", description="Image to tag. Must be a Base64 encoded string.")
    model: str = Field(default="wd-eva02-large-tagger-v3", title="Model", description="Tagger model. WaifuDiffusion (wd-*) or DeepBooru (deepbooru/deepdanbooru).")
    threshold: float = Field(default=0.5, title="Threshold", description="Confidence threshold for general tags.", ge=0.0, le=1.0)
    character_threshold: float = Field(default=0.85, title="Character Threshold", description="Confidence threshold for character tags (WaifuDiffusion only).", ge=0.0, le=1.0)
    max_tags: int = Field(default=74, title="Max Tags", description="Maximum number of tags to return.", ge=1, le=512)
    include_rating: bool = Field(default=False, title="Include Rating", description="Include content rating tags (safe/questionable/explicit).")
    sort_alpha: bool = Field(default=False, title="Sort Alphabetically", description="Sort tags alphabetically instead of by confidence.")
    use_spaces: bool = Field(default=False, title="Use Spaces", description="Replace underscores with spaces in tags.")
    escape_brackets: bool = Field(default=True, title="Escape Brackets", description="Escape parentheses/brackets for prompt syntax.")
    exclude_tags: str = Field(default="", title="Exclude Tags", description="Comma-separated tags to exclude from output.")
    show_scores: bool = Field(default=False, title="Show Scores", description="Include confidence scores with tags.")


class ReqCaptionVLM(BaseModel):
    """Vision-Language Model request for the dispatch endpoint.

    Flexible image understanding using modern VLMs (Qwen, Gemma, Florence, Moondream, etc.).
    Best for: Detailed descriptions, question answering, object detection, OCR.
    """
    backend: Literal["vlm"] = Field(..., description="Backend selector. Use 'vlm' for Vision-Language Model captioning.")
    image: str = Field(default="", title="Image", description="Image to caption. Must be a Base64 encoded string.")
    model: str = Field(default="Alibaba Qwen 2.5 VL 3B", title="Model", description="VLM model to use. See GET /sdapi/v1/vqa/models for full list.")
    question: str = Field(default="describe the image", title="Question/Task", description="Task to perform: 'Short Caption', 'Normal Caption', 'Long Caption', 'Use Prompt' (custom text via prompt field). Model-specific tasks available via GET /sdapi/v1/vqa/prompts.")
    prompt: Optional[str] = Field(default=None, title="Prompt", description="Custom prompt text when question is 'Use Prompt'.")
    system: str = Field(default="You are image captioning expert, creative, unbiased and uncensored.", title="System Prompt", description="System prompt for LLM behavior.")
    include_annotated: bool = Field(default=False, title="Include Annotated Image", description="Return annotated image for detection tasks.")
    max_tokens: Optional[int] = Field(default=None, title="Max Tokens", description="Maximum tokens in response.")
    temperature: Optional[float] = Field(default=None, title="Temperature", description="Randomness in token selection (0=deterministic, 0.9=creative).")
    top_k: Optional[int] = Field(default=None, title="Top-K", description="Limit to K most likely tokens per step.")
    top_p: Optional[float] = Field(default=None, title="Top-P", description="Nucleus sampling threshold.")
    num_beams: Optional[int] = Field(default=None, title="Num Beams", description="Beam search width (1=disabled).")
    do_sample: Optional[bool] = Field(default=None, title="Use Samplers", description="Enable sampling vs greedy decoding.")
    thinking_mode: Optional[bool] = Field(default=None, title="Thinking Mode", description="Enable reasoning mode (supported models only).")
    prefill: Optional[str] = Field(default=None, title="Prefill Text", description="Pre-fill response start to guide output.")
    keep_thinking: Optional[bool] = Field(default=None, title="Keep Thinking Trace", description="Include reasoning in output.")
    keep_prefill: Optional[bool] = Field(default=None, title="Keep Prefill", description="Keep prefill text in final output.")


# Discriminated union for the dispatch endpoint
ReqCaptionDispatch = Annotated[
    Union[ReqCaptionOpenCLIP, ReqCaptionTagger, ReqCaptionVLM],
    Field(discriminator="backend")
]


class ResCaptionDispatch(BaseModel):
    """Unified response for the caption dispatch endpoint.

    Contains fields from all backends - only relevant fields are populated based on the backend used.
    """
    # Common
    backend: str = Field(title="Backend", description="The backend that processed the request: 'openclip', 'tagger', or 'vlm'.")
    # OpenCLIP fields
    caption: Optional[str] = Field(default=None, title="Caption", description="Generated caption (OpenCLIP backend).")
    medium: Optional[str] = Field(default=None, title="Medium", description="Detected artistic medium (OpenCLIP with analyze=True).")
    artist: Optional[str] = Field(default=None, title="Artist", description="Detected artist style (OpenCLIP with analyze=True).")
    movement: Optional[str] = Field(default=None, title="Movement", description="Detected art movement (OpenCLIP with analyze=True).")
    trending: Optional[str] = Field(default=None, title="Trending", description="Trending tags (OpenCLIP with analyze=True).")
    flavor: Optional[str] = Field(default=None, title="Flavor", description="Flavor descriptors (OpenCLIP with analyze=True).")
    # Tagger fields
    tags: Optional[str] = Field(default=None, title="Tags", description="Comma-separated tags (Tagger backend).")
    scores: Optional[dict] = Field(default=None, title="Scores", description="Tag confidence scores (Tagger with show_scores=True).")
    # VLM fields
    answer: Optional[str] = Field(default=None, title="Answer", description="VLM response (VLM backend).")
    annotated_image: Optional[str] = Field(default=None, title="Annotated Image", description="Base64 annotated image (VLM with include_annotated=True).")


# =============================================================================
# Shared Core Logic (eliminates duplication between direct and dispatch endpoints)
# =============================================================================

def _validate_image(image_b64: str):
    """Validate and decode a base64 image string, returning an RGB PIL Image.

    Raises:
        HTTPException(404): If image data is missing or too short to be valid.
    """
    if image_b64 is None or len(image_b64) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(image_b64)
    return image.convert('RGB')


def _build_clip_overrides(req) -> dict:
    """Build clip_interrogator overrides dict from request fields."""
    overrides = {}
    for key in ('max_length', 'chunk_size', 'min_flavors', 'max_flavors', 'flavor_count', 'num_beams'):
        val = getattr(req, key, None)
        if val is not None:
            overrides[key] = val
    return overrides or None


def _get_top_item(result):
    """Extract top-ranked item from a Gradio update dict."""
    if isinstance(result, dict) and 'value' in result:
        value = result['value']
        if isinstance(value, dict) and value:
            return next(iter(value.keys()))
        if isinstance(value, str):
            return value
    return None


def _do_openclip(image, req):
    """Core OpenCLIP captioning logic shared by direct and dispatch endpoints.

    Returns (caption, medium, artist, movement, trending, flavor).
    Analysis fields are None when analyze is False.
    """
    from modules.caption.openclip import caption_image, analyze_image, refresh_clip_models
    if req.model not in refresh_clip_models():
        raise HTTPException(status_code=404, detail="Model not found")
    clip_overrides = _build_clip_overrides(req)
    try:
        caption = caption_image(image, clip_model=req.clip_model, blip_model=req.blip_model, mode=req.mode, overrides=clip_overrides)
    except Exception as e:
        caption = str(e)
    if not req.analyze:
        return caption, None, None, None, None, None
    results = analyze_image(image, clip_model=req.clip_model, blip_model=req.blip_model)
    return caption, _get_top_item(results[0]), _get_top_item(results[1]), _get_top_item(results[2]), _get_top_item(results[3]), _get_top_item(results[4])


def _build_vqa_kwargs(req) -> dict:
    """Build generation kwargs dict from VQA request fields."""
    kwargs = {}
    for key in ('max_tokens', 'temperature', 'top_k', 'top_p', 'num_beams', 'do_sample', 'keep_thinking', 'keep_prefill'):
        val = getattr(req, key, None)
        if val is not None:
            kwargs[key] = val
    return kwargs or None


def _do_vqa(image, req):
    """Core VLM captioning logic shared by direct and dispatch endpoints.

    Returns (answer, annotated_b64).
    """
    from modules.caption import vqa
    answer = vqa.caption(
        question=req.question,
        system_prompt=req.system,
        prompt=req.prompt or '',
        image=image,
        model_name=req.model,
        prefill=req.prefill,
        thinking_mode=req.thinking_mode,
        generation_kwargs=_build_vqa_kwargs(req)
    )
    if isinstance(answer, str) and answer.startswith('Error:'):
        raise HTTPException(status_code=422, detail=answer)
    annotated_b64 = None
    if req.include_annotated:
        annotated_img = vqa.get_last_annotated_image()
        if annotated_img is not None:
            annotated_b64 = helpers.encode_pil_to_base64(annotated_img)
    return answer, annotated_b64


def _parse_tagger_scores(tags: str) -> dict:
    """Parse confidence scores from tagger output string."""
    scores = {}
    for item in tags.split(', '):
        item = item.strip()
        if item.startswith('(') and item.endswith(')') and ':' in item:
            inner = item[1:-1]
            tag, score_str = inner.rsplit(':', 1)
            try:
                scores[tag.strip()] = float(score_str.strip())
            except ValueError:
                pass
        elif ':' in item:
            tag, score_str = item.rsplit(':', 1)
            try:
                scores[tag.strip()] = float(score_str.strip())
            except ValueError:
                pass
    return scores or None


def _do_tagger(image, req):
    """Core tagger logic shared by direct and dispatch endpoints.

    Returns (tags, scores).
    """
    from modules.caption import tagger
    is_deepbooru = req.model.lower() in ('deepbooru', 'deepdanbooru')
    original_opts = {
        'tagger_threshold': shared.opts.tagger_threshold,
        'tagger_max_tags': shared.opts.tagger_max_tags,
        'tagger_include_rating': shared.opts.tagger_include_rating,
        'tagger_sort_alpha': shared.opts.tagger_sort_alpha,
        'tagger_use_spaces': shared.opts.tagger_use_spaces,
        'tagger_escape_brackets': shared.opts.tagger_escape_brackets,
        'tagger_exclude_tags': shared.opts.tagger_exclude_tags,
        'tagger_show_scores': shared.opts.tagger_show_scores,
    }
    if not is_deepbooru:
        original_opts['waifudiffusion_character_threshold'] = shared.opts.waifudiffusion_character_threshold
        original_opts['waifudiffusion_model'] = shared.opts.waifudiffusion_model
    try:
        shared.opts.tagger_threshold = req.threshold
        shared.opts.tagger_max_tags = req.max_tags
        shared.opts.tagger_include_rating = req.include_rating
        shared.opts.tagger_sort_alpha = req.sort_alpha
        shared.opts.tagger_use_spaces = req.use_spaces
        shared.opts.tagger_escape_brackets = req.escape_brackets
        shared.opts.tagger_exclude_tags = req.exclude_tags
        shared.opts.tagger_show_scores = req.show_scores
        if not is_deepbooru:
            shared.opts.waifudiffusion_character_threshold = req.character_threshold
            shared.opts.waifudiffusion_model = req.model
        tags = tagger.tag(image, model_name='DeepBooru' if is_deepbooru else None)
        scores = _parse_tagger_scores(tags) if req.show_scores else None
        return tags, scores
    finally:
        for key, value in original_opts.items():
            setattr(shared.opts, key, value)


# =============================================================================
# API Handler Functions
# =============================================================================

def get_caption():
    """
    List available OpenCLIP models.

    Returns model identifiers in ``architecture/pretrained_dataset`` format for use with
    POST /sdapi/v1/openclip or POST /sdapi/v1/caption (``backend="openclip"``).

    **Example Response:**
    ```json
    ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k", "ViT-bigG-14/laion2b_s39b_b160k"]
    ```
    """
    from modules.caption.openclip import refresh_clip_models
    return refresh_clip_models()


def post_caption(req: ReqCaption):
    """
    Caption an image using OpenCLIP/BLIP.

    Generate Stable Diffusion prompts by combining BLIP image captioning with CLIP
    image-text similarity matching. BLIP describes the image content, then CLIP enriches
    the caption with style, artist, and flavor terms.

    For a unified interface that can also dispatch to Tagger or VLM backends, use
    POST /sdapi/v1/caption with ``backend="openclip"`` instead.

    **Modes:**
    - ``best``: Highest quality — combines multiple techniques (default)
    - ``fast``: Quick results with fewer flavor iterations
    - ``classic``: Traditional CLIP interrogator style
    - ``caption``: BLIP caption only, no CLIP flavor matching
    - ``negative``: Generate terms suitable for negative prompts

    Set ``analyze=True`` for a detailed breakdown into medium, artist, movement,
    trending, and flavor categories.

    **Error Codes:**
    - ``404``: Image not provided or model not found
    """
    image = _validate_image(req.image)
    caption, medium, artist, movement, trending, flavor = _do_openclip(image, req)
    return ResCaption(caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)


def post_vqa(req: ReqVQA):
    """
    Caption an image using Vision-Language Models (VLM).

    Flexible image understanding using modern VLMs. Supports 60+ models including
    Google Gemma, Alibaba Qwen, Microsoft Florence, Moondream, JoyCaption, and more.

    For a unified interface that can also dispatch to OpenCLIP or Tagger backends, use
    POST /sdapi/v1/caption with ``backend="vlm"`` instead.

    **Common Tasks** (all models):
    - ``question="Short Caption"`` / ``"Normal Caption"`` / ``"Long Caption"``
    - ``question="Use Prompt"`` with custom text in the ``prompt`` field

    **Florence-2 Tasks** (Florence and PromptGen models):
    - ``question="Object Detection"`` — Returns bounding boxes with labels
    - ``question="Dense Region Caption"`` — Captions for detected regions
    - ``question="OCR (Read Text)"`` — Extract text from image
    - ``question="Phrase Grounding"`` — Locate phrases in the image
    - ``question="Analyze"`` / ``"Generate Tags"`` / ``"Mixed Caption"`` (PromptGen only)

    **Moondream Tasks:**
    - ``question="Point at..."`` with ``prompt="the red car"`` — Returns point coordinates
    - ``question="Detect all..."`` with ``prompt="faces"`` — Returns bounding boxes for all instances
    - ``question="Detect Gaze"`` — Detect face and gaze direction (Moondream 2 only)

    **Annotated Images:**
    Set ``include_annotated=True`` to receive a Base64 PNG with bounding boxes/points drawn.
    Applicable to detection tasks on Florence-2 and Moondream models.

    **Model Tiers:**
    - Small/Fast: Florence 2 Base, SmolVLM 0.5B, FastVLM 0.5B
    - Balanced: Qwen 2.5 VL 3B, Florence 2 Large, Gemma 3 4B
    - High Quality: Qwen 3 VL 8B, JoyCaption Beta, Moondream 3

    Use GET /sdapi/v1/vqa/models for the complete model list with capabilities.

    **Error Codes:**
    - ``404``: Image not provided
    - ``422``: Model returned an error (e.g., unsupported task for model)
    """
    image = _validate_image(req.image)
    answer, annotated_b64 = _do_vqa(image, req)
    return ResVQA(answer=answer, annotated_image=annotated_b64)


def post_caption_dispatch(req: ReqCaptionDispatch):
    """
    Unified caption endpoint — dispatches to any backend based on the ``backend`` field.

    Provides a single entry point for all captioning. The request body is a discriminated
    union: set the ``backend`` field to select the backend, then include that backend's
    parameters in the same request.

    **Backends:**

    1. **OpenCLIP** (``backend: "openclip"``):
       CLIP/BLIP captioning for Stable Diffusion prompts. Response populates ``caption``
       (and ``medium``, ``artist``, ``movement``, ``trending``, ``flavor`` when ``analyze=True``).

    2. **Tagger** (``backend: "tagger"``):
       WaifuDiffusion or DeepBooru anime/illustration tagging. Response populates ``tags``
       (and ``scores`` when ``show_scores=True``).

    3. **VLM** (``backend: "vlm"``):
       Vision-Language Models for flexible image understanding. Response populates ``answer``
       (and ``annotated_image`` when ``include_annotated=True`` with detection tasks).

    **Direct Endpoints** (backend-specific models, simpler interface):
    - POST /sdapi/v1/openclip — OpenCLIP only
    - POST /sdapi/v1/tagger — Tagger only
    - POST /sdapi/v1/vqa — VLM only

    **Error Codes:**
    - ``400``: Unknown backend value
    - ``404``: Image not provided or model not found
    - ``422``: VLM model returned an error
    """
    if req.backend == "openclip":
        image = _validate_image(req.image)
        caption, medium, artist, movement, trending, flavor = _do_openclip(image, req)
        return ResCaptionDispatch(backend="openclip", caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)
    elif req.backend == "tagger":
        image = _validate_image(req.image)
        tags, scores = _do_tagger(image, req)
        return ResCaptionDispatch(backend="tagger", tags=tags, scores=scores)
    elif req.backend == "vlm":
        image = _validate_image(req.image)
        answer, annotated_b64 = _do_vqa(image, req)
        return ResCaptionDispatch(backend="vlm", answer=answer, annotated_image=annotated_b64)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")


def post_tagger(req: ReqTagger):
    """
    Tag an image using WaifuDiffusion or DeepBooru.

    Generate anime/illustration booru-style tags. This is the only endpoint for DeepBooru
    tagging. For a unified interface, use POST /sdapi/v1/caption with ``backend="tagger"``.

    **WaifuDiffusion Models** (recommended):
    - ``wd-eva02-large-tagger-v3`` (best quality)
    - ``wd-vit-tagger-v3``, ``wd-convnext-tagger-v3``, ``wd-swinv2-tagger-v3``
    - Supports separate ``threshold`` (general) and ``character_threshold`` (characters)

    **DeepBooru** (legacy):
    - Set ``model="deepbooru"`` or ``model="deepdanbooru"``
    - Uses only the ``threshold`` parameter; ``character_threshold`` is ignored

    Use GET /sdapi/v1/tagger/models for the full model list.

    **Error Codes:**
    - ``404``: Image not provided
    """
    image = _validate_image(req.image)
    tags, scores = _do_tagger(image, req)
    return ResTagger(tags=tags, scores=scores)


def get_vqa_models():
    """
    List available VLM models for captioning.

    Returns all Vision-Language Models available for POST /sdapi/v1/vqa.

    **Response includes:**
    - `name`: Display name
    - `repo`: HuggingFace repository ID
    - `prompts`: Available prompts/tasks
    - `capabilities`: Model features (caption, vqa, detection, ocr, thinking)
    """
    from modules.caption import vqa
    models_list = []
    for name, repo in vqa.vlm_models.items():
        prompts = vqa.get_prompts_for_model(name)
        capabilities = ["caption", "vqa"]
        name_lower = name.lower()
        if 'florence' in name_lower or 'promptgen' in name_lower:
            capabilities.extend(["detection", "ocr"])
        if 'moondream' in name_lower:
            capabilities.append("detection")
        if vqa.is_thinking_model(name):
            capabilities.append("thinking")
        models_list.append({
            "name": name,
            "repo": repo,
            "prompts": prompts,
            "capabilities": list(set(capabilities))
        })
    return models_list


def get_vqa_prompts(model: Optional[str] = None):
    """
    List available prompts/tasks for VLM models.

    Without ``model`` parameter, returns all prompt categories. With ``model``, returns
    only the prompts available for that specific model in the ``available`` field.

    **Query Parameters:**
    - ``model`` (optional): Filter prompts for a specific model name (e.g., "Alibaba Qwen 2.5 VL 3B")

    **Prompt Categories:**
    - **common**: Use Prompt, Short/Normal/Long Caption (all models)
    - **florence**: Phrase Grounding, Object Detection, OCR, Dense Region Caption, Region Proposal, OCR with Regions (Florence-2 models)
    - **promptgen**: Analyze, Generate Tags, Mixed Caption, Mixed Caption+ (MiaoshouAI PromptGen fine-tunes only)
    - **moondream**: Point at..., Detect all... (Moondream 2 and 3)
    - **moondream2_only**: Detect Gaze (Moondream 2 only)
    """
    from modules.caption import vqa
    if model:
        prompts = vqa.get_prompts_for_model(model)
        return {"available": prompts}
    return {
        "common": vqa.vlm_prompts_common,
        "florence": vqa.vlm_prompts_florence,
        "promptgen": vqa.vlm_prompts_promptgen,
        "moondream": vqa.vlm_prompts_moondream,
        "moondream2_only": vqa.vlm_prompts_moondream2
    }


def get_tagger_models():
    """
    List available tagger models.

    Returns WaifuDiffusion and DeepBooru models for image tagging.

    **WaifuDiffusion Models:**
    - `wd-eva02-large-tagger-v3` (recommended)
    - `wd-vit-tagger-v3`, `wd-convnext-tagger-v3`, `wd-swinv2-tagger-v3`

    **DeepBooru:**
    - Legacy tagger for anime images
    """
    from modules.caption import waifudiffusion
    models_list = []
    for name in waifudiffusion.get_models():
        models_list.append({"name": name, "type": "waifudiffusion"})
    models_list.append({"name": "deepbooru", "type": "deepbooru"})
    return models_list


# =============================================================================
# Route Registration
# =============================================================================

def register_api():
    from modules.shared import api
    api.add_api_route("/sdapi/v1/openclip", get_caption, methods=["GET"], response_model=List[str], tags=["Caption"])
    api.add_api_route("/sdapi/v1/caption", post_caption_dispatch, methods=["POST"], response_model=ResCaptionDispatch, tags=["Caption"])
    api.add_api_route("/sdapi/v1/openclip", post_caption, methods=["POST"], response_model=ResCaption, tags=["Caption"])
    api.add_api_route("/sdapi/v1/vqa", post_vqa, methods=["POST"], response_model=ResVQA, tags=["Caption"])
    api.add_api_route("/sdapi/v1/vqa/models", get_vqa_models, methods=["GET"], response_model=List[ItemVLMModel], tags=["Caption"])
    api.add_api_route("/sdapi/v1/vqa/prompts", get_vqa_prompts, methods=["GET"], response_model=ResVLMPrompts, tags=["Caption"])
    api.add_api_route("/sdapi/v1/tagger", post_tagger, methods=["POST"], response_model=ResTagger, tags=["Caption"])
    api.add_api_route("/sdapi/v1/tagger/models", get_tagger_models, methods=["GET"], response_model=List[ItemTaggerModel], tags=["Caption"])
