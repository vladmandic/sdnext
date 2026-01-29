# Moondream 3 Preview VLM Implementation
# Source: https://huggingface.co/moondream/moondream3-preview
# Model: 9.3GB, gated (requires HuggingFace authentication)
# Architecture: Mixture-of-Experts (9B total params, 2B active)
import os
import re
import hashlib
import collections
import transformers
from PIL import Image
from modules import shared, devices, sd_models
from modules.caption import vqa_detection


# Debug logging - function-based to avoid circular import
debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None

def debug(*args, **kwargs):
    if debug_enabled:
        shared.log.trace(*args, **kwargs)


# Global state
moondream3_model = None
loaded = None
image_cache: collections.OrderedDict = collections.OrderedDict()  # Bounded LRU cache for encoded image tensors
IMAGE_CACHE_MAX = 8


def get_settings():
    """
    Build settings dict for Moondream 3 API from global VQA options.
    Moondream 3 accepts: temperature, top_p, max_tokens
    """
    settings = {}
    if shared.opts.caption_vlm_max_length > 0:
        settings['max_tokens'] = shared.opts.caption_vlm_max_length
    if shared.opts.caption_vlm_temperature > 0:
        settings['temperature'] = shared.opts.caption_vlm_temperature
    if shared.opts.caption_vlm_top_p > 0:
        settings['top_p'] = shared.opts.caption_vlm_top_p
    return settings if settings else None


def load_model(repo: str):
    """Load Moondream 3 model."""
    global moondream3_model, loaded  # pylint: disable=global-statement

    if moondream3_model is None or loaded != repo:
        shared.log.debug(f'Caption load: vlm="{repo}"')
        moondream3_model = None

        moondream3_model = transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )

        moondream3_model.eval()  # required: trust_remote_code model

        # Initialize KV caches before moving to device (they're lazy by default)
        if hasattr(moondream3_model, '_setup_caches'):
            moondream3_model._setup_caches() # pylint: disable=protected-access

        # Disable flex_attention decoding (can cause hangs due to torch.compile)
        if hasattr(moondream3_model, 'model') and hasattr(moondream3_model.model, 'use_flex_decoding'):
            moondream3_model.model.use_flex_decoding = False

        loaded = repo
        devices.torch_gc()

    # Move model to active device
    sd_models.move_model(moondream3_model, devices.device)
    return moondream3_model


def _image_hash(image: Image.Image) -> str:
    """Content-based hash for cache keys using image size and pixel sample."""
    h = hashlib.md5(usedforsecurity=False)
    h.update(f"{image.size}{image.mode}".encode())
    h.update(image.tobytes()[:4096])
    return h.hexdigest()


def encode_image(image: Image.Image, cache_key: str = None):
    """
    Encode image for reuse across multiple queries.

    Args:
        image: PIL Image
        cache_key: Optional cache key for storing encoded image

    Returns:
        Encoded image tensor
    """
    if cache_key and cache_key in image_cache:
        image_cache.move_to_end(cache_key)  # LRU: mark as recently used
        debug(f'VQA caption: handler=moondream3 using cached encoding for cache_key="{cache_key}"')
        return image_cache[cache_key]

    model = load_model(loaded)

    with devices.inference_context():
        encoded = model.encode_image(image)

    if cache_key:
        image_cache[cache_key] = encoded
        while len(image_cache) > IMAGE_CACHE_MAX:
            evicted_key, _ = image_cache.popitem(last=False)  # Evict oldest
            debug(f'VQA caption: handler=moondream3 evicted cache_key="{evicted_key}" cache_size={len(image_cache)}')
        debug(f'VQA caption: handler=moondream3 cached encoding cache_key="{cache_key}" cache_size={len(image_cache)}')

    return encoded


def query(image: Image.Image, question: str, repo: str, stream: bool = False,
          temperature: float = None, top_p: float = None, max_tokens: int = None,
          use_cache: bool = False, reasoning: bool = True):
    """
    Visual question answering with optional streaming.

    Args:
        image: PIL Image
        question: Question about the image
        repo: Model repository
        stream: Enable streaming output (generator)
        temperature: Sampling temperature (overrides global setting)
        top_p: Nucleus sampling parameter (overrides global setting)
        max_tokens: Maximum tokens to generate (overrides global setting)
        use_cache: Use cached image encoding if available

    Returns:
        Answer dict or string (or generator if stream=True)
    """
    model = load_model(repo)

    # Build settings - per-call parameters override global settings
    settings = get_settings() or {}
    if temperature is not None:
        settings['temperature'] = temperature
    if top_p is not None:
        settings['top_p'] = top_p
    if max_tokens is not None:
        settings['max_tokens'] = max_tokens

    debug(f'VQA caption: handler=moondream3 method=query question="{question}" stream={stream} settings={settings}')

    # Use cached encoding if requested
    if use_cache:
        cache_key = f"{_image_hash(image)}_{question}"
        image_input = encode_image(image, cache_key)
    else:
        image_input = image

    with devices.inference_context():
        response = model.query(
            image=image_input,
            question=question,
            stream=stream,
            settings=settings if settings else None,
            reasoning=reasoning
        )

    # Log response structure (for non-streaming)
    if not stream:
        if isinstance(response, dict):
            debug(f'VQA caption: handler=moondream3 response_type=dict keys={list(response.keys())}')
            if 'reasoning' in response:
                reasoning_text = response['reasoning'].get('text', '')[:100] + '...' if len(response['reasoning'].get('text', '')) > 100 else response['reasoning'].get('text', '')
                debug(f'VQA caption: handler=moondream3 reasoning="{reasoning_text}"')
            if 'answer' in response:
                debug(f'VQA caption: handler=moondream3 answer="{response["answer"]}"')

    return response


def caption(image: Image.Image, repo: str, length: str = 'normal', stream: bool = False,
            temperature: float = None, top_p: float = None, max_tokens: int = None):
    """
    Generate image captions at different lengths.

    Args:
        image: PIL Image
        repo: Model repository
        length: Caption length - 'short', 'normal', or 'long'
        stream: Enable streaming output (generator)
        temperature: Sampling temperature (overrides global setting)
        top_p: Nucleus sampling parameter (overrides global setting)
        max_tokens: Maximum tokens to generate (overrides global setting)

    Returns:
        Caption dict or string (or generator if stream=True)
    """
    model = load_model(repo)

    # Build settings - per-call parameters override global settings
    settings = get_settings() or {}
    if temperature is not None:
        settings['temperature'] = temperature
    if top_p is not None:
        settings['top_p'] = top_p
    if max_tokens is not None:
        settings['max_tokens'] = max_tokens

    debug(f'VQA caption: handler=moondream3 method=caption length={length} stream={stream} settings={settings}')

    with devices.inference_context():
        response = model.caption(
            image,
            length=length,
            stream=stream,
            settings=settings if settings else None
        )

    # Log response structure (for non-streaming)
    if not stream and isinstance(response, dict):
        debug(f'VQA caption: handler=moondream3 response_type=dict keys={list(response.keys())}')

    return response


def point(image: Image.Image, object_name: str, repo: str):
    """
    Identify coordinates of all instances of a specific object in the image.

    Args:
        image: PIL Image
        object_name: Name of object to locate
        repo: Model repository

    Returns:
        List of (x, y) tuples with coordinates normalized to 0-1 range, or None if not found
        Example: [(0.733, 0.442), (0.5, 0.6)] for 2 instances
    """
    model = load_model(repo)

    debug(f'VQA caption: handler=moondream3 method=point object_name="{object_name}"')

    with devices.inference_context():
        result = model.point(image, object_name)

    debug(f'VQA caption: handler=moondream3 point_raw_result="{result}" type={type(result)}')
    if isinstance(result, dict):
        debug(f'VQA caption: handler=moondream3 point_raw_result_keys={list(result.keys())}')

    points = vqa_detection.parse_points(result)
    if points:
        debug(f'VQA caption: handler=moondream3 point_result={len(points)} points found')
        return points

    debug('VQA caption: handler=moondream3 point_result=not found')
    return None


def detect(image: Image.Image, object_name: str, repo: str, max_objects: int = 10):
    """
    Detect all instances of a specific object with bounding boxes.

    Args:
        image: PIL Image
        object_name: Name of object to detect
        repo: Model repository
        max_objects: Maximum number of objects to return

    Returns:
        List of detection dicts with keys:
        - 'bbox': [x1, y1, x2, y2] normalized to 0-1
        - 'label': Object label
        - 'confidence': Detection confidence (0-1)
        Returns empty list if no objects found.
    """
    model = load_model(repo)

    debug(f'VQA caption: handler=moondream3 method=detect object_name="{object_name}" max_objects={max_objects}')

    with devices.inference_context():
        result = model.detect(image, object_name)

    debug(f'VQA caption: handler=moondream3 detect_raw_result="{result}" type={type(result)}')
    if isinstance(result, dict):
        debug(f'VQA caption: handler=moondream3 detect_raw_result_keys={list(result.keys())}')

    detections = vqa_detection.parse_detections(result, object_name, max_objects)
    debug(f'VQA caption: handler=moondream3 detect_result={len(detections)} objects found')
    return detections


def predict(question: str, image: Image.Image, repo: str, model_name: str = None, thinking_mode: bool = False,
            mode: str = None, stream: bool = False, use_cache: bool = False, **kwargs):
    """
    Main entry point for Moondream 3 VQA - auto-detects mode from question.

    Args:
        question: The question/prompt (e.g., "caption", "where is the cat?", "describe this")
        image: PIL Image
        repo: Model repository
        model_name: Display name for logging
        thinking_mode: Enable reasoning mode for query
        mode: Force specific mode ('query', 'caption', 'caption_short', 'caption_long', 'point', 'detect')
        stream: Enable streaming output (for query/caption)
        use_cache: Use cached image encoding (for query)
        **kwargs: Additional parameters (max_objects for detect, etc.)

    Returns:
        Response string (detection data stored on VQA singleton instance.last_detection_data)
        (or generator if stream=True for query/caption modes)
    """
    debug(f'VQA caption: handler=moondream3 model_name="{model_name}" repo="{repo}" question="{question}" image_size={image.size if image else None} mode={mode} stream={stream}')

    # Clean question
    question = question.replace('<', '').replace('>', '').replace('_', ' ') if question else ''

    # Auto-detect mode from question if not specified
    if mode is None:
        question_lower = question.lower()

        # Caption detection
        if question in ['CAPTION', 'caption'] or 'caption' in question_lower:
            if 'more detailed' in question_lower or 'very long' in question_lower:
                mode = 'caption_long'
            elif 'detailed' in question_lower or 'long' in question_lower:
                mode = 'caption_normal'
            elif 'short' in question_lower or 'brief' in question_lower:
                mode = 'caption_short'
            else:
                # Default caption mode (matches vqa.py legacy behavior)
                if question == 'CAPTION':
                    mode = 'caption_short'
                elif question == 'DETAILED CAPTION':
                    mode = 'caption_normal'
                elif question == 'MORE DETAILED CAPTION':
                    mode = 'caption_long'
                else:
                    mode = 'caption_normal'

        # Point detection
        elif 'where is' in question_lower or 'locate' in question_lower or 'find' in question_lower or 'point' in question_lower:
            mode = 'point'

        # Object detection
        elif 'detect' in question_lower or 'bounding box' in question_lower or 'bbox' in question_lower:
            mode = 'detect'

        # Default to query
        else:
            mode = 'query'

    debug(f'VQA caption: handler=moondream3 mode_selected={mode}')

    # Dispatch to appropriate method
    try:
        if mode == 'caption_short':
            response = caption(image, repo, length='short', stream=stream)
        elif mode == 'caption_long':
            response = caption(image, repo, length='long', stream=stream)
        elif mode in ['caption', 'caption_normal']:
            response = caption(image, repo, length='normal', stream=stream)
        elif mode == 'point':
            # Extract object name from question - case insensitive, preserve object names
            object_name = question
            for phrase in ['point at', 'where is', 'locate', 'find']:
                object_name = re.sub(rf'\b{phrase}\b', '', object_name, flags=re.IGNORECASE)
            object_name = re.sub(r'[?.!,]', '', object_name).strip()
            object_name = re.sub(r'^\s*the\s+', '', object_name, flags=re.IGNORECASE)
            debug(f'VQA caption: handler=moondream3 point_extracted_object="{object_name}"')
            result = point(image, object_name, repo)
            if result:
                from modules.caption import vqa
                vqa.get_instance().last_detection_data = {'points': result}
                return vqa_detection.format_points_text(result)
            return "Object not found"
        elif mode == 'detect':
            # Extract object name from question - case insensitive
            object_name = question
            for phrase in ['detect', 'find all', 'bounding box', 'bbox', 'find']:
                object_name = re.sub(rf'\b{phrase}\b', '', object_name, flags=re.IGNORECASE)
            object_name = re.sub(r'[?.!,]', '', object_name).strip()
            object_name = re.sub(r'^\s*the\s+', '', object_name, flags=re.IGNORECASE)
            if ' and ' in object_name.lower():
                object_name = re.split(r'\s+and\s+', object_name, flags=re.IGNORECASE)[0].strip()
            debug(f'VQA caption: handler=moondream3 detect_extracted_object="{object_name}"')

            results = detect(image, object_name, repo, max_objects=kwargs.get('max_objects', 10))
            if results:
                from modules.caption import vqa
                vqa.get_instance().last_detection_data = {'detections': results}
                return vqa_detection.format_detections_text(results)
            return "No objects detected"
        else:  # mode == 'query'
            if len(question) < 2:
                question = "Describe this image."
            response = query(image, question, repo, stream=stream, use_cache=use_cache, reasoning=thinking_mode)

        debug(f'VQA caption: handler=moondream3 response_before_clean="{response}"')
        return response

    except Exception as e:
        from modules import errors
        errors.display(e, 'Moondream3')
        return f"Error: {str(e)}"
    finally:
        if shared.opts.caption_offload and moondream3_model is not None:
            sd_models.move_model(moondream3_model, devices.cpu, force=True)


def clear_cache():
    """Clear image encoding cache."""
    cache_size = len(image_cache)
    image_cache.clear()
    debug(f'VQA caption: handler=moondream3 cleared image cache cache_size_was={cache_size}')
    shared.log.debug(f'Moondream3: Cleared image cache ({cache_size} entries)')


def unload():
    """Release Moondream 3 model from GPU/memory."""
    global moondream3_model, loaded  # pylint: disable=global-statement
    if moondream3_model is not None:
        shared.log.debug(f'Moondream3 unload: model="{loaded}"')
        sd_models.move_model(moondream3_model, devices.cpu, force=True)
        moondream3_model = None
        loaded = None
        clear_cache()
        devices.torch_gc(force=True)
    else:
        shared.log.debug('Moondream3 unload: no model loaded')
