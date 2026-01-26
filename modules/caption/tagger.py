# Unified Tagger Interface - Dispatches to WaifuDiffusion or DeepBooru based on model selection
# Provides a common interface for the Booru Tags tab

from modules import shared

DEEPBOORU_MODEL = "DeepBooru"


def get_models() -> list:
    """Return combined list: DeepBooru + WaifuDiffusion models."""
    from modules.caption import waifudiffusion
    return [DEEPBOORU_MODEL] + waifudiffusion.get_models()


def refresh_models() -> list:
    """Refresh and return all models."""
    return get_models()


def is_deepbooru(model_name: str) -> bool:
    """Check if selected model is DeepBooru."""
    return model_name == DEEPBOORU_MODEL


def load_model(model_name: str) -> bool:
    """Load appropriate backend."""
    if is_deepbooru(model_name):
        from modules.caption import deepbooru
        return deepbooru.load_model()
    else:
        from modules.caption import waifudiffusion
        return waifudiffusion.load_model(model_name)


def unload_model():
    """Unload both backends to ensure memory is freed."""
    from modules.caption import deepbooru, waifudiffusion
    deepbooru.unload_model()
    waifudiffusion.unload_model()


def tag(image, model_name: str = None, **kwargs) -> str:
    """Unified tagging - dispatch to correct backend.

    Args:
        image: PIL Image to tag
        model_name: Model to use (DeepBooru or WaifuDiffusion model name)
        **kwargs: Additional arguments passed to the backend

    Returns:
        Formatted tag string
    """
    if model_name is None:
        model_name = shared.opts.waifudiffusion_model

    if is_deepbooru(model_name):
        from modules.caption import deepbooru
        return deepbooru.tag(image, **kwargs)
    else:
        from modules.caption import waifudiffusion
        return waifudiffusion.tag(image, model_name=model_name, **kwargs)


def batch(model_name: str, **kwargs) -> str:
    """Unified batch processing.

    Args:
        model_name: Model to use (DeepBooru or WaifuDiffusion model name)
        **kwargs: Additional arguments passed to the backend

    Returns:
        Combined tag results
    """
    if is_deepbooru(model_name):
        from modules.caption import deepbooru
        return deepbooru.batch(model_name=model_name, **kwargs)
    else:
        from modules.caption import waifudiffusion
        return waifudiffusion.batch(model_name=model_name, **kwargs)
