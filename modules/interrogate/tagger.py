# Unified Tagger Interface - Dispatches to WD14 or DeepBooru based on model selection
# Provides a common interface for the Booru Tags tab

from modules import shared

DEEPBOORU_MODEL = "DeepBooru"


def get_models() -> list:
    """Return combined list: DeepBooru + WD14 models."""
    from modules.interrogate import wd14
    return [DEEPBOORU_MODEL] + wd14.get_models()


def refresh_models() -> list:
    """Refresh and return all models."""
    return get_models()


def is_deepbooru(model_name: str) -> bool:
    """Check if selected model is DeepBooru."""
    return model_name == DEEPBOORU_MODEL


def load_model(model_name: str) -> bool:
    """Load appropriate backend."""
    if is_deepbooru(model_name):
        from modules.interrogate import deepbooru
        return deepbooru.load_model()
    else:
        from modules.interrogate import wd14
        return wd14.load_model(model_name)


def unload_model():
    """Unload both backends to ensure memory is freed."""
    from modules.interrogate import deepbooru, wd14
    deepbooru.unload_model()
    wd14.unload_model()


def tag(image, model_name: str = None, **kwargs) -> str:
    """Unified tagging - dispatch to correct backend.

    Args:
        image: PIL Image to tag
        model_name: Model to use (DeepBooru or WD14 model name)
        **kwargs: Additional arguments passed to the backend

    Returns:
        Formatted tag string
    """
    if model_name is None:
        model_name = shared.opts.wd14_model

    if is_deepbooru(model_name):
        from modules.interrogate import deepbooru
        return deepbooru.tag(image, **kwargs)
    else:
        from modules.interrogate import wd14
        return wd14.tag(image, model_name=model_name, **kwargs)


def batch(model_name: str, **kwargs) -> str:
    """Unified batch processing.

    Args:
        model_name: Model to use (DeepBooru or WD14 model name)
        **kwargs: Additional arguments passed to the backend

    Returns:
        Combined tag results
    """
    if is_deepbooru(model_name):
        from modules.interrogate import deepbooru
        return deepbooru.batch(model_name=model_name, **kwargs)
    else:
        from modules.interrogate import wd14
        return wd14.batch(model_name=model_name, **kwargs)
