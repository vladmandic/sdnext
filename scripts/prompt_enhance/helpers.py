import base64
import io
import gradio as gr
from modules import ui_symbols
from .options import Options


def b64(image):
    if image is None:
        return ''
    if isinstance(image, gr.Image): # should not happen
        return None
    with io.BytesIO() as stream:
        image.convert('RGB').save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def is_cloud_model(model_name: str) -> bool:
    if not model_name:
        return False
    return model_name in Options.cloud


def is_vision_model(model_name: str) -> bool:
    """Check if model supports vision/image input."""
    if not model_name:
        return False
    return model_name in Options.img2img or model_name in Options.cloud


def is_thinking_model(model_name: str) -> bool:
    """Check if model supports thinking/reasoning mode."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    # Match VQA's detection patterns for consistency
    thinking_indicators = [
        'thinking',      # Qwen3-VL-*-Thinking models
        'reasoning',     # Ministral-3-*-Reasoning models
        'moondream3',    # Moondream 3 supports thinking
        'moondream 3',
        'moondream2',    # Moondream 2 supports reasoning mode
        'moondream 2',
        'mimo',          # XiaomiMiMo models
        'qwen3.5',      # Qwen3.5 native thinking (repo names)
        'qwen 3.5',     # Qwen3.5 native thinking (display names)
    ]
    return any(indicator in model_lower for indicator in thinking_indicators)


def get_model_display_name(model_repo: str) -> str:
    """Generate display name with vision/reasoning symbols."""
    symbols = []
    if model_repo in Options.img2img:
        symbols.append(ui_symbols.vision)
    if model_repo in Options.cloud:
        symbols.append(ui_symbols.cloud)
    if is_thinking_model(model_repo):
        symbols.append(ui_symbols.reasoning)
    return f"{model_repo} {' '.join(symbols)}" if symbols else model_repo


def get_model_repo_from_display(display_name: str) -> str:
    """Strip symbols from display name to get repo."""
    if not display_name:
        return display_name
    result = display_name
    for symbol in [ui_symbols.vision, ui_symbols.reasoning, ui_symbols.cloud]:
        result = result.replace(symbol, '')
    return result.strip()


def keep_think_block_open(text_prompt: str) -> str:
    """Remove closing </think> so model can continue reasoning with prefill."""
    think_open = "<think>"
    think_close = "</think>"
    last_open = text_prompt.rfind(think_open)
    if last_open == -1:
        return text_prompt
    close_index = text_prompt.find(think_close, last_open)
    if close_index == -1:
        return text_prompt
    end_close = close_index + len(think_close)
    while end_close < len(text_prompt) and text_prompt[end_close] in ' \t\r\n':
        end_close += 1
    return text_prompt[:close_index] + text_prompt[end_close:]
