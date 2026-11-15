import torch
from modules.logger import log


ATTENTIONS = ['eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention', 'paged|eager', 'paged|sdpa', 'paged|flash_attention_2', 'paged|flash_attention_3']


def get_first_attention_block(model): # for models still using transformers==4 internal architecture
    m = model.model
    if hasattr(m, "layers"):
        layer = m.layers[0]
        for name in ("self_attn", "attention", "linear_attn"):
            attention = getattr(layer, name, None)
            if attention is not None:
                return attention
        return None
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer.layers[0].attention
    if hasattr(m, "block"):
        return m.block[0].attention
    return None


def get_attention(model: torch.nn.Module):
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        return model.config._attn_implementation # pylint: disable=protected-access
    if hasattr(model, "_attn_implementation"):
        return model._attn_implementation # pylint: disable=protected-access
    if hasattr(model, "_get_attn_implementation"):
        return model._get_attn_implementation() # pylint: disable=protected-access
    first_block = get_first_attention_block(model)
    return first_block.__class__.__name__ if getattr(first_block, "__class__", None) else first_block


def set_attention(model: torch.nn.Module, attention: str|None = None, verify: bool = False):
    if not hasattr(model, 'set_attn_implementation'):
        return None
    supported = []
    unsupported = []
    default = get_attention(model)
    desired = attention or default

    if verify:
        for name in ATTENTIONS: # type: ignore
            try:
                model.set_attn_implementation(name)
                supported.append(name)
            except Exception:
                unsupported.append(name)
        log.debug(f"LLM attention: cls={model.__class__.__name__} attention={desired} default={default} supported={supported} unsupported={unsupported}")
    else:
        log.debug(f"LLM attention: cls={model.__class__.__name__} attention={desired} default={default}")
    if desired != default:
        try:
            model.set_attn_implementation(desired) # restore default or set specified attention
        except Exception:
            pass
    return desired
