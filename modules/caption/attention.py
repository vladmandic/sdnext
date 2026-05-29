from modules.logger import log


ATTENTIONS = ['eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention', 'paged|eager', 'paged|sdpa', 'paged|flash_attention_2', 'paged|flash_attention_3']


def get_first_attention_block(model): # for models still using transformers==4 internal architecture
    m = model.model
    if hasattr(m, "layers"):
        return m.layers[0].self_attn if hasattr(m.layers[0], "self_attn") else m.layers[0].attention
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer.layers[0].attention
    if hasattr(m, "block"):
        return m.block[0].attention
    return None


def set_attention(model):
    if not hasattr(model, 'set_attn_implementation'):
        return
    supported = []
    unsupported = []
    if hasattr(model, "_attn_implementation"):
        default = model._attn_implementation # pylint: disable=protected-access
    elif hasattr(model, "_get_attn_implementation"):
        default = model._get_attn_implementation() # pylint: disable=protected-access
    else:
        default = get_first_attention_block(model)
        log.debug(f"LLM attention: cls={model.__class__.__name__} default={default} fixed")
        return

    for name in ATTENTIONS: # type: ignore
        try:
            model.set_attn_implementation(name)
            supported.append(name)
        except Exception:
            unsupported.append(name)
    log.debug(f"LLM attention: cls={model.__class__.__name__} default={default} supported={supported} unsupported={unsupported}")
    model.set_attn_implementation(default) # restore default
