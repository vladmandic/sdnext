def hijack_transformers():
    # transformers>=4.56 flattened CLIPTextModel internals; diffusers single-file loader still expects `text_model`.
    return
    try:
        import transformers
        if hasattr(transformers, 'CLIPTextModel') and not hasattr(transformers.CLIPTextModel, 'text_model'):
            transformers.CLIPTextModel.text_model = property(lambda self: self)
    except Exception:
        pass
