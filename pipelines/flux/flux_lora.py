def calculate_module_shape(model, base_module=None, base_weight_param_name=None):
    def _get_weight_shape(weight):
        if weight.__class__.__name__ == "Params4bit":
            return weight.quant_state.shape
        elif weight.__class__.__name__ == "GGUFParameter":
            return weight.quant_shape
        else:
            return weight.shape

    if base_module is not None:
        if hasattr(base_module, "sdnq_dequantizer"):
            return base_module.sdnq_dequantizer.original_shape
        else:
            return _get_weight_shape(base_module.weight)
    elif base_weight_param_name is not None:
        from diffusers.utils import get_submodule_by_name
        if not base_weight_param_name.endswith(".weight"):
            raise ValueError(f"Invalid `base_weight_param_name` passed as it does not end with '.weight' {base_weight_param_name=}.")
        module_path = base_weight_param_name.rsplit(".weight", 1)[0]
        submodule = get_submodule_by_name(model, module_path)
        if hasattr(submodule, "sdnq_dequantizer"):
            return submodule.sdnq_dequantizer.original_shape
        else:
            return _get_weight_shape(submodule.weight)

    raise ValueError("Either `base_module` or `base_weight_param_name` must be provided.")


def apply_patch():
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    FluxLoraLoaderMixin._calculate_module_shape = calculate_module_shape # pylint: disable=protected-access
