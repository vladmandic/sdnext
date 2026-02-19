"""Loaded Models Inventory — enumerates all models currently in memory.

Returns a list of loaded models across all subsystems (pipeline, ControlNet,
LoRA, upscalers, IP-adapter, caption, detailer) with device/size/dtype info.
"""

from typing import List
import torch
from modules import shared
from modules.api.models import ItemLoadedModel


def _safe_device(model) -> str | None:
    try:
        if hasattr(model, 'device'):
            return str(model.device)
        p = next(model.parameters(), None)
        if p is not None:
            return str(p.device)
    except Exception:
        pass
    return None


def _safe_compute_dtype(model) -> str | None:
    """Get the compute dtype (e.g. bfloat16) from model.dtype or first parameter."""
    try:
        if hasattr(model, 'dtype') and model.dtype is not None:
            return str(model.dtype).replace('torch.', '')
        p = next(model.parameters(), None)
        if p is not None:
            return str(p.dtype).replace('torch.', '')
    except Exception:
        pass
    return None


def _safe_quant_dtype(model) -> str | None:
    """Get the quantization weight dtype (e.g. uint4) from quantization_config.weights_dtype."""
    try:
        qcfg = getattr(model, 'quantization_config', None)
        if qcfg is not None:
            wd = getattr(qcfg, 'weights_dtype', None)
            if wd is not None:
                return str(wd)
            # BitsAndBytes: check load_in_4bit / load_in_8bit
            if getattr(qcfg, 'load_in_4bit', False):
                return 'nf4' if getattr(qcfg, 'bnb_4bit_quant_type', '') == 'nf4' else 'int4'
            if getattr(qcfg, 'load_in_8bit', False):
                return 'int8'
    except Exception:
        pass
    return None


def _safe_quant_method(model) -> str | None:
    """Get the quantization method name (e.g. SDNQ, BNB, QUANTO)."""
    try:
        q = getattr(model, 'quantization_method', None)
        if q is not None:
            s = str(q)
            return s.split('.')[-1] if '.' in s else s
    except Exception:
        pass
    return None


def _safe_dtype(model) -> str | None:
    """Return the effective dtype: quant weight dtype if quantized, else compute dtype."""
    return _safe_quant_dtype(model) or _safe_compute_dtype(model)


def _safe_size(model) -> int | None:
    try:
        return sum(p.nelement() * p.element_size() for p in model.parameters())
    except Exception:
        return None


def _component_extra(comp, **kwargs) -> dict:
    extra = dict(kwargs)
    quant = _safe_quant_method(comp)
    if quant:
        extra['quant'] = quant
    compute_dtype = _safe_compute_dtype(comp)
    quant_dtype = _safe_quant_dtype(comp)
    if quant_dtype and compute_dtype and compute_dtype != quant_dtype:
        extra['compute_dtype'] = compute_dtype
    return extra


def _enumerate_pipeline(pipe, role: str) -> List[ItemLoadedModel]:
    items = []
    if pipe is None:
        return items
    # Pipeline entry
    name = 'unknown'
    checkpoint_info = getattr(pipe, 'sd_checkpoint_info', None)
    if checkpoint_info is not None:
        name = getattr(checkpoint_info, 'name', None) or getattr(checkpoint_info, 'title', 'unknown')
    items.append(ItemLoadedModel(
        name=name,
        category='pipeline',
        device=_safe_device(pipe) if isinstance(pipe, torch.nn.Module) else None,
        size_bytes=None,
        dtype=None,
        extra={'role': role, 'class': pipe.__class__.__name__},
    ))
    # Sub-components
    components = getattr(pipe, 'components', None)
    if components and isinstance(components, dict):
        for comp_name, comp in components.items():
            if comp is None or not isinstance(comp, torch.nn.Module):
                continue
            items.append(ItemLoadedModel(
                name=comp_name,
                category='component',
                device=_safe_device(comp),
                size_bytes=_safe_size(comp),
                dtype=_safe_dtype(comp),
                extra=_component_extra(comp, role=role, **{'class': comp.__class__.__name__}),
            ))
    return items


def _enumerate_control() -> List[ItemLoadedModel]:
    items = []
    try:
        from modules.control import unit as control_unit
        for u in control_unit.current:
            # ControlNet
            cn = getattr(u, 'controlnet', None)
            if cn is not None:
                model = getattr(cn, 'model', None)
                if model is not None and isinstance(model, torch.nn.Module):
                    model_id = getattr(cn, 'model_id', None) or 'unknown'
                    items.append(ItemLoadedModel(
                        name=model_id,
                        category='controlnet',
                        device=_safe_device(model),
                        size_bytes=_safe_size(model),
                        dtype=_safe_dtype(model),
                        extra=_component_extra(model, **{'class': model.__class__.__name__}),
                    ))
            # T2I Adapter
            adapter = getattr(u, 'adapter', None)
            if adapter is not None:
                model = getattr(adapter, 'model', None)
                if model is not None and isinstance(model, torch.nn.Module):
                    model_id = getattr(adapter, 'model_id', None) or 'unknown'
                    items.append(ItemLoadedModel(
                        name=model_id,
                        category='t2iadapter',
                        device=_safe_device(model),
                        size_bytes=_safe_size(model),
                        dtype=_safe_dtype(model),
                        extra=_component_extra(model, **{'class': model.__class__.__name__}),
                    ))
    except Exception:
        pass
    return items


def _enumerate_lora() -> List[ItemLoadedModel]:
    items = []
    try:
        from modules.lora import lora_common, lora_load
        for net in lora_common.loaded_networks:
            items.append(ItemLoadedModel(
                name=getattr(net, 'name', 'unknown'),
                category='lora',
            ))
        active_names = {getattr(n, 'name', None) for n in lora_common.loaded_networks}
        for name in lora_load.lora_cache:
            if name not in active_names:
                items.append(ItemLoadedModel(
                    name=name,
                    category='lora_cached',
                ))
    except Exception:
        pass
    return items


def _enumerate_ipadapter() -> List[ItemLoadedModel]:
    items = []
    try:
        from modules import ipadapter
        if ipadapter.clip_loaded:
            items.append(ItemLoadedModel(
                name=ipadapter.clip_loaded,
                category='ipadapter',
                extra={'type': 'clip_encoder'},
            ))
        for name in ipadapter.adapters_loaded:
            items.append(ItemLoadedModel(
                name=name,
                category='ipadapter',
                extra={'type': 'adapter'},
            ))
    except Exception:
        pass
    return items


def _enumerate_upscalers() -> List[ItemLoadedModel]:
    items = []
    try:
        for upscaler in shared.sd_upscalers:
            # ESRGAN-style: dict cache in .models
            models_cache = getattr(upscaler, 'models', None)
            if isinstance(models_cache, dict):
                for path, model in models_cache.items():
                    if model is not None and isinstance(model, torch.nn.Module):
                        items.append(ItemLoadedModel(
                            name=getattr(upscaler, 'name', 'unknown'),
                            category='upscaler',
                            device=_safe_device(model),
                            size_bytes=_safe_size(model),
                            dtype=_safe_dtype(model),
                            extra={'class': model.__class__.__name__, 'path': str(path)},
                        ))
            # Spandrel/SeedVR-style: single .model attribute
            model = getattr(upscaler, 'model', None)
            if model is not None and isinstance(model, torch.nn.Module):
                items.append(ItemLoadedModel(
                    name=getattr(upscaler, 'name', 'unknown'),
                    category='upscaler',
                    device=_safe_device(model),
                    size_bytes=_safe_size(model),
                    dtype=_safe_dtype(model),
                    extra={'class': model.__class__.__name__},
                ))
    except Exception:
        pass
    return items


def _enumerate_detailer() -> List[ItemLoadedModel]:
    items = []
    try:
        yolo = getattr(shared, 'yolo', None)
        if yolo is not None:
            models_cache = getattr(yolo, 'models', None)
            if isinstance(models_cache, dict):
                for name, model in models_cache.items():
                    if model is not None:
                        items.append(ItemLoadedModel(
                            name=name,
                            category='detailer',
                            device=_safe_device(model) if isinstance(model, torch.nn.Module) else None,
                            size_bytes=_safe_size(model) if isinstance(model, torch.nn.Module) else None,
                            dtype=_safe_dtype(model) if isinstance(model, torch.nn.Module) else None,
                        ))
    except Exception:
        pass
    return items


def _enumerate_prompt_enhance() -> List[ItemLoadedModel]:
    items = []
    try:
        from modules.scripts_manager import scripts_txt2img
        if scripts_txt2img is None:
            return items
        instances = [s for s in scripts_txt2img.scripts if 'prompt_enhance.py' in s.filename]
        if not instances:
            return items
        instance = instances[0]
        if instance.llm is not None:
            items.append(ItemLoadedModel(
                name=instance.model or 'unknown',
                category='enhance',
                device=_safe_device(instance.llm),
                size_bytes=_safe_size(instance.llm),
                dtype=_safe_dtype(instance.llm),
                extra=_component_extra(instance.llm, **{'class': instance.llm.__class__.__name__}),
            ))
    except Exception:
        pass
    return items


def _enumerate_caption() -> List[ItemLoadedModel]:
    items = []
    # VQA
    try:
        from modules.caption import vqa
        instance = vqa._instance  # pylint: disable=protected-access
        if instance is not None and instance.model is not None:
            items.append(ItemLoadedModel(
                name=instance.loaded or 'unknown',
                category='caption',
                device=_safe_device(instance.model) if isinstance(instance.model, torch.nn.Module) else None,
                size_bytes=_safe_size(instance.model) if isinstance(instance.model, torch.nn.Module) else None,
                dtype=_safe_dtype(instance.model) if isinstance(instance.model, torch.nn.Module) else None,
                extra={'type': 'vqa'},
            ))
    except Exception:
        pass
    # OpenCLIP
    try:
        from modules.caption import openclip
        if openclip.ci is not None:
            items.append(ItemLoadedModel(
                name='openclip',
                category='caption',
                extra={'type': 'openclip'},
            ))
    except Exception:
        pass
    # DeepBooru
    try:
        from modules.caption import deepbooru
        if deepbooru.model.model is not None:
            items.append(ItemLoadedModel(
                name='DeepDanbooru',
                category='caption',
                device=_safe_device(deepbooru.model.model),
                size_bytes=_safe_size(deepbooru.model.model),
                dtype=_safe_dtype(deepbooru.model.model),
                extra={'type': 'deepbooru', 'class': deepbooru.model.model.__class__.__name__},
            ))
    except Exception:
        pass
    # WaifuDiffusion
    try:
        from modules.caption import waifudiffusion
        if waifudiffusion.tagger.session is not None:
            items.append(ItemLoadedModel(
                name=waifudiffusion.tagger.model_name or 'unknown',
                category='caption',
                device=None,
                size_bytes=None,
                dtype=None,
                extra={'type': 'waifudiffusion', 'runtime': 'onnx'},
            ))
    except Exception:
        pass
    return items


def get_loaded_models() -> List[ItemLoadedModel]:
    """
    Enumerate all models currently loaded in memory.

    Returns entries for the main pipeline and refiner (with sub-components), ControlNet,
    LoRA, IP-Adapter, upscaler, detailer, and caption models. Each entry includes
    category, device, size in bytes, dtype, and quantization info where available.
    """
    items = []
    # Main pipeline + refiner
    try:
        from modules.modeldata import model_data
        items.extend(_enumerate_pipeline(model_data.sd_model, 'main'))
        items.extend(_enumerate_pipeline(model_data.sd_refiner, 'refiner'))
    except Exception:
        pass
    items.extend(_enumerate_control())
    items.extend(_enumerate_lora())
    items.extend(_enumerate_ipadapter())
    items.extend(_enumerate_upscalers())
    items.extend(_enumerate_detailer())
    items.extend(_enumerate_caption())
    items.extend(_enumerate_prompt_enhance())
    return items


def register_api():
    from modules.shared import api as api_instance
    api_instance.add_api_route("/sdapi/v1/loaded-models", get_loaded_models, methods=["GET"], response_model=List[ItemLoadedModel], tags=["Models"])
