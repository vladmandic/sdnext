from dataclasses import dataclass, field
from typing import Optional

from modules.logger import log


@dataclass
class LTXCaps:
    name: str
    repo_cls_name: str
    family: str  # '0.9' or '2.x'
    is_distilled: bool
    is_ltx_2_3: bool
    is_i2v: bool
    supports_input_media: bool
    supports_multi_condition: bool
    supports_image_cond_noise_scale: bool
    supports_decode_timestep: bool
    supports_stg: bool
    supports_audio: bool
    supports_frame_rate_kwarg: bool
    # 2.3 transformer cross-attn reads the other modality's sigma; unset falls back to 2.0's
    # independent-sigma path, which is a joint-distribution mismatch for 2.3 weights.
    use_cross_timestep: bool
    default_cfg: float
    default_steps: int
    default_sampler_shift: float
    default_dynamic_shift: bool
    default_width: int
    default_height: int
    default_frames: int
    default_frame_rate: int
    stg_default_scale: float = 0.0
    stg_default_blocks: list = field(default_factory=list)
    # Dev 2.x trained under cfg + stg + modality + rescale four-way composition;
    # distilled bakes these into its sigma schedule and stays at pipeline identity.
    modality_default_scale: float = 1.0
    guidance_rescale_default: float = 0.0
    supports_canonical_stage2: bool = False
    stage2_dev_lora_repo: Optional[str] = None


CONDITION_CLASSES = {'LTXConditionPipeline', 'LTX2ConditionPipeline'}
LTX2_CLASSES = {'LTX2Pipeline', 'LTX2ImageToVideoPipeline', 'LTX2ConditionPipeline'}
ALL_LTX_CLASSES = {
    'LTXPipeline',
    'LTXImageToVideoPipeline',
    'LTXConditionPipeline',
    'LTX2Pipeline',
    'LTX2ImageToVideoPipeline',
    'LTX2ConditionPipeline',
}


def _repo_cls_name(model_name: str) -> Optional[str]:
    from modules.video_models.models_def import models
    entries = models.get('LTX Video', [])
    for m in entries:
        if m.name == model_name:
            if m.repo_cls is None:
                return None
            return m.repo_cls.__name__
    return None


def get_caps(model_name: str) -> Optional[LTXCaps]:
    if not model_name or model_name == 'None':
        return None
    cls_name = _repo_cls_name(model_name)
    if cls_name is None:
        log.warning(f'LTX caps: model="{model_name}" has no repo_cls registered')
        return None
    if cls_name not in ALL_LTX_CLASSES:
        log.warning(f'LTX caps: model="{model_name}" repo_cls="{cls_name}" is not an LTX pipeline')
        return None

    is_ltx2 = cls_name in LTX2_CLASSES
    family = '2.x' if is_ltx2 else '0.9'
    is_distilled = 'Distilled' in model_name
    is_i2v = 'I2V' in model_name or cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline')
    is_condition_cls = cls_name in CONDITION_CLASSES
    supports_input_media = is_i2v or is_condition_cls
    is_ltx_2_3 = is_ltx2 and '2.3' in model_name

    caps = LTXCaps(
        name=model_name,
        repo_cls_name=cls_name,
        family=family,
        is_distilled=is_distilled,
        is_ltx_2_3=is_ltx_2_3,
        is_i2v=is_i2v,
        supports_input_media=supports_input_media,
        supports_multi_condition=is_condition_cls,
        supports_image_cond_noise_scale=(cls_name == 'LTXConditionPipeline'),
        supports_decode_timestep=(family == '0.9'),
        supports_stg=is_ltx2,
        supports_audio=is_ltx2,
        supports_frame_rate_kwarg=is_ltx2,
        use_cross_timestep=is_ltx_2_3,
        default_cfg=3.0,
        default_steps=30 if is_ltx2 else 50,
        default_sampler_shift=-1.0,
        default_dynamic_shift=is_ltx2,
        default_width=768,
        default_height=512,
        default_frames=121 if is_ltx2 else 161,
        default_frame_rate=24 if is_ltx2 else 25,
    )

    if is_distilled:
        caps.default_cfg = 1.0
        caps.default_steps = 8

    if is_ltx2 and not is_distilled:
        if is_ltx_2_3:
            caps.stage2_dev_lora_repo = 'CalamitousFelicitousness/LTX-2.3-distilled-lora-384-Diffusers'
        elif '2.0' in model_name:
            caps.stage2_dev_lora_repo = 'CalamitousFelicitousness/LTX-2.0-distilled-lora-384-Diffusers'
    caps.supports_canonical_stage2 = caps.stage2_dev_lora_repo is not None

    if is_ltx2:
        if '2.3' in model_name:
            caps.stg_default_blocks = [28]
        elif '2.0' in model_name:
            caps.stg_default_blocks = [29]
        else:
            caps.stg_default_blocks = [28]
        if not is_distilled:
            # canonical T2V composition from huggingface/diffusers#13217
            caps.stg_default_scale = 1.0
            caps.modality_default_scale = 3.0
            caps.guidance_rescale_default = 0.7

    return caps
