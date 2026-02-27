import asyncio
from fastapi import APIRouter, HTTPException
from modules.api.v2.models import (
    ItemPromptEnhanceModelV2,
    ReqPromptEnhanceV2, ResPromptEnhanceV2,
)

router = APIRouter(prefix="/sdapi/v2", tags=["Prompt Enhance"])

_ENHANCE_GROUP_RULES = [
    # Gemma official (google/ only)
    ("google/gemma", "Gemma"),
    # Gemma Finetunes
    ("nidum", "Gemma Finetunes"),
    ("allura", "Gemma Finetunes"),
    ("davidau/gemma", "Gemma Finetunes"),
    ("coder3101/gemma", "Gemma Finetunes"),
    ("coder3101/big-tiger-gemma", "Gemma Finetunes"),
    ("p-e-w/gemma", "Gemma Finetunes"),
    # Qwen3.5
    ("qwen/qwen3.5", "Qwen3.5"),
    ("qwen3.5", "Qwen3.5 Finetunes"),
    # Qwen3-VL official vs finetunes
    ("qwen/qwen3-vl", "Qwen3-VL"),
    ("qwen3-vl", "Qwen3-VL Finetunes"),
    # Qwen3
    ("qwen3", "Qwen3"),
    # Qwen2.5-VL official vs finetunes
    ("qwen/qwen2.5-vl", "Qwen2.5-VL"),
    ("qwen2.5-vl", "Qwen2.5-VL Finetunes"),
    ("qwen2.5-omni", "Qwen2.5-VL"),
    # Qwen2.5
    ("qwen2.5", "Qwen2.5"),
    ("qwen2-vl", "Qwen2-VL"),
    ("qwen2", "Qwen2"),
    ("qwen", "Qwen"),
    # Mistral official
    ("mistralai/", "Mistral"),
    # Mistral Finetunes
    ("mistral", "Mistral Finetunes"),
    # Llama
    ("llama", "Llama"),
    ("dolphin", "Llama"),
    # SmolLM
    ("smollm", "SmolLM"),
    # Phi
    ("phi", "Phi"),
]


def _enhance_group(repo: str) -> str:
    lower = repo.lower()
    for pattern, group in _ENHANCE_GROUP_RULES:
        if pattern in lower:
            return group
    return "Other"


@router.get("/prompt-enhance/models", response_model=list[ItemPromptEnhanceModelV2])
async def get_prompt_enhance_models_v2():
    """List available prompt enhancement models with capability flags."""
    from scripts.prompt_enhance import Options, is_vision_model, is_thinking_model
    return [
        ItemPromptEnhanceModelV2(name=repo, group=_enhance_group(repo), vision=is_vision_model(repo), thinking=is_thinking_model(repo))
        for repo in Options.models
    ]


@router.post("/prompt-enhance", response_model=ResPromptEnhanceV2)
async def post_prompt_enhance_v2(req: ReqPromptEnhanceV2):
    """Enhance a prompt using an LLM. Supports text, image-conditioned, and video prompt enhancement modes."""
    def _run():
        from modules import processing_helpers
        from modules.api.helpers import decode_base64_to_image
        seed = req.seed or -1
        seed = processing_helpers.get_fixed_seed(seed)
        prompt = ''
        if req.type in ('text', 'image'):
            from modules.scripts_manager import scripts_txt2img
            default_model = 'google/gemma-3-4b-it' if req.type == 'image' else 'google/gemma-3-1b-it'
            model = default_model if req.model is None or len(req.model) < 4 else req.model
            matches = [s for s in scripts_txt2img.scripts if 'prompt_enhance.py' in s.filename]
            if not matches:
                raise HTTPException(status_code=503, detail="Prompt enhance script not loaded")
            instance = matches[0]
            prompt = instance.enhance(
                model=model,
                prompt=req.prompt,
                system=req.system_prompt,
                prefix=req.prefix,
                suffix=req.suffix,
                sample=req.do_sample,
                tokens=req.max_tokens,
                temperature=req.temperature,
                penalty=req.repetition_penalty,
                top_k=req.top_k,
                top_p=req.top_p,
                thinking=req.thinking,
                keep_thinking=req.keep_thinking,
                use_vision=req.use_vision,
                prefill=req.prefill or '',
                keep_prefill=req.keep_prefill,
                image=decode_base64_to_image(req.image) if req.image else None,
                seed=seed,
                nsfw=req.nsfw,
            )
        elif req.type == 'video':
            from modules.ui_video_vlm import enhance_prompt
            model = 'Google Gemma 3 4B' if req.model is None or len(req.model) < 4 else req.model
            prompt = enhance_prompt(
                enable=True,
                image=decode_base64_to_image(req.image) if req.image else None,
                prompt=req.prompt,
                model=model,
                system_prompt=req.system_prompt,
                nsfw=req.nsfw,
            )
        else:
            raise HTTPException(status_code=400, detail="prompt enhancement: invalid type")
        return prompt, seed
    prompt, seed = await asyncio.to_thread(_run)
    return ResPromptEnhanceV2(ok=True, prompt=prompt, seed=seed)
