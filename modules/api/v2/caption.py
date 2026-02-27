import asyncio
from fastapi import APIRouter
from modules.api.v2.models import (
    ReqOpenClipV2, ResOpenClipV2,
    ReqVqaV2, ResVqaV2, ItemVlmModelV2,
    ReqTaggerV2, ResTaggerV2, ItemTaggerModelV2,
)

router = APIRouter(prefix="/sdapi/v2", tags=["Caption"])


@router.get("/caption/openclip/models", response_model=list[str])
async def get_openclip_models_v2():
    """List available OpenCLIP models."""
    from modules.api.caption import get_caption
    return get_caption()


@router.post("/caption/openclip", response_model=ResOpenClipV2)
async def post_openclip_v2(req: ReqOpenClipV2):
    """Caption an image using OpenCLIP/BLIP."""
    def _run():
        from modules.api.caption import validate_image, do_openclip
        image = validate_image(req.image)
        return do_openclip(image, req)
    caption, medium, artist, movement, trending, flavor = await asyncio.to_thread(_run)
    return ResOpenClipV2(ok=True, caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)


@router.get("/caption/vlm/models", response_model=list[ItemVlmModelV2])
async def get_vlm_models_v2():
    """List available VLM models with capabilities."""
    from modules.api.caption import get_vqa_models
    return get_vqa_models()


@router.post("/caption/vlm", response_model=ResVqaV2)
async def post_vlm_v2(req: ReqVqaV2):
    """Caption an image using a Vision-Language Model."""
    def _run():
        from modules.api.caption import validate_image, do_vqa
        image = validate_image(req.image)
        return do_vqa(image, req)
    answer, annotated_b64 = await asyncio.to_thread(_run)
    return ResVqaV2(ok=True, answer=answer, annotated_image=annotated_b64)


@router.get("/caption/tagger/models", response_model=list[ItemTaggerModelV2])
async def get_tagger_models_v2():
    """List available tagger models."""
    from modules.api.caption import get_tagger_models
    return get_tagger_models()


@router.post("/caption/tagger", response_model=ResTaggerV2)
async def post_tagger_v2(req: ReqTaggerV2):
    """Tag an image using WaifuDiffusion or DeepBooru."""
    def _run():
        from modules.api.caption import validate_image, do_tagger
        image = validate_image(req.image)
        return do_tagger(image, req)
    tags, scores = await asyncio.to_thread(_run)
    return ResTaggerV2(ok=True, tags=tags, scores=scores)
