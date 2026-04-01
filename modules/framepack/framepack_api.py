from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
from modules import shared
from modules.logger import log


class ReqFramepack(BaseModel):
    variant: str = Field(default=None, title="Model variant", description="Model variant to use")
    prompt: str = Field(default=None, title="Prompt", description="Prompt for the model")
    init_image: str = Field(default=None, title="Initial image", description="Base64 encoded initial image")
    end_image: str | None = Field(default=None, title="End image", description="Base64 encoded end image")
    start_weight: float | None = Field(default=1.0, title="Start weight", description="Weight of the initial image")
    end_weight: float | None = Field(default=1.0, title="End weight", description="Weight of the end image")
    vision_weight: float | None = Field(default=1.0, title="Vision weight", description="Weight of the vision model")
    system_prompt: str | None = Field(default=None, title="System prompt", description="System prompt for the model")
    optimized_prompt: bool | None = Field(default=True, title="Optimized system prompt", description="Use optimized system prompt for the model")
    section_prompt: str | None = Field(default=None, title="Section prompt", description="Prompt for each section")
    negative_prompt: str | None = Field(default=None, title="Negative prompt", description="Negative prompt for the model")
    styles: list[str] | None = Field(default=None, title="Styles", description="Styles for the model")
    seed: int | None = Field(default=None, title="Seed", description="Seed for the model")
    resolution: int | None = Field(default=640, title="Resolution", description="Resolution of the image")
    duration: float | None = Field(default=4, title="Duration", description="Duration of the video in seconds")
    latent_ws: int | None = Field(default=9, title="Latent window size", description="Size of the latent window")
    steps: int | None = Field(default=25, title="Video steps", description="Number of steps for the video generation")
    cfg_scale: float | None = Field(default=1.0, title="CFG scale", description="CFG scale for the model")
    cfg_distilled: float | None = Field(default=10.0, title="Distilled CFG scale", description="Distilled CFG scale for the model")
    cfg_rescale: float | None = Field(default=0.0, title="CFG re-scale", description="CFG re-scale for the model")
    shift: float | None = Field(default=0, title="Sampler shift", description="Shift for the sampler")
    use_teacache: bool | None = Field(default=True, title="Enable TeaCache", description="Use TeaCache for the model")
    use_cfgzero: bool | None = Field(default=False, title="Enable CFGZero", description="Use CFGZero for the model")
    mp4_fps: int | None = Field(default=30, title="FPS", description="Frames per second for the video")
    mp4_codec: str | None = Field(default="libx264", title="Codec", description="Codec for the video")
    mp4_sf: bool | None = Field(default=False, title="Save SafeTensors", description="Save SafeTensors for the video")
    mp4_video: bool | None = Field(default=True, title="Save Video", description="Save video")
    mp4_frames: bool | None = Field(default=False, title="Save Frames", description="Save frames for the video")
    mp4_opt: str | None = Field(default="crf:16", title="Options", description="Options for the video codec")
    mp4_ext: str | None = Field(default="mp4", title="Format", description="Format for the video")
    mp4_interpolate: int | None = Field(default=0, title="Interpolation", description="Interpolation for the video")
    attention: str | None = Field(default="Default", title="Attention", description="Attention type for the model")
    vae_type: str | None = Field(default="Local", title="VAE", description="VAE type for the model")
    vlm_enhance: bool | None = Field(default=False, title="VLM enhance", description="Enable VLM enhance")
    vlm_model: str | None = Field(default=None, title="VLM model", description="VLM model to use")
    vlm_system_prompt: str | None = Field(default=None, title="VLM system prompt", description="System prompt for the VLM model")


class ResFramepack(BaseModel):
    id: str = Field(title="TaskID", description="Task ID")
    filename: str = Field(title="TaskID", description="Task ID")
    message: str = Field(title="TaskID", description="Task ID")


def framepack_post(request: ReqFramepack):
    import numpy as np
    from modules.api import helpers
    from framepack_wrappers import run_framepack
    task_id = shared.state.get_id()

    try:
        if request.init_image is not None:
            init_image = np.array(helpers.decode_base64_to_image(request.init_image)) if request.init_image else None
        else:
            init_image = None
    except Exception as e:
        log.error(f"API FramePack: id={task_id} cannot decode init image: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if request.end_image is not None:
            end_image = np.array(helpers.decode_base64_to_image(request.end_image)) if request.end_image else None
        else:
            end_image = None
    except Exception as e:
        log.error(f"API FramePack: id={task_id} cannot decode end image: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    del request.init_image
    del request.end_image
    log.trace(f"API FramePack: id={task_id} init={init_image.shape} end={end_image.shape if end_image else None} {request}")

    generator = run_framepack(
        _ui_state=None,
        task_id=f'task({task_id})',
        variant=request.variant,
        init_image=init_image,
        end_image=end_image,
        start_weight=request.start_weight,
        end_weight=request.end_weight,
        vision_weight=request.vision_weight,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        optimized_prompt=request.optimized_prompt,
        section_prompt=request.section_prompt,
        negative_prompt=request.negative_prompt,
        styles=request.styles,
        seed=request.seed,
        resolution=request.resolution,
        duration=request.duration,
        latent_ws=request.latent_ws,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        cfg_distilled=request.cfg_distilled,
        cfg_rescale=request.cfg_rescale,
        shift=request.shift,
        use_teacache=request.use_teacache,
        use_cfgzero=request.use_cfgzero,
        use_preview=False,
        mp4_fps=request.mp4_fps,
        mp4_codec=request.mp4_codec,
        mp4_sf=request.mp4_sf,
        mp4_video=request.mp4_video,
        mp4_frames=request.mp4_frames,
        mp4_opt=request.mp4_opt,
        mp4_ext=request.mp4_ext,
        mp4_interpolate=request.mp4_interpolate,
        attention=request.attention,
        vae_type=request.vae_type,
        vlm_enhance=request.vlm_enhance,
        vlm_model=request.vlm_model,
        vlm_system_prompt=request.vlm_system_prompt,
    )
    response = ResFramepack(id=task_id, filename='', message='')
    for message in generator:
        if isinstance(message, tuple) and len(message) == 3:
            if isinstance(message[0], str):
                response.filename = message[0]
            if isinstance(message[2], str):
                response.message = message[2]
    return response


def create_api(_fastapi, _gradioapp):
    shared.api.add_api_route("/sdapi/v1/framepack", framepack_post, methods=["POST"], response_model=ResFramepack)
