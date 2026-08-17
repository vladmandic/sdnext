from types import SimpleNamespace
from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
from modules import errors, shared, scripts_manager, ui
from modules.api import script, helpers
from modules.paths import resolve_output_path
from modules.video_models import models_def, video_load, video_run


errors.install()


class ReqVideo(BaseModel):
    engine: str | None = Field(default=None, title="Engine", description="Video engine family; omit together with model to use the currently loaded checkpoint")
    model: str | None = Field(default=None, title="Model", description="Video model name within the engine; see GET /sdapi/v1/video/models")
    prompt: str = Field(default="", title="Prompt", description="Text prompt")
    negative_prompt: str = Field(default="", title="Negative prompt", description="Negative text prompt")
    styles: list[str] = Field(default=[], title="Styles", description="Prompt style names to apply")
    width: int = Field(default=832, ge=64, le=4096, title="Width", description="Output width; snapped to the model canvas multiple")
    height: int = Field(default=480, ge=64, le=4096, title="Height", description="Output height; snapped to the model canvas multiple")
    frames: int = Field(default=17, ge=1, le=1024, title="Frames", description="Number of frames; 1 produces a single still image on workflow models")
    steps: int = Field(default=50, ge=1, le=200, title="Steps", description="Number of inference steps")
    sampler_name: str = Field(default="Default", title="Sampler", description="Sampler name; Default keeps the model scheduler")
    sampler_shift: float = Field(default=-1.0, title="Sampler shift", description="Scheduler flow shift; -1 keeps the model default")
    dynamic_shift: bool = Field(default=False, title="Dynamic shift", description="Enable dynamic scheduler shifting")
    seed: int = Field(default=-1, title="Seed", description="Generation seed; -1 for random")
    guidance_scale: float = Field(default=-1.0, title="Guidance scale", description="CFG scale; -1 keeps the model default")
    guidance_true: float = Field(default=-1.0, title="True guidance", description="True CFG scale; -1 keeps the model default")
    init_image: str | None = Field(default=None, title="Init image", description="Base64 or data URI for the first-frame image; an upload reference resolves only where an extension provides the upload store")
    init_strength: float = Field(default=0.8, ge=0.0, le=1.0, title="Init strength", description="Denoising strength for the init image")
    last_image: str | None = Field(default=None, title="Last image", description="Base64 or data URI for the last-frame image; an upload reference resolves only where an extension provides the upload store")
    references: list[str] = Field(default=[], title="References", description="Reference images for a reference workflow, in the order the model reads them; base64 or data URIs, or upload references where an extension provides the upload store. Images only: the video core also conditions on video and audio references, which this endpoint cannot carry. At most 9, each within a 1:4 to 4:1 aspect ratio. Rejected on models that do not condition on references")
    vae_type: str = Field(default="Default", title="VAE type", description="Decode variant: Default, Tiny, Remote, or Upscale")
    vae_tile_frames: int = Field(default=16, ge=1, le=64, title="VAE tile frames", description="Frames per VAE decode tile")
    audio: bool = Field(default=True, title="Audio", description="Generate audio on models that support it")
    mp4_fps: int = Field(default=24, ge=1, le=60, title="FPS", description="Frames per second of the saved video")
    mp4_interpolate: int = Field(default=0, ge=0, le=10, title="Interpolation", description="RIFE interpolation passes between frames")
    mp4_codec: str = Field(default="libx264", title="Codec", description="Video codec; none skips video encoding")
    mp4_ext: str = Field(default="mp4", title="Container", description="Container extension; the muxer is inferred from it")
    mp4_opt: str = Field(default="crf=16", title="Codec options", description="Encoder options as key=value pairs separated by : or ,")
    mp4_video: bool = Field(default=True, title="Save video", description="Write the video container to disk")
    mp4_frames: bool = Field(default=False, title="Save frames", description="Write individual frame images to disk")
    mp4_sf: bool = Field(default=False, title="Save safetensors", description="Write raw frames as a safetensors file")
    mp4_thumb: bool = Field(default=True, title="Save thumbnail", description="Write a thumbnail image next to the video")
    override_settings: dict = Field(default={}, title="Override settings", description="Setting overrides applied for this generation only")
    script_args: list = Field(default=[], title="Script args", description="Positional arguments for a selectable script")
    alwayson_scripts: dict = Field(default={}, title="Always-on scripts", description="Per-script argument overrides, keyed by script name")
    send_video: bool = Field(default=True, title="Send video", description="Return the video base64-encoded in the response")
    send_frames: bool = Field(default=False, title="Send frames", description="Return every frame base64-encoded in the response")
    send_thumbnail: bool = Field(default=True, title="Send thumbnail", description="Return the thumbnail base64-encoded in the response")
    extra: dict | None = Field(default={}, exclude=True, title="Extra", description="Extra attributes set on the processing object")


class ResVideo(BaseModel):
    video: str | None = Field(default=None, title="Video", description="Base64-encoded video file; empty when not requested, above the size cap, or in still mode")
    video_path: str | None = Field(default=None, title="Video path", description="Server path of the saved video; fetch via GET /sdapi/v1/video/file")
    thumbnail: str | None = Field(default=None, title="Thumbnail", description="Base64-encoded thumbnail image")
    thumbnail_path: str | None = Field(default=None, title="Thumbnail path", description="Server path of the saved thumbnail")
    frames: list[str] = Field(default=[], title="Frames", description="Base64-encoded frames; always populated in still mode")
    frames_count: int = Field(default=0, title="Frame count", description="Number of frames written, after interpolation")
    fps: float = Field(default=0.0, title="FPS", description="Effective frames per second of the saved video")
    duration: float = Field(default=0.0, title="Duration", description="Video duration in seconds")
    has_audio: bool = Field(default=False, title="Has audio", description="Whether the video carries an audio track")
    still: bool = Field(default=False, title="Still", description="Single-frame result; the product is in frames and no video was written")
    params: dict = Field(default={}, title="Parameters", description="Echo of the request parameters used for generation")
    info: str = Field(default="", title="Info", description="Generation info string with seed, sampler, and pipeline details")


class ItemVideoModel(BaseModel):
    engine: str = Field(title="Engine", description="Video engine family")
    name: str = Field(title="Name", description="Model name; pass together with engine to select it")
    repo: str = Field(default="", title="Repo", description="Model repository or path")
    url: str = Field(default="", title="URL", description="Model information page")
    mode: str = Field(title="Mode", description="Input mode: workflow, t2v, i2v, flf2v, vace, animate, condition, or unknown; condition models accept conditioning the generic path does not wire and run as text to video here")
    workflow: str | None = Field(default=None, title="Workflow", description="Modular workflow name when the model dispatches on inputs; ref2va conditions on references and ignores the keyframe images")
    base: bool = Field(default=False, title="Base", description="Also listed in the base checkpoint dropdown")
    loaded: bool = Field(default=False, title="Loaded", description="Currently loaded through the video registry")


def model_mode(m: models_def.Model) -> str:
    return models_def.dispatch_mode(m)


class APIVideo:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg_video = []

    def prepare_scripts(self, p_stub, req: ReqVideo):
        script_runner = scripts_manager.scripts_video
        if not script_runner.scripts:
            script_runner.initialize_scripts(is_img2img=False, is_control=False, is_video=True)
            ui.create_ui(None)
        if not self.default_script_arg_video:
            self.default_script_arg_video = script.init_default_script_args(script_runner)
        script_args = script.init_script_args(p_stub, req, self.default_script_arg_video, None, None, script_runner)
        return script_runner, script_args

    def sanitize_b64(self, req: ReqVideo):
        def sanitize_str(args: list):
            for idx in range(0, len(args)):
                if isinstance(args[idx], str) and len(args[idx]) >= 1000:
                    args[idx] = f"<str {len(args[idx])}>"
        for name in ('init_image', 'last_image'):
            val = getattr(req, name, None)
            if isinstance(val, str) and len(val) >= 1000:
                setattr(req, name, f"<str {len(val)}>")
        if req.references:
            sanitize_str(req.references)
        if req.script_args:
            sanitize_str(req.script_args)
        if req.alwayson_scripts:
            for script_obj in req.alwayson_scripts.values():
                if script_obj and "args" in script_obj and script_obj["args"]:
                    sanitize_str(script_obj["args"])

    def post_video(self, req: ReqVideo):
        """Generate a video, or a single still frame, using a video model.

        Omit `engine` and `model` to drive the currently loaded checkpoint when it is
        video-capable; this covers models loaded from local folders that have no registry
        entry. Pass both names to select a registry model, which is loaded on demand;
        `GET /sdapi/v1/video/models` enumerates the valid pairs.

        `frames` of 1 on a workflow model produces a single still image returned in `frames`.
        Disk outputs are controlled by `mp4_video`, `mp4_frames`, `mp4_sf`, and `mp4_thumb`;
        response payloads are controlled independently by `send_video`, `send_frames`, and
        `send_thumbnail`. Artifacts above the base64 size cap return `video` empty with
        `video_path` set; fetch those via `GET /sdapi/v1/video/file`.

        `init_image` and `last_image` accept base64 data or data URIs. An `upload:` reference
        resolves only where an extension registers an upload store; without one it is rejected.
        Models whose workflow is `ref2va` condition on `references` instead: an ordered list of
        images the prompt addresses as `<Picture 1>`, `<Picture 2>` and so on, following list
        order. A single reference may also be passed as `init_image`. Reference images do not
        set the output canvas, and `last_image` is ignored. The workflow also conditions on video
        and audio references, addressed as `<Video i>` and `<Audio i>`, but they decode from files
        rather than from the wire, so this endpoint carries images alone.

        Progress is reported on `GET /sdapi/v1/progress`; `POST /sdapi/v1/interrupt` cancels.
        Switching checkpoints via `override_settings` is not supported here; use
        `POST /sdapi/v1/checkpoint` before generating.
        """
        try:
            selected, needs_load = video_run.resolve_model(req.engine, req.model)
        except video_run.VideoError as e:
            raise HTTPException(status_code=e.code, detail=str(e)) from e
        sampler_name = helpers.validate_sampler_name(req.sampler_name)
        init_image = helpers.decode_base64_to_image(req.init_image) if req.init_image else None
        last_image = helpers.decode_base64_to_image(req.last_image) if req.last_image else None
        references = [helpers.decode_base64_to_image(x) for x in (req.references or [])]
        overrides = dict(req.override_settings or {})
        for key in ('sd_model_checkpoint', 'sd_model_refiner'):
            if key in overrides:
                raise HTTPException(status_code=400, detail=f"{key} override is not supported here: switch models via POST /sdapi/v1/checkpoint before generating")
        p_stub = SimpleNamespace(per_script_args={})
        script_runner, script_args = self.prepare_scripts(p_stub, req)
        extra = getattr(req, 'extra', {}) or {}

        with self.queue_lock:
            jobid = shared.state.begin('API-VID', api=True)
            try:
                res = video_run.run(
                    selected,
                    prompt=req.prompt,
                    negative=req.negative_prompt,
                    styles=req.styles,
                    width=req.width,
                    height=req.height,
                    frames=req.frames,
                    steps=req.steps,
                    sampler_name=sampler_name,
                    sampler_shift=req.sampler_shift,
                    dynamic_shift=req.dynamic_shift,
                    seed=req.seed,
                    guidance_scale=req.guidance_scale,
                    guidance_true=req.guidance_true,
                    init_image=init_image,
                    init_strength=req.init_strength,
                    last_image=last_image,
                    references=references,
                    vae_type=req.vae_type,
                    vae_tile_frames=req.vae_tile_frames,
                    audio=req.audio,
                    mp4_fps=req.mp4_fps,
                    mp4_interpolate=req.mp4_interpolate,
                    mp4_codec=req.mp4_codec,
                    mp4_ext=req.mp4_ext,
                    mp4_opt=req.mp4_opt,
                    mp4_video=req.mp4_video,
                    mp4_frames=req.mp4_frames,
                    mp4_sf=req.mp4_sf,
                    mp4_thumb=req.mp4_thumb,
                    override_settings=overrides,
                    engine=req.engine,
                    scripts=script_runner,
                    script_args=script_args,
                    per_script_args=p_stub.per_script_args,
                    extra_p=extra,
                    needs_load=needs_load,
                )
            except video_run.VideoError as e:
                raise HTTPException(status_code=e.code, detail=str(e)) from e
            finally:
                shared.state.end(jobid, api=False)

        send_frames = req.send_frames or res.still # a still request has no other product to return
        b64_frames = list(map(helpers.encode_pil_to_base64, res.images)) if send_frames else []
        video_b64 = helpers.encode_file_to_base64(res.video_path) if req.send_video and res.video_path else None
        thumb_b64 = helpers.encode_file_to_base64(res.thumb_path) if req.send_thumbnail and res.thumb_path else None
        duration = round(res.num_frames / res.fps, 3) if res.fps > 0 else 0.0
        self.sanitize_b64(req)
        params = {k: v for k, v in vars(req).items() if k != 'extra'}
        return ResVideo(
            video=video_b64,
            video_path=res.video_path,
            thumbnail=thumb_b64,
            thumbnail_path=res.thumb_path,
            frames=b64_frames,
            frames_count=res.num_frames,
            fps=res.fps,
            duration=duration,
            has_audio=res.has_audio,
            still=res.still,
            params=params,
            info=res.processed.info,
        )

    def get_video_models(self, engine: str | None = None):
        """List video engines and models; optionally filter by engine."""
        items = []
        for family, rows in models_def.models.items():
            if engine is not None and family.lower() != engine.lower():
                continue
            for m in rows:
                if not models_def.is_model(m):
                    continue
                items.append(ItemVideoModel(
                    engine=family,
                    name=m.name,
                    repo=m.repo or '',
                    url=m.url or '',
                    mode=model_mode(m),
                    workflow=m.workflow,
                    base=m.base,
                    loaded=(m.name == video_load.loaded_model),
                ))
        return items

    def get_video_file(self, file: str):
        """Serve a video artifact produced by this endpoint; the path must resolve inside the video output directory."""
        import mimetypes
        from pathlib import Path
        from starlette.responses import FileResponse
        if not file or not file.strip():
            raise HTTPException(status_code=400, detail="file path is required")
        root = Path(resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)).resolve()
        target = Path(file).resolve()
        if root not in target.parents:
            raise HTTPException(status_code=403, detail=f"file {file}: must be inside the video output directory")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {file}")
        media_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        return FileResponse(str(target), media_type=media_type, filename=target.name)
