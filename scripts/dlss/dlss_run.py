import os
import time
import numpy as np
import torch
from PIL import Image
from rich import progress
from modules import devices, shared
from modules.logger import log
from modules.processing import StableDiffusionProcessing, Processed
from modules.scripts_postprocessing import PostprocessedImage
from modules.dlss import (
    DLSSNRPipeline,
    DLSSNRTemporalOptions,
    DLSSNRTemporalSession,
    DLSSVSRPipeline,
    DLSSVSRTemporalOptions,
    DLSSVSRTemporalSession,
    DLSSFGPipeline,
    DLSSFGSession,
    DLSSFGOptions,
)


NR_MODEL = 'DLSSNeuralRender.safetensors'
SS_MODEL = 'DLSSSuperRes.safetensors'
FG_MODEL = 'DLSSFrameGen.safetensors'
debug = os.environ.get('SD_DLSS_DEBUG', None) is not None


def _to_ndarray_rgb_float(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
    elif hasattr(img, 'convert'):
        arr = np.array(img.convert('RGB'))
    else:
        arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float32) if arr.max() <= 1.0 else (arr / 255.0).astype(np.float32)
    return arr.astype(np.float32) / 255.0


def dlss_run(p: StableDiffusionProcessing | None,
             pp: Processed | PostprocessedImage,
             *args,
             **kwargs,
            ):
    if debug:
        log.trace(f'DLSS p={p}')
        log.trace(f'DLSS pp={pp}')
        log.trace(f'DLSS args={args}')
        log.trace(f'DLSS kwargs={kwargs}')
    if len(args) == 0 and len(kwargs) == 0:
        log.warning('DLSS run called with no additional arguments')
        return None
    elif len(args) > 0: # called with positional arguments from generate / pp=Processed
        dlss_enabled, dlss_graph, dlss_chunk, dlss_full, nr_profile, nr_motion, nr_scale, nr_intensity, nr_blend, nr_detail, nr_colour, nr_radius, nr_threshold, nr_normalized, nr_local_tone, nr_local_structure, nr_skin_structure, nr_mask_structure, ss_profile, ss_scale, ss_detail, ss_colour, ss_radius, ss_threshold, fg_profile, fg_mode, fg_factor, fg_threshold = args
    elif len(kwargs) > 0: # called with keyword arguments from postprocess / pp=PostprocessedImage
        dlss_enabled = kwargs.get("dlss_enabled", [])
        dlss_graph = kwargs.get("dlss_graph", False)
        dlss_chunk = kwargs.get("dlss_chunk", 131072)
        dlss_full = kwargs.get("dlss_full", False)
        nr_profile = kwargs.get("nr_profile", "None")
        nr_motion = kwargs.get("nr_motion", "flow")
        nr_scale = kwargs.get("nr_scale", 1.0)
        nr_intensity = kwargs.get("nr_intensity", 1.0)
        nr_blend = kwargs.get("nr_blend", 0.73974609375)
        nr_detail = kwargs.get("nr_detail", 1.0)
        nr_colour = kwargs.get("nr_colour", 1.0)
        nr_radius = kwargs.get("nr_radius", 4.0)
        nr_threshold = kwargs.get("nr_threshold", 0.3)
        nr_normalized = kwargs.get("nr_normalized", 0.0)
        nr_local_tone = kwargs.get("nr_local_tone", 1.0)
        nr_local_structure = kwargs.get("nr_local_structure", 1.0)
        nr_skin_structure = kwargs.get("nr_skin_structure", 0.0)
        nr_mask_structure = kwargs.get("nr_mask_structure", 0.0)
        ss_profile = kwargs.get("ss_profile", "None")
        ss_scale = kwargs.get("ss_scale", 2.0)
        ss_detail = kwargs.get("ss_detail", 1.0)
        ss_colour = kwargs.get("ss_colour", 1.0)
        ss_radius = kwargs.get("ss_radius", 4.0)
        ss_threshold = kwargs.get("ss_threshold", 0.4)
        fg_profile = kwargs.get("fg_profile", "None")
        fg_mode = kwargs.get("fg_mode", "fps")
        fg_factor = kwargs.get("fg_factor", 2)
        fg_threshold = kwargs.get("fg_threshold", 0.4)
    else:
        log.warning('DLSS run called with unexpected argument structure')
        return None
    if p is not None: # override load args from processing object
        if hasattr(p, 'nr_profile'):
            nr_profile = getattr(p, 'nr_profile', nr_profile)
            dlss_enabled.append('NeuralRender')
        if hasattr(p, 'ss_profile'):
            ss_profile = getattr(p, 'ss_profile', ss_profile)
            dlss_enabled.append('SuperRes')
        if hasattr(p, 'fg_profile'):
            fg_profile = getattr(p, 'fg_profile', fg_profile)
            dlss_enabled.append('FrameGen')
        dlss_enabled = list(set(dlss_enabled))
    if debug:
        log.trace(f'DLSS enabled={dlss_enabled} graph={dlss_graph} chunk={dlss_chunk} full={dlss_full} nr_profile={nr_profile} ss_profile={ss_profile} fg_profile={fg_profile} nr_motion={nr_motion} nr_scale={nr_scale} nr_intensity={nr_intensity} nr_blend={nr_blend} nr_detail={nr_detail} nr_colour={nr_colour} nr_radius={nr_radius} nr_threshold={nr_threshold} nr_normalized={nr_normalized} nr_local_tone={nr_local_tone} nr_local_structure={nr_local_structure} nr_skin_structure={nr_skin_structure} nr_mask_structure={nr_mask_structure} ss_scale={ss_scale} ss_detail={ss_detail} ss_colour={ss_colour} ss_radius={ss_radius} ss_threshold={ss_threshold} fg_mode={fg_mode} fg_factor={fg_factor} fg_threshold={fg_threshold}')

    if isinstance(dlss_enabled, str):
        enabled_ops = [op.strip() for op in dlss_enabled.split(',') if op.strip()]
    elif isinstance(dlss_enabled, (list, tuple, set)):
        enabled_ops = list(dlss_enabled)
    else:
        enabled_ops = []
    if not enabled_ops:
        return None

    has_images = hasattr(pp, "images") and pp.images is not None
    has_image = hasattr(pp, "image") and pp.image is not None
    has_video = hasattr(pp, "video") and pp.video is not None
    if has_images:
        raw_inputs = pp.images if isinstance(pp.images, list) else [pp.images]
    elif has_image:
        raw_inputs = [pp.image]
    elif has_video:
        from modules import video
        frames, fps, duration, w, h, codec, _frame = video.get_video_params(pp.video)
        log.info(f'DLSS video: file="{pp.video}" frames={frames} fps={fps} duration={duration} w={w} h={h} codec={codec}')
        raw_inputs = video.read_video(pp.video)
    else:
        raw_inputs = []

    if not raw_inputs:
        log.warning('DLSS: No input frames')
        return None

    if debug:
        log.trace(f'DLSS ops={enabled_ops} frames={len(raw_inputs)}')

    from scripts.dlss.dlss_model import _resolve_model_path, _get_nr_pipeline, _get_fg_pipeline, _get_vsr_pipeline # delayed import
    dtype = torch.float32 if dlss_full else torch.float16

    active_pipelines = []
    try:
        t0 = time.perf_counter()
        active_chain = []
        for op in enabled_ops:
            op_lower = op.lower() if isinstance(op, str) else ''
            if ('neural' in op_lower) or (op_lower == 'n'):
                nr_model_path = _resolve_model_path(NR_MODEL)
                if nr_model_path:
                    nr_pipe: DLSSNRPipeline = _get_nr_pipeline(nr_model_path, dlss_graph, dlss_chunk, dtype)
                    active_pipelines.append(nr_pipe)
                    nr_opt_profile = nr_profile if nr_profile and nr_profile != 'None' else 'Standard'
                    nr_options = DLSSNRTemporalOptions(
                        profile=nr_opt_profile,
                        scale=nr_scale,
                        intensity=nr_intensity,
                        blend_scale=nr_blend,
                        detail_strength=nr_detail,
                        colour_strength=nr_colour,
                        detail_radius=nr_radius,
                        normalized_style=nr_normalized,
                        local_tone_strength=nr_local_tone,
                        local_structure_strength=nr_local_structure,
                        skin_structure_strength=nr_skin_structure,
                        mask_structure_strength=nr_mask_structure,
                        scene_cut_threshold=nr_threshold,
                    )
                    nr_session = DLSSNRTemporalSession(nr_pipe, options=nr_options, motion=nr_motion)
                    log.debug(f'DLSS init: {nr_session}')
                    if p is not None:
                        p.extra_generation_params['DLSSNR'] = f'{nr_opt_profile}'
                    active_chain.append(('nr', nr_session))
            elif ('frame' in op_lower) or (op_lower == 'fg') or (op_lower == 'f'):
                if len(raw_inputs) < 2:
                    log.warning('DLSS FrameGen: not enough input frames')
                    fg_model_path = None
                else:
                    fg_model_path = _resolve_model_path(FG_MODEL)
                if fg_model_path:
                    fg_pipe: DLSSFGPipeline = _get_fg_pipeline(fg_model_path, dlss_graph, dtype)
                    active_pipelines.append(fg_pipe)
                    fg_opt_profile = fg_profile if fg_profile and fg_profile != 'None' else '2x'
                    fg_options = DLSSFGOptions(
                        profile=fg_opt_profile,
                        factor=int(fg_factor),
                        mode=fg_mode,
                        scene_cut_threshold=fg_threshold,
                    )
                    fg_session = DLSSFGSession(fg_pipe, options=fg_options)
                    log.debug(f'DLSS init: {fg_session}')
                    if p is not None:
                        p.extra_generation_params['DLSSFG'] = f'{fg_opt_profile}'
                    active_chain.append(('fg', fg_session))
            elif ('super' in op_lower) or ('vsr' in op_lower) or (op_lower == 's'):
                ss_model_path = _resolve_model_path(SS_MODEL)
                if ss_model_path:
                    vsr_pipe: DLSSVSRPipeline = _get_vsr_pipeline(ss_model_path, dlss_graph, dtype)
                    active_pipelines.append(vsr_pipe)
                    ss_opt_profile = ss_profile if ss_profile and ss_profile != 'None' else 'Ultra'
                    vsr_options = DLSSVSRTemporalOptions(
                        profile=ss_opt_profile,
                        scale=float(ss_scale),
                        detail_strength=float(ss_detail),
                        colour_strength=float(ss_colour),
                        detail_radius=float(ss_radius),
                        scene_cut_threshold=float(ss_threshold),
                    )
                    vsr_session = DLSSVSRTemporalSession(vsr_pipe, options=vsr_options)
                    log.debug(f'DLSS init: {vsr_session}')
                    if p is not None:
                        p.extra_generation_params['DLSSSR'] = f'{ss_opt_profile}'
                    active_chain.append(('vsr', vsr_session))

        if not active_chain:
            return None

        t1 = time.perf_counter()
        is_pil = isinstance(raw_inputs[0], Image.Image)
        input_ndarrays = [_to_ndarray_rgb_float(img) for img in raw_inputs]

        pbar = progress.Progress(
            progress.TextColumn("DLSS"),
            progress.BarColumn(),
            progress.TaskProgressColumn(),
            progress.MofNCompleteColumn(),
            progress.TimeElapsedColumn(),
            progress.TimeRemainingColumn(),
            progress.TextColumn("{task.description}"),
        )
        pbar_desc = f'ops: {", ".join([name for name, _ in active_chain])}'
        par_enabled = len(input_ndarrays) > 1
        pbar_task = pbar.add_task(description=pbar_desc, total=len(input_ndarrays)) if par_enabled else None

        jobid = shared.state.begin('DLSS')
        with pbar:
            total_produced_frames = []
            for in_nd in input_ndarrays:
                current_frames = [in_nd]
                for stage_name, session in active_chain:
                    next_frames = []
                    for f in current_frames:
                        if stage_name == 'nr':
                            next_frames.append(session(f))
                        elif stage_name == 'fg':
                            next_frames.extend(session(f))
                        elif stage_name == 'vsr':
                            next_frames.append(session(f))
                        else:
                            next_frames.append(f)
                    current_frames = next_frames
                total_produced_frames.extend(current_frames)
                if par_enabled:
                    pbar.update(pbar_task, advance=1)

        output_images = []
        for frame_nd in total_produced_frames:
            frame_uint8 = (255.0 * np.clip(frame_nd, 0.0, 1.0)).astype(np.uint8)
            if is_pil:
                output_images.append(Image.fromarray(frame_uint8))
            else:
                output_images.append(frame_uint8)

        t2 = time.perf_counter()
        log.debug(f'DLSS output: frames={len(output_images)} init={t1 - t0:.4f} time={t2 - t1:.4f} its={len(output_images) / (t2 - t1):.4f}')
        shared.state.end(jobid)

        if has_images:
            pp.images = output_images
        elif has_image:
            pp.image = output_images[0] if output_images else None
        elif has_video:
            output_video = video.save_video(p, output_images, video_type='mp4', duration=duration, sync=True)
            pp.video = output_video

    finally:
        for pipe in active_pipelines:
            try:
                pipe.device = devices.cpu
                pipe.model = pipe.model.to(devices.cpu)
            except Exception as e:
                log.warning(f'DLSS: Error offloading model to CPU: {e}')
        if active_pipelines:
            devices.torch_gc()

    return pp
