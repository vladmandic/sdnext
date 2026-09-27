import argparse
import os
import pathlib
import time
import av
import numpy as np
from rich import print as rp
from rich import progress

from . import (
    DLSSNRPipeline,
    DLSSNRTemporalOptions,
    DLSSNRTemporalSession,
    NR_PROFILES,
    DLSSFGPipeline,
    DLSSFGOptions,
    DLSSFGSession,
    FG_PROFILES,
    DLSSVSRPipeline,
    DLSSVSRTemporalOptions,
    DLSSVSRTemporalSession,
    VSR_PROFILES,
)

_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_NR_MODEL = str(_DIR / "DLSSNeuralRender.safetensors")
DEFAULT_FG_MODEL = str(_DIR / "DLSSFrameGen.safetensors")
DEFAULT_VSR_MODEL = str(_DIR / "DLSSSuperRes.safetensors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified DLSS Pipeline: NeuralRender -> FrameGen -> SuperRes"
    )

    # General / IO
    io_group = parser.add_argument_group("I/O & Hardware Options")
    io_group.add_argument("--input", type=str, required=True, help="Input video or image file path")
    io_group.add_argument("--output", type=str, required=False, help="Output video file path")
    io_group.add_argument("--ops", type=str, default="nfs", help="Execution operations sequence: n=NeuralRender, f=FrameGen, s=SuperRes (e.g. nfs, fns, fs, s)")
    io_group.add_argument("--device", type=str, default="cuda", help="Computation device (cuda, cpu)")
    io_group.add_argument("--dtype", type=str, default=None, help="Inference precision dtype (e.g. fast, float16, float32)")
    io_group.add_argument("--graph", action="store_true", help="Enable CUDA Graphs for static extent caching")
    io_group.add_argument("--max-frames", type=int, default=0, help="Max input frames to process (0 = all)")
    io_group.add_argument("--crf", type=int, default=17, help="H.264 video encoding quality (0-51, lower = higher quality, default: 17)")
    io_group.add_argument("--preset", type=str, default="slow", help="H.264 encoder preset (e.g. slow, medium, fast)")

    # DLSS-NeuralRender Options
    nr_group = parser.add_argument_group("DLSS-NeuralRender Options")
    nr_group.add_argument("--nr-model", type=str, default=DEFAULT_NR_MODEL, help="Path to DLSSNeuralRender.safetensors")
    nr_group.add_argument("--nr-profile", type=str, default="standard", choices=list(NR_PROFILES), help="NR profile preset")
    nr_group.add_argument("--nr-scale", type=float, default=1.0, help="NR spatial scale factor")
    nr_group.add_argument("--nr-intensity", type=float, default=1.0, help="NR enhancement intensity")
    nr_group.add_argument("--nr-blend", type=float, default=0.73974609375, help="NR history blend scale")
    nr_group.add_argument("--nr-detail", type=float, default=1.0, help="NR detail strength")
    nr_group.add_argument("--nr-colour", type=float, default=1.0, help="NR colour strength")
    nr_group.add_argument("--nr-radius", type=float, default=4.0, help="NR detail radius")
    nr_group.add_argument("--nr-motion", type=str, default="flow", choices=["flow", "zero"], help="NR motion vector source")
    nr_group.add_argument("--nr-threshold", type=float, default=0.3, help="NR scene cut threshold")
    nr_group.add_argument("--nr-normalized", type=float, default=None, help="NR normalised style override")
    nr_group.add_argument("--nr-local-tone", type=float, default=None, help="NR local tone strength override")
    nr_group.add_argument("--nr-local-structure", type=float, default=None, help="NR local structure strength override")
    nr_group.add_argument("--nr-skin-structure", type=float, default=None, help="NR skin structure strength override")
    nr_group.add_argument("--nr-mask-structure", type=float, default=None, help="NR mask structure strength override")

    # DLSS-FrameGen Options
    fg_group = parser.add_argument_group("DLSS-FrameGen Options")
    fg_group.add_argument("--fg-model", type=str, default=DEFAULT_FG_MODEL, help="Path to DLSSFrameGen.safetensors")
    fg_group.add_argument("--fg-profile", type=str, default="2x", choices=list(FG_PROFILES), help="FG profile preset")
    fg_group.add_argument("--fg-factor", type=int, default=2, help="FG frame interpolation multiplier (e.g. 2, 4)")
    fg_group.add_argument("--fg-mode", type=str, default="fps", choices=["fps", "slowmo"], help="FG mode: fps (multiplier) or slowmo (stretch)")
    fg_group.add_argument("--fg-threshold", type=float, default=0.4, help="FG scene cut detection threshold")

    # DLSS-SuperRes Options
    vsr_group = parser.add_argument_group("DLSS-SuperRes (VSR) Options")
    vsr_group.add_argument("--vsr-model", type=str, default=DEFAULT_VSR_MODEL, help="Path to DLSSSuperRes.safetensors")
    vsr_group.add_argument("--vsr-profile", type=str, default="ultra", choices=list(VSR_PROFILES), help="VSR quality preset")
    vsr_group.add_argument("--vsr-scale", type=float, default=2.0, help="VSR upscale target factor (e.g. 2.0, 4.0)")
    vsr_group.add_argument("--vsr-detail", type=float, default=1.0, help="VSR detail strength")
    vsr_group.add_argument("--vsr-colour", type=float, default=1.0, help="VSR colour strength")
    vsr_group.add_argument("--vsr-radius", type=float, default=4.0, help="VSR detail radius")
    vsr_group.add_argument("--vsr-threshold", type=float, default=0.4, help="VSR scene cut detection threshold")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input or not os.path.exists(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    ops = args.ops.lower().strip()
    valid_chars = {"n", "f", "s"}
    if not ops or any(c not in valid_chars for c in ops):
        raise ValueError(f"Invalid ops '{args.ops}'. Use any combination of 'n', 'f', 's' (e.g. 'nfs', 'fns', 'fs', 's')")

    # 1. Initialize Stage: NeuralRender
    nr_session = None
    if "n" in ops and os.path.exists(args.nr_model):
        nr_options = DLSSNRTemporalOptions(
            profile=args.nr_profile,
            scale=args.nr_scale,
            intensity=args.nr_intensity,
            blend_scale=args.nr_blend,
            detail_strength=args.nr_detail,
            colour_strength=args.nr_colour,
            detail_radius=args.nr_radius,
            normalized_style=args.nr_normalized,
            local_tone_strength=args.nr_local_tone,
            local_structure_strength=args.nr_local_structure,
            skin_structure_strength=args.nr_skin_structure,
            mask_structure_strength=args.nr_mask_structure,
            scene_cut_threshold=args.nr_threshold,
        )
        nr_pipe = DLSSNRPipeline.from_safetensors(
            args.nr_model, device=args.device, dtype=args.dtype, graphs=args.graph
        )
        nr_session = DLSSNRTemporalSession(nr_pipe, options=nr_options, motion=args.nr_motion)
        rp("[bold green]DLSS-NeuralRender (n)[/bold green]:", nr_pipe)
    else:
        rp("[yellow]DLSS-NeuralRender (n): BYPASSED[/yellow]")

    # 2. Initialize Stage: FrameGen
    fg_session = None
    if "f" in ops and os.path.exists(args.fg_model):
        fg_options = DLSSFGOptions(
            profile=args.fg_profile,
            factor=args.fg_factor,
            mode=args.fg_mode,
            scene_cut_threshold=args.fg_threshold,
        )
        fg_pipe = DLSSFGPipeline.from_safetensors(
            args.fg_model, device=args.device, dtype=args.dtype, graphs=args.graph
        )
        fg_session = DLSSFGSession(fg_pipe, options=fg_options)
        rp("[bold green]DLSS-FrameGen (f)[/bold green]:", fg_pipe)
    else:
        rp("[yellow]DLSS-FrameGen (f): BYPASSED[/yellow]")

    # 3. Initialize Stage: SuperRes
    vsr_session = None
    if "s" in ops and os.path.exists(args.vsr_model):
        vsr_options = DLSSVSRTemporalOptions(
            profile=args.vsr_profile,
            scale=args.vsr_scale,
            detail_strength=args.vsr_detail,
            colour_strength=args.vsr_colour,
            detail_radius=args.vsr_radius,
            scene_cut_threshold=args.vsr_threshold,
        )
        vsr_pipe = DLSSVSRPipeline.from_safetensors(
            args.vsr_model, device=args.device, dtype=args.dtype, graphs=args.graph
        )
        vsr_session = DLSSVSRTemporalSession(vsr_pipe, options=vsr_options)
        rp("[bold green]DLSS-SuperRes (s)[/bold green]:", vsr_pipe)
    else:
        rp("[yellow]DLSS-SuperRes (s): BYPASSED[/yellow]")

    in_container = av.open(args.input)
    in_stream = in_container.streams.video[0]
    frames = in_stream.frames
    in_fps = in_stream.guessed_rate or 30.0

    fg_mult = fg_session.options.factor if (fg_session and fg_session.options.mode == "fps") else 1
    out_fps = in_fps * fg_mult

    stage_names = {"n": "NeuralRender", "f": "FrameGen", "s": "SuperRes"}
    active_chain = " -> ".join([stage_names[c] for c in ops if (c == "n" and nr_session) or (c == "f" and fg_session) or (c == "s" and vsr_session)]) or "Passthrough"
    rp(f'Pipeline Operations: [bold cyan]{ops}[/bold cyan] ({active_chain})')
    rp(f'Input: file="{args.input}" stream={in_stream} in_fps={in_fps} out_fps={out_fps} frames={frames}')

    ou_container = None
    ou_stream = None
    if args.output:
        ou_container = av.open(args.output, "w")
        ou_stream = ou_container.add_stream("libx264", rate=out_fps)
        ou_stream.pix_fmt = "yuv420p"
        ou_stream.options = {"crf": str(args.crf), "preset": args.preset}
        if in_stream.width and in_stream.height:
            v_scale = vsr_session.options.scale if vsr_session else 1.0
            ou_stream.width = int(round(in_stream.width * v_scale))
            ou_stream.height = int(round(in_stream.height * v_scale))

    i = 0
    total_produced_frames = 0
    total_frames = (
        min(frames, args.max_frames)
        if (args.max_frames > 0 and frames)
        else (args.max_frames if args.max_frames > 0 else frames)
    )

    pbar = progress.Progress(
        progress.TextColumn("DLSS Chain"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.MofNCompleteColumn(),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        progress.TextColumn("{task.description}"),
    )
    task = pbar.add_task(description=f"{args.input}", total=total_frames if total_frames else None)
    t_start = time.perf_counter()

    with pbar:
        for frame in in_container.decode(video=0):
            i += 1
            if args.max_frames > 0 and i > args.max_frames:
                break
            in_nd = frame.to_ndarray(format="rgb24").astype("float32") / 255.0
            t_frame = time.perf_counter()

            current_frames = [in_nd]
            for stage_char in ops:
                next_frames = []
                for f in current_frames:
                    if stage_char == "n" and nr_session is not None:
                        next_frames.append(nr_session(f))
                    elif stage_char == "f" and fg_session is not None:
                        next_frames.extend(fg_session(f))
                    elif stage_char == "s" and vsr_session is not None:
                        next_frames.append(vsr_session(f))
                    else:
                        next_frames.append(f)
                current_frames = next_frames
            final_frames = current_frames

            total_produced_frames += len(final_frames)
            pbar.update(
                task,
                advance=1,
                description=f"in={in_nd.shape} out={final_frames[0].shape} produced={len(final_frames)} time={(time.perf_counter() - t_frame):.3f}",
            )

            if ou_stream is not None:
                if ou_stream.width is None:
                    ou_stream.height, ou_stream.width, _ = final_frames[0].shape
                for of in final_frames:
                    ou_bytes = (255.0 * np.clip(of, 0.0, 1.0)).astype("uint8")
                    output_frame = av.VideoFrame.from_ndarray(ou_bytes, format="rgb24")
                    for packet in ou_stream.encode(output_frame):
                        ou_container.mux(packet)

    if ou_stream is not None:
        for packet in ou_stream.encode():
            ou_container.mux(packet)
        ou_container.close()

    in_container.close()
    t_total = time.perf_counter() - t_start
    rp(f'Complete: output="{args.output}" in_frames={i} out_frames={total_produced_frames} time={t_total:.3f} fps={(total_produced_frames / t_total):.2f}')


if __name__ == "__main__":
    main()
