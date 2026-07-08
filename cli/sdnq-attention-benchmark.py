#!/usr/bin/env python
"""
Benchmark and validate SDNQ attention on the local GPU.

Runs the kernel from modules/sdnq/kernels/triton_atten.py directly and compares speed and
numerical error against torch scaled_dot_product_attention and sageattention when installed.
Verifies mask, causal, GQA and padding code paths, probes float8 support, and prints
recommended values for the Compute Settings -> SDNQ Attention section.

Shape presets follow real model geometries: sd15, sdxl, anima, flux2 (Klein), wan22 (A14B),
ltx2 (LTX 2.3), plus a masked joint-attention preset. Run from the sdnext root with the venv
active:
    python cli/sdnq-attention-benchmark.py
    python cli/sdnq-attention-benchmark.py --shapes all
    python cli/sdnq-attention-benchmark.py --shapes wan22,ltx2 --iters 20

The first run of each shape includes triton autotune time; tuning results are cached on disk
and reused by the webui for matching shapes.
"""

import os
import sys
import time
import signal
import logging
import argparse
import importlib.metadata
from contextlib import contextmanager

import torch
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


console = Console(width=None if sys.stdout.isatty() else 140) # keep tables readable when piped to a file
ROUNDED_BOX = box.ROUNDED # modules.logger monkeypatches rich's box.ROUNDED to box.SIMPLE at import; keep the real one for panels
shared = None
sdnq_triton_atten = None
transcript = [] # final tables and panels, written out by --save


def emit(renderable):
    console.print(renderable)
    transcript.append(renderable)


def save_transcript(path):
    with open(path, "w", encoding="utf-8") as fh:
        file_console = Console(file=fh, width=140) # file consoles are non-terminal: markup renders to plain text
        for renderable in transcript:
            file_console.print(renderable)
    console.print(f"results saved to {path}")

shape_presets = {
    # name: (batch, heads, tokens, head_dim, description); geometry from the model transformer configs
    "sd15": (2, 8, 4096, 40, "SD 1.5 unet self-attention at 512px, batched cfg, head dim padded 40 to 64"),
    "sdxl": (2, 10, 4096, 64, "SDXL unet self-attention at 1024px, batched cfg"),
    "anima": (1, 16, 4096, 128, "Anima 1.0 self-attention at 1024px, one cfg pass"),
    "flux2": (1, 32, 4608, 128, "FLUX.2 Klein 9B joint attention at 1024px, 4096 image plus 512 text tokens"),
    "wan22": (1, 40, 32760, 128, "Wan 2.2 A14B self-attention, 832x480 81 frames, one cfg pass"),
    "ltx2": (1, 32, 13376, 128, "LTX 2.3 self-attention, 1216x704 121 frames, one cfg pass"),
    "masked": (1, 32, 4608, 128, "FLUX.2 Klein shape with boolean key-padding mask, 25% of keys masked"),
}
full_run = ["sd15", "sdxl", "anima", "flux2", "wan22", "ltx2"]
default_shapes = "sdxl,flux2"

# benchmark configs: id, label, kwargs for sdnq_triton_atten (None = external baseline)
bench_configs = [
    ("base", "torch sdpa (bf16)", None),
    ("sage", "sageattention", None),
    ("noquant", "sdnq, quantized matmul off", dict(do_quantize=False)),
    ("int8", "sdnq int8 qk (auto, default)", dict(matmul_dtype="auto", pv_matmul_dtype="auto")),
    ("smooth", "sdnq int8 qk + smooth k", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True)),
    ("hadamard", "sdnq int8 qk + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", use_hadamard=True)),
    ("smooth_hadamard", "sdnq int8 qk + smooth + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True, use_hadamard=True)),
    ("fp16pv", "sdnq int8 qk + fp16 pv", dict(matmul_dtype="auto", pv_matmul_dtype="float16")),
    ("int8pv", "sdnq int8 qk + int8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="int8")),
    ("fp16qk", "sdnq fp16 qk", dict(matmul_dtype="float16", pv_matmul_dtype="auto")),
]
video_config_ids = ["base", "sage", "noquant", "int8", "smooth", "hadamard", "fp16pv"]
masked_config_ids = ["base", "noquant", "int8"]
# hadamard configs excluded: compiling hadamard with a non pow2 head dim currently hangs torch inductor
sd15_config_ids = ["base", "noquant", "int8", "smooth", "fp16pv", "int8pv", "fp16qk"]

atten_settings = [
    ("sdnq_attention_use_quantized_matmul", "Use Quantized MatMul"),
    ("sdnq_attention_matmul_type", "MatMul type"),
    ("sdnq_attention_pv_matmul_type", "PV MatMul type"),
    ("sdnq_attention_smooth_k", "Use Smooth K"),
    ("sdnq_attention_use_hadamard", "Use Hadamard"),
    ("sdnq_attention_hadamard_group_size", "Hadamard Group Size"),
]


def parse_cli():
    parser = argparse.ArgumentParser(description="benchmark and validate sdnq attention on the local gpu")
    parser.add_argument("--shapes", type=str, default=default_shapes, help=f"comma-separated shape presets: {', '.join(shape_presets)}; 'all' runs {', '.join(full_run)} (default: %(default)s)")
    parser.add_argument("--iters", type=int, default=12, help="minimum timed iterations per config, scaled up for fast kernels (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=4, help="minimum warmup iterations per config, scaled up for fast kernels (default: %(default)s)")
    parser.add_argument("--skip-checks", action="store_true", help="skip kernel correctness checks")
    parser.add_argument("--skip-bench", action="store_true", help="skip benchmarks, run checks and fp8 probe only")
    parser.add_argument("--config-timeout", type=int, default=300, help="best effort: abort a config whose compile plus first call exceeds this many seconds, 0 disables; cannot interrupt native-level hangs (default: %(default)s)")
    parser.add_argument("--save", type=str, default=None, help="write a plain-text copy of all tables and notes to this file; keeps colors and live progress on the terminal, unlike piping through tee")
    args = parser.parse_args()
    sys.argv = sys.argv[:1] # sdnext parses argv again on import and rejects unknown arguments
    return args


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def load_sdnext():
    global shared, sdnq_triton_atten # pylint: disable=global-statement
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if not torch.cuda.is_available():
        console.print("[red]no cuda or rocm device available: sdnq attention requires a gpu with triton[/red]")
        return False
    stock_sdpa = torch.nn.functional.scaled_dot_product_attention
    try:
        from modules import shared as shared_module
        from modules.sdnq.kernels.triton_atten import sdnq_triton_atten as atten
    except Exception as e:
        console.print(f"[red]failed to import the sdnq attention kernel: {e}[/red]")
        console.print("run from the sdnext root with the venv active; triton is required")
        return False
    # importing modules.shared installs the configured sdp override hijacks in this process;
    # restore stock sdpa so baselines and references measure torch itself
    torch.nn.functional.scaled_dot_product_attention = stock_sdpa
    shared = shared_module
    sdnq_triton_atten = atten
    # inductor dumps failing buffers at CRITICAL to raw stderr, tearing the live
    # display; failures still surface as exceptions in the tables
    logging.getLogger("torch._inductor").setLevel(logging.CRITICAL + 1)
    return True


def sage_attention():
    # mirror the backend selection from modules/attention.py: sm86 needs the cuda backend
    try:
        if torch.cuda.get_device_capability(torch.device("cuda")) == (8, 6):
            from sageattention import sageattn_qk_int8_pv_fp16_cuda
            def sage_fn(q, k, v, scale):
                return sageattn_qk_int8_pv_fp16_cuda(q=q, k=k, v=v, tensor_layout="HND", is_causal=False, sm_scale=scale, return_lse=False, pv_accum_dtype="fp32")
        else:
            from sageattention import sageattn
            def sage_fn(q, k, v, scale):
                return sageattn(q=q, k=k, v=v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)
        return sage_fn
    except Exception:
        return None


def make_qkv(batch, heads, tokens, head_dim, structured=True, kv_heads=None):
    # structured keys carry a shared per-channel bias plus a few outlier channels, mimicking the
    # key statistics that motivate smoothing and rotation; absolute error varies by model
    # architecture while the relative ordering of configs holds
    generator = torch.Generator(device="cuda").manual_seed(1234)
    kv_heads = kv_heads or heads
    q = torch.randn(batch, heads, tokens, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    if structured:
        k = k + torch.randn(1, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator) * 3.0
        k[..., [3, head_dim // 2, head_dim - 5]] *= 4.0
    return q, k, v


def rel_err(out, ref):
    out = out.to(torch.float32)
    return ((out - ref).norm() / ref.norm()).item()


def error_summary(e, limit=60):
    # dynamo appends advice lines ("Set TORCHDYNAMO_VERBOSE=1 ...") after the real error;
    # keep the last substantive line so table rows show the actual failure
    lines = [line.strip() for line in str(e).splitlines()]
    lines = [line for line in lines if line and "TORCHDYNAMO_VERBOSE" not in line and "TORCH_LOGS" not in line]
    text = lines[-1] if lines else type(e).__name__
    return text[:limit]


compile_error_markers = ("InductorError", "BackendCompilerFailed", "CompilationError", "TritonError")


def failure_text(e):
    # label known compile failures plainly, technical detail dimmed
    detail = error_summary(e)
    if isinstance(e, TimeoutError):
        return f"[red]failed: compile timed out[/red] [dim]({detail})[/dim]"
    if type(e).__name__ in compile_error_markers or any(marker in str(e) for marker in compile_error_markers):
        return f"[red]failed: torch compile error[/red] [dim]({detail})[/dim]"
    return f"[red]{type(e).__name__}[/red]: {detail}"


def speedup_cell(base_ms, ms):
    ratio = base_ms / ms
    text = f"x{ratio:4.2f}"
    if ratio >= 1.10:
        return f"[green]{text}[/green]"
    if ratio < 0.95:
        return f"[red]{text}[/red]"
    return text


def live_progress():
    # status line shown under a live-filling table
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console)
    task = progress.add_task("starting", total=None)
    return progress, task


@contextmanager
def time_limit(seconds, label):
    # torch.compile can spin indefinitely in sympy/inductor on pathological graphs
    # (e.g. hadamard over a padded non pow2 head dim); turn a wedged compile into a red row
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    def handler(signum, frame): # pylint: disable=unused-argument
        raise TimeoutError(f"{label}: compile plus first call exceeded {seconds}s")
    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def bench(fn, warmup, iters, on_phase=None):
    if on_phase:
        on_phase("warmup")
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    estimate = max(time.perf_counter() - started, 1e-6)
    # scale counts to a time budget so short kernels get enough activity to ramp gpu clocks
    warmup = min(max(warmup, int(0.2 / estimate)), 500)
    iters = min(max(iters, int(0.5 / estimate)), 500)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if on_phase:
        on_phase(f"timing {iters} iterations")
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in events)
    return times[len(times) // 2]


def free_vram_gb():
    free, _total = torch.cuda.mem_get_info()
    return free / 1024**3


def print_environment(fp8_result):
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    lines = [
        f"device: [cyan]{torch.cuda.get_device_name(device)}[/cyan] capability={capability[0]}.{capability[1]}",
        f"torch: {torch.__version__}  triton: {package_version('triton')}  sageattention: {package_version('sageattention') or 'not installed'}  flash-attn: {package_version('flash-attn') or 'not installed'}",
        f"float8_e4m3fn matmul: {'[green]supported[/green]' if fp8_result['qk'][0] else '[red]not supported, selecting it fails generation[/red]'}",
        f"sdnq attention enabled in current config: {'[green]yes[/green]' if 'SDNQ attention' in shared.opts.sdp_overrides else '[yellow]no, enable via Compute Settings -> SDP overrides (requires restart)[/yellow]'}",
    ]
    overrides = [f"{key}={value}" for key, value in os.environ.items() if key.startswith("SDNQ_TRITON_ATTEN")]
    if overrides:
        lines.append(f"env overrides: {' '.join(overrides)}")
    emit(Panel("\n".join(lines), title="environment", box=ROUNDED_BOX))


def print_banner(selected, args):
    lines = [
        "measures sdnq attention speed and accuracy on this gpu and recommends values for [cyan]Compute Settings -> SDNQ Attention[/cyan]",
    ]
    if args.skip_bench:
        lines.append("running correctness checks and the float8 probe only (--skip-bench)")
    else:
        lines.append(f"shapes: [cyan]{', '.join(selected)}[/cyan]  (available: {', '.join(shape_presets)}; pass --shapes to match the models you use)")
        lines.append("first run compiles triton kernels per shape and can take several minutes; repeat runs are much faster")
    if args.save:
        lines.append(f"a plain-text copy of the results will be saved to [cyan]{args.save}[/cyan]")
    console.print(Panel("\n".join(lines), title="sdnq attention benchmark", box=ROUNDED_BOX))


def probe_fp8():
    # small real-kernel calls: pre-Ada nvidia and pre-RDNA4/CDNA3 amd fail at triton compile time
    q, k, v = make_qkv(1, 2, 256, 64, structured=False)
    result = {}
    with console.status("probing float8 support") as status:
        for name, kwargs in [("qk", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="auto")), ("pv", dict(matmul_dtype="auto", pv_matmul_dtype="float8_e4m3fn"))]:
            status.update(f"probing float8 support: {name} matmul (a slow compile failure here is normal on older gpus)")
            try:
                out = sdnq_triton_atten(q, k, v, **kwargs)
                torch.cuda.synchronize()
                result[name] = (not bool(torch.isnan(out).any().item()), "compiles and runs")
            except Exception as e:
                result[name] = (False, f"{type(e).__name__}: {error_summary(e, 120)}")
    if not (result["qk"][0] and result["pv"][0]):
        torch._dynamo.reset() # pylint: disable=protected-access # drop the failed compile state before the real runs
    return result


def run_correctness():
    checks = []
    q, k, v = make_qkv(2, 8, 512, 64, structured=False)
    checks.append(("plain", (q, k, v), {}, {}))
    checks.append(("causal", (q, k, v), dict(is_causal=True), {}))
    qg, kg, vg = make_qkv(2, 8, 512, 64, structured=False, kv_heads=2)
    checks.append(("gqa 8:2 heads", (qg, kg, vg), dict(enable_gqa=True), {}))
    bool_mask = torch.zeros(2, 1, 1, 512, device="cuda", dtype=torch.bool)
    bool_mask[..., :384] = True
    checks.append(("bool key-padding mask", (q, k, v), dict(attn_mask=bool_mask), {}))
    float_mask = torch.zeros(2, 1, 1, 512, device="cuda", dtype=torch.bfloat16)
    float_mask[..., 384:] = float("-inf")
    checks.append(("float additive mask", (q, k, v), dict(attn_mask=float_mask), {}))
    checks.append(("smooth k + hadamard", (q, k, v), {}, dict(smooth_k=True, use_hadamard=True)))
    qp, kp, vp = make_qkv(2, 8, 512, 40, structured=False)
    checks.append(("head dim 40 (padded)", (qp, kp, vp), {}, {}))
    checks.append(("head dim 40 + hadamard", (qp, kp, vp), {}, dict(use_hadamard=True)))
    qn, kn, vn = make_qkv(2, 8, 1000, 64, structured=False)
    checks.append(("non-pow2 sequence 1000", (qn, kn, vn), {}, {}))

    table = Table(title="kernel correctness (small shapes, vs fp32 sdpa reference)", box=box.SIMPLE_HEAVY)
    table.add_column("code path")
    table.add_column("kernel error", justify="right")
    table.add_column("int8 error", justify="right")
    table.add_column("result", justify="center")
    failed = []
    dynamo_disable = getattr(torch._dynamo.config, "disable", False) # pylint: disable=protected-access
    torch._dynamo.config.disable = True # pylint: disable=protected-access # checks target kernel behavior, run the input prep eager
    progress, task = live_progress()
    with Live(Group(table, progress), console=console, refresh_per_second=4) as live:
        for index, (name, (cq, ck, cv), kwargs, sdnq_kwargs) in enumerate(checks, start=1):
            progress.reset(task, description=f"check {index}/{len(checks)}  {name}")
            ref_kwargs = dict(kwargs)
            if ref_kwargs.get("attn_mask", None) is not None and torch.is_floating_point(ref_kwargs["attn_mask"]):
                ref_kwargs["attn_mask"] = ref_kwargs["attn_mask"].to(torch.float32) # stock sdpa requires the additive mask dtype to match query
            ref = torch.nn.functional.scaled_dot_product_attention(cq.to(torch.float32), ck.to(torch.float32), cv.to(torch.float32), **ref_kwargs)
            try:
                plain = sdnq_triton_atten(cq, ck, cv, do_quantize=False, **kwargs)
                quant = sdnq_triton_atten(cq, ck, cv, matmul_dtype="auto", pv_matmul_dtype="auto", **kwargs, **sdnq_kwargs)
                err_plain = rel_err(plain, ref)
                err_quant = rel_err(quant, ref)
                has_nan = bool(torch.isnan(quant).any().item())
                ok = err_plain < 0.01 and err_quant < 0.2 and not has_nan
                if not ok:
                    failed.append(name)
                verdict = "[green]pass[/green]" if ok else "[red]fail[/red]"
                if has_nan:
                    verdict = "[red]nan[/red]"
                table.add_row(name, f"{err_plain:.5f}", f"{err_quant:.5f}", verdict)
            except Exception as e:
                failed.append(name)
                table.add_row(name, "-", "-", failure_text(e))
        live.update(table)
    if not console.is_terminal:
        console.line() # Live's final frame lacks a trailing newline when output is piped
    transcript.append(table)
    torch._dynamo.config.disable = dynamo_disable # pylint: disable=protected-access
    if failed:
        emit(f"[red]failed checks: {', '.join(failed)}[/red]")
    if "head dim 40 + hadamard" in failed and shared.opts.sdnq_attention_use_hadamard:
        emit("[yellow]current config has Use Hadamard enabled: models with non power-of-2 head dims (SD 1.5) fail at generation with it[/yellow]")
    return failed


def bench_shape(preset, iters, warmup, position=None, config_timeout=300):
    batch, heads, tokens, head_dim, description = shape_presets[preset]
    config_ids = {"wan22": video_config_ids, "ltx2": video_config_ids, "masked": masked_config_ids, "sd15": sd15_config_ids}.get(preset)
    if preset == "sd15":
        emit("[yellow]sd15: hadamard configs skipped, compiling hadamard with a non pow2 head dim currently hangs torch inductor[/yellow]")
    attn_mask = None
    if preset == "masked":
        attn_mask = torch.zeros(batch, 1, 1, tokens, device="cuda", dtype=torch.bool)
        attn_mask[..., :int(tokens * 0.75)] = True
    sage = sage_attention()
    selected_configs = []
    for config_id, label, kwargs in bench_configs:
        if config_ids is not None and config_id not in config_ids:
            continue
        if config_id == "sage" and (sage is None or attn_mask is not None or head_dim not in {64, 96, 128}):
            continue
        selected_configs.append((config_id, label, kwargs))

    def make_table():
        shape_table = Table(title=f"{preset}: batch={batch} heads={heads} tokens={tokens} head_dim={head_dim}", caption=description, caption_style="dim", box=box.SIMPLE_HEAVY)
        shape_table.add_column("config")
        shape_table.add_column("median time", justify="right")
        shape_table.add_column("speedup", justify="right")
        shape_table.add_column("error", justify="right")
        return shape_table

    table = make_table()
    results = {}
    rows = []
    base_ms = None
    prefix = f"shape {position[0]}/{position[1]}  " if position else ""
    progress, task = live_progress()
    with Live(Group(table, progress), console=console, refresh_per_second=4) as live:
        progress.update(task, description=f"{prefix}{preset}: preparing inputs and fp32 reference")
        q, k, v = make_qkv(batch, heads, tokens, head_dim)
        scale = head_dim ** -0.5
        ref = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=attn_mask)
        for index, (config_id, label, kwargs) in enumerate(selected_configs, start=1):
            def phase(step, current_label=label, current_index=index):
                progress.update(task, description=f"{prefix}config {current_index}/{len(selected_configs)}  {current_label}: {step}")
            if config_id == "base":
                def fn(mask=attn_mask):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            elif config_id == "sage":
                def fn(sm=scale):
                    return sage(q, k, v, sm)
            else:
                def fn(kw=kwargs, mask=attn_mask):
                    return sdnq_triton_atten(q, k, v, attn_mask=mask, **kw)
            try:
                torch._dynamo.reset() # pylint: disable=protected-access # one compiled input-prep specialization per config, matching how the webui runs a fixed config
                progress.reset(task) # restart the elapsed clock so it times the current config
                phase("compiling")
                with time_limit(config_timeout, label):
                    err = rel_err(fn(), ref)
                ms = bench(fn, warmup, iters, on_phase=phase)
                if base_ms is None:
                    base_ms = ms
                results[config_id] = (ms, err)
                row = (label, f"{ms:8.3f} ms", speedup_cell(base_ms, ms), f"{err:.5f}")
                rows.append((config_id, ms, row))
                table.add_row(*row)
            except Exception as e:
                results[config_id] = (None, None)
                row = (label, "-", "-", failure_text(e))
                rows.append((config_id, None, row))
                table.add_row(*row)
        # rebuild the table to star the fastest sdnq config, when one actually beats the baseline
        candidates = [(config_id, ms) for config_id, ms, _ in rows if ms is not None and config_id not in ("base", "sage")]
        if base_ms and candidates:
            best_id, best_ms = min(candidates, key=lambda item: item[1])
            if best_ms < base_ms:
                table = make_table()
                for config_id, _ms, row in rows:
                    if config_id == best_id:
                        table.add_row(f"★ {row[0]}", *row[1:], style="bold")
                    else:
                        table.add_row(*row)
        live.update(table)
    if not console.is_terminal:
        console.line() # Live's final frame lacks a trailing newline when output is piped
    transcript.append(table)
    return results


def measured(results, config_id):
    ms, err = results.get(config_id, (None, None))
    return (ms, err) if ms is not None else (None, None)


def build_recommendations(all_results, fp8_result):
    # prefer an image dit shape with the full config set as reference, then the video shapes
    reference = None
    for preset in ["flux2", "anima", "sdxl", "wan22", "ltx2", "sd15"]:
        if preset in all_results and all_results[preset].get("int8", (None, None))[0] is not None:
            reference = preset
            break
    if reference is None:
        emit("[yellow]no successful int8 benchmark, recommendations unavailable[/yellow]")
        return
    results = all_results[reference]
    base_ms, _base_err = measured(results, "base")
    int8_ms, int8_err = measured(results, "int8")
    int8_speedup = base_ms / int8_ms if base_ms and int8_ms else None

    table = Table(title=f"recommended settings (Compute Settings -> SDNQ Attention), measured at the {reference} shape", box=ROUNDED_BOX)
    table.add_column("setting")
    table.add_column("current", justify="center")
    table.add_column("recommended", justify="center")
    table.add_column("reason")

    def current(key):
        return str(getattr(shared.opts, key))

    rows = []
    if int8_speedup and int8_speedup >= 1.10:
        rows.append(("Use Quantized MatMul", current("sdnq_attention_use_quantized_matmul"), "True", f"int8 qk measured x{int8_speedup:.2f} vs torch sdpa"))
    else:
        rows.append(("Use Quantized MatMul", current("sdnq_attention_use_quantized_matmul"), "False", f"int8 qk gain is marginal on this gpu (x{int8_speedup:.2f})" if int8_speedup else "int8 qk failed to run"))

    qk_reason = "resolves to int8; uint8 remaps to int8"
    if fp8_result["qk"][0]:
        qk_reason += "; float8 compiles here but per-token int8 keeps finer granularity"
    rows.append(("MatMul type", current("sdnq_attention_matmul_type"), "auto", qk_reason))

    int8pv_ms, _int8pv_err = measured(results, "int8pv")
    if fp8_result["pv"][0]:
        rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "float8_e4m3fn", "hardware float8 available for the pv stage"))
    elif int8pv_ms and int8_ms and int8pv_ms < int8_ms * 0.95:
        rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "int8", f"int8 pv measured x{int8_ms / int8pv_ms:.2f} over int8 qk alone; slightly higher error"))
    else:
        pv_note = "auto keeps pv unquantized; int8 pv measured no gain here" if int8pv_ms else "auto keeps pv unquantized"
        rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "auto", pv_note))

    smooth_ms, smooth_err = measured(results, "smooth")
    if smooth_ms and int8_ms and int8_err and smooth_err:
        cost = smooth_ms / int8_ms - 1.0
        gain = int8_err / smooth_err if smooth_err > 0 else 1.0
        recommend = gain >= 1.3 and cost <= 0.25 # attention-level cost is a few percent end to end
        rows.append(("Use Smooth K", current("sdnq_attention_smooth_k"), str(recommend), f"int8 error x{gain:.1f} lower for {cost:+.0%} time"))

    hadamard_ms, hadamard_err = measured(results, "hadamard")
    if hadamard_ms and int8_ms and int8_err and hadamard_err:
        cost = hadamard_ms / int8_ms - 1.0
        gain = int8_err / hadamard_err if hadamard_err > 0 else 1.0
        if gain >= 1.3 and cost <= 0.15:
            rows.append(("Use Hadamard", current("sdnq_attention_use_hadamard"), "True", f"int8 error x{gain:.1f} lower for {cost:+.0%} time; hangs torch compile on non pow2 head dims (SD 1.5)"))
        else:
            rows.append(("Use Hadamard", current("sdnq_attention_use_hadamard"), "False", f"error x{gain:.1f} lower but {cost:+.0%} time; consider for long-sequence sessions; hangs torch compile on non pow2 head dims (SD 1.5)"))

    rows.append(("Hadamard Group Size", current("sdnq_attention_hadamard_group_size"), "256", "values above head dim are clamped; non pow2 values floor to the nearest power of 2"))

    differing = 0
    for setting, current_value, recommended, reason in rows:
        if current_value == recommended:
            table.add_row(setting, f"[green]{current_value}[/green]", recommended, reason)
        else:
            differing += 1
            table.add_row(setting, f"[yellow]{current_value}[/yellow]", recommended, reason)
    emit(table)
    if differing == 1:
        emit("[yellow]1 setting differs from the recommended value, change it in Compute Settings -> SDNQ Attention[/yellow]")
    elif differing:
        emit(f"[yellow]{differing} settings differ from the recommended values, change them in Compute Settings -> SDNQ Attention[/yellow]")
    else:
        emit("[green]current settings already match the recommendations[/green]")

    notes = []
    if not fp8_result["qk"][0]:
        notes.append("[red]float8_e4m3fn is unsupported on this gpu: selecting it in either dropdown fails generation with a compile error[/red]")
    sage_ms, _sage_err = measured(results, "sage")
    if sage_ms and int8_ms:
        notes.append(f"sageattention comparison at {reference}: sdnq int8 {int8_ms:.2f} ms vs sage {sage_ms:.2f} ms; sdnq additionally covers masks, gqa and causal attention")
    notes.append("[yellow]non pow2 head dims (sd 1.5): quantized matmul configs currently fail torch compile and hadamard hangs it; disable quantized matmul for sd 1.5 sessions or set SDNQ_COMPILE_KWARGS='{\"dynamic\": false}'[/yellow]")
    notes.append("the six settings above apply on the next generation; toggling the 'SDNQ attention' SDP override requires a restart")
    notes.append("errors are measured on synthetic tensors with outlier-heavy keys; real models, especially qk-normed dits, sit lower")
    emit(Panel("\n".join(notes), title="notes", box=ROUNDED_BOX))


def main():
    args = parse_cli()
    if not load_sdnext():
        sys.exit(1)
    selected = list(full_run) if args.shapes.strip().lower() == "all" else [s.strip() for s in args.shapes.split(",") if s.strip()]
    unknown = [s for s in selected if s not in shape_presets]
    if unknown:
        console.print(f"[red]unknown shape preset(s): {', '.join(unknown)}; available: {', '.join(shape_presets)}[/red]")
        sys.exit(1)
    print_banner(selected, args)
    fp8_result = probe_fp8()
    print_environment(fp8_result)
    if not args.skip_checks:
        run_correctness()
    if args.skip_bench:
        if args.save:
            save_transcript(args.save)
        return
    all_results = {}
    minimum_vram = {"wan22": 6.0, "ltx2": 3.0, "masked": 6.0}
    for index, preset in enumerate(selected, start=1):
        needed = minimum_vram.get(preset, 2.0)
        if free_vram_gb() < needed:
            emit(f"[yellow]skipping {preset}: needs about {needed:.0f} gb free vram, {free_vram_gb():.1f} gb available[/yellow]")
            continue
        all_results[preset] = bench_shape(preset, args.iters, args.warmup, position=(index, len(selected)), config_timeout=args.config_timeout)
    build_recommendations(all_results, fp8_result)
    if args.save:
        save_transcript(args.save)


if __name__ == "__main__":
    main()
