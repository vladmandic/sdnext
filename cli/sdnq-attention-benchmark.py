#!/usr/bin/env python
"""
Benchmark and validate SDNQ attention on the local GPU.

Runs the kernel from modules/sdnq/kernels/triton_atten.py directly and compares speed and
numerical error against torch scaled_dot_product_attention and sageattention when installed.
Verifies mask, causal, GQA and padding code paths, probes float8 hardware support and the
torch.compile input prep, and prints recommended values for the Compute Settings -> SDNQ
Attention section.

Benchmarks run in the webui's configured dtype (--dtype overrides). The prep column is the
q/k/v quantization cost outside the kernel, included in the median. Recommendations come
from measured comparisons only.

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
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


console = Console(width=None if sys.stdout.isatty() else 140) # keep tables readable when piped to a file
ROUNDED_BOX = box.ROUNDED # modules.logger monkeypatches rich's box.ROUNDED to box.SIMPLE at import; keep the real one for panels
shared = None
devices = None
sdnq_triton_atten = None
bench_dtype = torch.bfloat16 # resolved to the webui's configured dtype in main
transcript = [] # final tables and panels, written out by --save


def dtype_label():
    return str(bench_dtype).replace("torch.", "")


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

# benchmark configs: id, label, kwargs for sdnq_triton_atten (None = external baseline);
# fp8 configs run only on gpus where the float8 probe passes
bench_configs = [
    ("base", "torch sdpa", None),
    ("sage", "sageattention", None),
    ("noquant", "sdnq, quantized matmul off", dict(do_quantize=False)),
    ("int8", "sdnq int8 qk", dict(matmul_dtype="auto", pv_matmul_dtype="auto")),
    ("smooth", "sdnq int8 qk + smooth k", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True)),
    ("hadamard", "sdnq int8 qk + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", use_hadamard=True)),
    ("smooth_hadamard", "sdnq int8 qk + smooth + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True, use_hadamard=True)),
    ("fp16pv", "sdnq int8 qk + fp16 pv", dict(matmul_dtype="auto", pv_matmul_dtype="float16")),
    ("int8pv", "sdnq int8 qk + int8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="int8")),
    ("fp8pv", "sdnq int8 qk + fp8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="float8_e4m3fn")),
    ("fp16qk", "sdnq fp16 qk", dict(matmul_dtype="float16", pv_matmul_dtype="auto")),
    ("fp8qk", "sdnq fp8 qk", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="auto")),
]
video_config_ids = ["base", "sage", "noquant", "int8", "smooth", "hadamard", "fp16pv", "fp8pv"]
masked_config_ids = ["base", "noquant", "int8"]
# hadamard configs excluded: compiling hadamard with a non pow2 head dim currently hangs torch inductor
sd15_config_ids = ["base", "noquant", "int8", "smooth", "fp16pv", "int8pv", "fp8pv", "fp16qk", "fp8qk"]

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
    parser.add_argument("--skip-bench", action="store_true", help="skip benchmarks, run checks and the fp8 and compile probes only")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16"], help="tensor dtype for benchmarks; auto uses the dtype the webui selected for this gpu (default: %(default)s)")
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


def triton_version():
    # distribution name varies by platform (triton-windows, pytorch-triton-rocm); report the module version
    try:
        import triton
        return triton.__version__
    except Exception:
        return None


def load_sdnext():
    global shared, devices, sdnq_triton_atten # pylint: disable=global-statement
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if not torch.cuda.is_available():
        console.print("[red]no cuda or rocm device available: sdnq attention requires a gpu with triton[/red]")
        return False
    stock_sdpa = torch.nn.functional.scaled_dot_product_attention
    try:
        from modules import shared as shared_module
        from modules import devices as devices_module
        from modules.sdnq.kernels.triton_atten import sdnq_triton_atten as atten
    except Exception as e:
        console.print(f"[red]failed to import the sdnq attention kernel: {e}[/red]")
        console.print("run from the sdnext root with the venv active; triton is required")
        return False
    # importing modules.shared installs the configured sdp override hijacks in this process;
    # restore stock sdpa so baselines and references measure torch itself
    torch.nn.functional.scaled_dot_product_attention = stock_sdpa
    shared = shared_module
    devices = devices_module
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
    q = torch.randn(batch, heads, tokens, head_dim, device="cuda", dtype=bench_dtype, generator=generator)
    k = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=bench_dtype, generator=generator)
    v = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=bench_dtype, generator=generator)
    if structured:
        k = k + torch.randn(1, kv_heads, 1, head_dim, device="cuda", dtype=bench_dtype, generator=generator) * 3.0
        k[..., [3, head_dim // 2, head_dim - 5]] *= 4.0
    return q, k, v


def fp32_reference(q, k, v, **kwargs):
    # sdnext enables tf32 globally; a math-backend dispatch fallback would degrade the reference to tf32 precision
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return torch.nn.functional.scaled_dot_product_attention(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), **kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn


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


def print_environment(fp8_result, prep_status, prep_detail):
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if fp8_result["qk"][0]:
        fp8_line = "float8_e4m3fn matmul: [green]supported[/green]"
    else:
        fp8_line = f"float8_e4m3fn matmul: [red]not supported on this gpu, selecting it fails generation[/red] [dim]({escape(fp8_result['qk'][1])})[/dim]"
    lines = [
        f"device: [cyan]{torch.cuda.get_device_name(device)}[/cyan] capability={capability[0]}.{capability[1]}",
        f"torch: {torch.__version__}  triton: {triton_version() or 'not installed'}  sageattention: {package_version('sageattention') or 'not installed'}  flash-attn: {package_version('flash-attn') or 'not installed'}",
        f"benchmark dtype: {dtype_label()}",
        fp8_line,
        f"sdnq attention enabled in current config: {'[green]yes[/green]' if 'SDNQ attention' in shared.opts.sdp_overrides else '[yellow]no, enable via Compute Settings -> SDP overrides (requires restart)[/yellow]'}",
    ]
    if prep_status == "disabled":
        lines.append("compiled input prep: torch.compile disabled in config, input prep runs eager")
    elif prep_status == "working":
        lines.append("compiled input prep: [green]working[/green]")
    else:
        lines.append(f"compiled input prep: [red]failing, every sdnq attention call errors at generation[/red] [dim]({escape(prep_detail)})[/dim]")
        if prep_status == "failing_dynamic":
            lines.append("  fix, verified on this machine: set SDNQ_COMPILE_KWARGS='{\"dynamic\": false}' (recompiles per shape); or install msvc build tools; or disable Compute Settings -> SDNQ -> Dequantize using torch.compile")
        else:
            lines.append("  fix: install a host c++ compiler (msvc build tools on windows), or disable Compute Settings -> SDNQ -> Dequantize using torch.compile")
    overrides = [f"{key}={value}" for key, value in os.environ.items() if key.startswith("SDNQ_TRITON_ATTEN")]
    if overrides:
        lines.append(f"env overrides: {' '.join(overrides)}")
    emit(Panel("\n".join(lines), title="environment", box=ROUNDED_BOX))


def print_banner(selected, args):
    lines = [
        "measures sdnq attention speed and accuracy on this gpu and recommends values for [cyan]Compute Settings -> SDNQ Attention[/cyan]",
    ]
    if args.skip_bench:
        lines.append("running correctness checks and the float8 and compile probes only (--skip-bench)")
    else:
        lines.append(f"shapes: [cyan]{', '.join(selected)}[/cyan]  (available: {', '.join(shape_presets)}; pass --shapes to match the models you use)")
        lines.append("first run compiles triton kernels per shape and can take several minutes; repeat runs are much faster")
    if args.save:
        lines.append(f"a plain-text copy of the results will be saved to [cyan]{args.save}[/cyan]")
    console.print(Panel("\n".join(lines), title="sdnq attention benchmark", box=ROUNDED_BOX))


def probe_fp8():
    # hardware probe, prep runs eager: fp8 kernel compile fails on pre-ada nvidia and
    # pre-rdna4/cdna3 amd; prep failures are dtype-independent and probed separately
    q, k, v = make_qkv(1, 2, 256, 64, structured=False)
    result = {}
    dynamo_disable = getattr(torch._dynamo.config, "disable", False) # pylint: disable=protected-access
    torch._dynamo.config.disable = True # pylint: disable=protected-access
    try:
        with console.status("probing float8 support") as status:
            for name, kwargs in [("qk", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="auto")), ("pv", dict(matmul_dtype="auto", pv_matmul_dtype="float8_e4m3fn"))]:
                status.update(f"probing float8 support: {name} matmul (a compile failure here is normal on older gpus)")
                try:
                    out = sdnq_triton_atten(q, k, v, **kwargs)
                    torch.cuda.synchronize()
                    result[name] = (not bool(torch.isnan(out).any().item()), "compiles and runs")
                except Exception as e:
                    result[name] = (False, f"{type(e).__name__}: {error_summary(e, 120)}")
    finally:
        torch._dynamo.config.disable = dynamo_disable # pylint: disable=protected-access
    return result


def probe_compiled_prep():
    # inductor lowers part of the dynamic-shape prep to a cpu helper kernel, so a missing
    # host c++ compiler (msvc on windows) fails every sdnq attention call at generation.
    # must run before any other sdnq_triton_atten call: a prior eager run masks the
    # cold-start failure the webui hits
    if not shared.opts.sdnq_dequantize_compile:
        return "disabled", None
    q, k, v = make_qkv(1, 2, 256, 64, structured=False)
    with console.status("probing compiled input prep (compiles on first run, cached afterwards)"):
        try:
            sdnq_triton_atten(q, k, v, matmul_dtype="auto", pv_matmul_dtype="auto")
            torch.cuda.synchronize()
            return "working", None
        except Exception as e:
            torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            detail = f"{type(e).__name__}: {error_summary(e, 120)}"
    # check whether the dynamic=false workaround holds
    from modules.sdnq.kernels import triton_atten as atten_module
    compiled_prep = atten_module.get_attn_inputs
    inner = getattr(compiled_prep, "_torchdynamo_orig_callable", None)
    if inner is None:
        return "failing", detail
    static_ok = False
    with console.status("compiled input prep failed, verifying the dynamic=false workaround"):
        atten_module.get_attn_inputs = torch.compile(inner, fullgraph=True, dynamic=False)
        try:
            sdnq_triton_atten(q, k, v, matmul_dtype="auto", pv_matmul_dtype="auto")
            torch.cuda.synchronize()
            static_ok = True
        except Exception:
            pass
        finally:
            atten_module.get_attn_inputs = compiled_prep
            torch._dynamo.reset() # pylint: disable=protected-access
    return ("failing_dynamic" if static_ok else "failing"), detail


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
    float_mask = torch.zeros(2, 1, 1, 512, device="cuda", dtype=bench_dtype)
    float_mask[..., 384:] = float("-inf")
    checks.append(("float additive mask", (q, k, v), dict(attn_mask=float_mask), {}))
    checks.append(("smooth k + hadamard", (q, k, v), {}, dict(smooth_k=True, use_hadamard=True)))
    qp, kp, vp = make_qkv(2, 8, 512, 40, structured=False)
    checks.append(("head dim 40 (padded)", (qp, kp, vp), {}, {}))
    checks.append(("head dim 40 + hadamard", (qp, kp, vp), {}, dict(use_hadamard=True)))
    qn, kn, vn = make_qkv(2, 8, 1000, 64, structured=False)
    checks.append(("non-pow2 sequence 1000", (qn, kn, vn), {}, {}))

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("code path")
    table.add_column("kernel error", justify="right")
    table.add_column("int8 error", justify="right")
    table.add_column("result", justify="center")
    panel = Panel(table, title="kernel correctness", subtitle="[dim]small shapes, vs fp32 sdpa reference[/dim]", box=ROUNDED_BOX, expand=False)
    failed = []
    dynamo_disable = getattr(torch._dynamo.config, "disable", False) # pylint: disable=protected-access
    torch._dynamo.config.disable = True # pylint: disable=protected-access # checks target kernel behavior, run the input prep eager
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        for index, (name, (cq, ck, cv), kwargs, sdnq_kwargs) in enumerate(checks, start=1):
            progress.reset(task, description=f"check {index}/{len(checks)}  {name}")
            ref_kwargs = dict(kwargs)
            if ref_kwargs.get("attn_mask", None) is not None and torch.is_floating_point(ref_kwargs["attn_mask"]):
                ref_kwargs["attn_mask"] = ref_kwargs["attn_mask"].to(torch.float32) # stock sdpa requires the additive mask dtype to match query
            ref = fp32_reference(cq, ck, cv, **ref_kwargs)
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
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch._dynamo.config.disable = dynamo_disable # pylint: disable=protected-access
    if failed:
        emit(f"[red]failed checks: {', '.join(failed)}[/red]")
    if "head dim 40 + hadamard" in failed and shared.opts.sdnq_attention_use_hadamard:
        emit("[yellow]current config has Use Hadamard enabled: models with non power-of-2 head dims (SD 1.5) fail at generation with it[/yellow]")
    return failed


def make_prep_fn(q, k, v, attn_mask, kwargs):
    # mirror sdnq_triton_atten's prep call so the prep column measures the same code path
    from modules.sdnq.kernels import triton_atten as atten_module
    from modules.sdnq.quant_utils import get_hadamard, get_hadamard_group_size
    from modules.sdnq.utils import next_power_of_2
    matmul_dtype = kwargs.get("matmul_dtype", "int8")
    do_quantize = kwargs.get("do_quantize", True)
    hadamard_group_size = kwargs.get("hadamard_group_size", 256)
    hadamard = None
    if kwargs.get("use_hadamard", False) and do_quantize and matmul_dtype not in {None, "none", "no"}:
        channel_size = next_power_of_2(min(q.shape[-1], k.shape[-1]))
        hadamard_group_size = min(hadamard_group_size, channel_size)
        enabled, hadamard_group_size = get_hadamard_group_size(channel_size, hadamard_group_size)
        if enabled:
            hadamard = get_hadamard(hadamard_group_size, dtype=q.dtype, device=q.device)
    def prep():
        return atten_module.get_attn_inputs( # module lookup so the static-compile patch applies
            query=q, key=k, value=v, hadamard=hadamard, attn_mask=attn_mask,
            dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
            smooth_k=kwargs.get("smooth_k", False), hadamard_group_size=hadamard_group_size,
            matmul_dtype=matmul_dtype, pv_matmul_dtype=kwargs.get("pv_matmul_dtype", None),
            do_quantize=do_quantize, out_dtype=None,
        )
    return prep


def bench_shape(preset, iters, warmup, position=None, config_timeout=300, fp8_result=None):
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
        if config_id == "fp8qk" and not (fp8_result and fp8_result["qk"][0]):
            continue
        if config_id == "fp8pv" and not (fp8_result and fp8_result["pv"][0]):
            continue
        selected_configs.append((config_id, label, kwargs))

    def make_table():
        shape_table = Table(box=box.SIMPLE_HEAVY)
        shape_table.add_column("config")
        shape_table.add_column("median time", justify="right")
        shape_table.add_column("prep", justify="right")
        shape_table.add_column("speedup", justify="right")
        shape_table.add_column("error", justify="right")
        return shape_table

    def make_panel(content):
        return Panel(content, title=f"{preset}: batch={batch} heads={heads} tokens={tokens} head_dim={head_dim} {dtype_label()}", subtitle=f"[dim]{description}[/dim]", box=ROUNDED_BOX, expand=False)

    table = make_table()
    panel = make_panel(table)
    results = {}
    rows = []
    base_ms = None
    prefix = f"shape {position[0]}/{position[1]}  " if position else ""
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        progress.update(task, description=f"{prefix}{preset}: preparing inputs and fp32 reference")
        q, k, v = make_qkv(batch, heads, tokens, head_dim)
        scale = head_dim ** -0.5
        ref = fp32_reference(q, k, v, attn_mask=attn_mask)
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
                prep_cell = "-"
                if kwargs is not None: # time the input prep alone; included in the median
                    try:
                        phase("timing input prep")
                        prep_ms = bench(make_prep_fn(q, k, v, attn_mask, kwargs), warmup, iters)
                        prep_cell = f"[dim]{prep_ms:8.3f} ms[/dim]"
                    except Exception:
                        pass
                if base_ms is None:
                    base_ms = ms
                results[config_id] = (ms, err)
                row = (label, f"{ms:8.3f} ms", prep_cell, speedup_cell(base_ms, ms), f"{err:.5f}")
                rows.append((config_id, ms, row))
                table.add_row(*row)
            except Exception as e:
                results[config_id] = (None, None)
                row = (label, "-", "-", "-", failure_text(e))
                rows.append((config_id, None, row))
                table.add_row(*row)
        # rebuild the table to star the best sdnq config when it beats the baseline
        best_id = best_config(results)
        if base_ms and best_id is not None and results[best_id][0] < base_ms:
            table = make_table()
            for config_id, _ms, row in rows:
                if config_id == best_id:
                    table.add_row(f"* {row[0]}", *row[1:], style="bold") # ascii marker: legacy windows consoles crash rendering non-cp1252 characters
                else:
                    table.add_row(*row)
            panel = make_panel(table)
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    return results


def measured(results, config_id):
    ms, err = results.get(config_id, (None, None))
    return (ms, err) if ms is not None else (None, None)


def best_config(results):
    # lowest error among rows within 5% of the fastest sdnq time
    candidates = [(config_id, ms, err) for config_id, (ms, err) in results.items() if ms is not None and config_id not in ("base", "sage")]
    if not candidates:
        return None
    fastest = min(ms for _config_id, ms, _err in candidates)
    near_fastest = [(config_id, ms, err) for config_id, ms, err in candidates if ms <= fastest * 1.05]
    return min(near_fastest, key=lambda item: item[2])[0]


def build_recommendations(all_results, fp8_result, prep_status):
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
    noquant_ms, _noquant_err = measured(results, "noquant")
    int8_ms, int8_err = measured(results, "int8")

    table = Table(title=f"recommended settings (Compute Settings -> SDNQ Attention), measured at the {reference} shape", box=ROUNDED_BOX)
    table.add_column("setting")
    table.add_column("current", justify="center")
    table.add_column("recommended", justify="center")
    table.add_column("reason")

    def current(key):
        return str(getattr(shared.opts, key))

    rows = []
    # compare against the unquantized sdnq row: quantization has to pay for its own prep
    use_quantized = bool(int8_ms and noquant_ms and int8_ms < noquant_ms * 0.95)
    if int8_ms and noquant_ms:
        if use_quantized:
            quant_reason = f"int8 qk measured x{noquant_ms / int8_ms:.2f} vs unquantized sdnq attention"
        else:
            quant_reason = f"unquantized sdnq measured x{int8_ms / noquant_ms:.2f} vs int8 qk with lower error; quantization prep outweighs the kernel gain on this gpu"
        rows.append(("Use Quantized MatMul", current("sdnq_attention_use_quantized_matmul"), str(use_quantized), quant_reason))
    elif int8_ms and base_ms:
        use_quantized = base_ms / int8_ms >= 1.10
        rows.append(("Use Quantized MatMul", current("sdnq_attention_use_quantized_matmul"), str(use_quantized), f"int8 qk measured x{base_ms / int8_ms:.2f} vs torch sdpa; unquantized sdnq row unavailable"))
    else:
        rows.append(("Use Quantized MatMul", current("sdnq_attention_use_quantized_matmul"), "False", "int8 qk failed to run"))

    qk_reason = "resolves to int8; uint8 remaps to int8"
    fp8qk_ms, _fp8qk_err = measured(results, "fp8qk")
    if fp8qk_ms and int8_ms:
        qk_reason += f"; float8 qk measured x{int8_ms / fp8qk_ms:.2f} vs int8, per-token int8 keeps finer granularity"
    elif fp8_result["qk"][0]:
        qk_reason += "; float8 compiles here but per-token int8 keeps finer granularity"
    rows.append(("MatMul type", current("sdnq_attention_matmul_type"), "auto", qk_reason))

    int8pv_ms, _int8pv_err = measured(results, "int8pv")
    fp8pv_ms, _fp8pv_err = measured(results, "fp8pv")
    pv_measured = [(dtype, name, ms) for dtype, name, ms in [("float8_e4m3fn", "fp8", fp8pv_ms), ("int8", "int8", int8pv_ms)] if ms]
    best_pv = min(pv_measured, key=lambda item: item[2]) if pv_measured else None
    if best_pv and int8_ms and best_pv[2] < int8_ms * 0.95:
        pv_dtype, pv_name, pv_ms = best_pv
        rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), pv_dtype, f"{pv_name} pv measured x{int8_ms / pv_ms:.2f} over int8 qk alone; slightly higher error"))
    else:
        pv_note = f"auto keeps pv unquantized; {' and '.join(name for _dtype, name, _ms in pv_measured)} pv measured no gain here" if pv_measured else "auto keeps pv unquantized"
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
    if not use_quantized:
        emit("[dim]matmul type, pv matmul type, smooth k and hadamard only take effect when quantized matmul is enabled; their rows show what to pick if you enable it[/dim]")

    notes = []
    labels = {config_id: label for config_id, label, _kwargs in bench_configs}
    best_id = best_config(results)
    if base_ms and best_id is not None:
        best_ms = results[best_id][0]
        if best_ms < base_ms * 0.95:
            notes.append(f"sdnq attention is worth enabling on this gpu: {labels[best_id]} measured x{base_ms / best_ms:.2f} vs torch sdpa at {reference}")
        else:
            notes.append(f"[yellow]sdnq attention measured no speedup over torch sdpa on this gpu (best: {labels[best_id]} at x{base_ms / best_ms:.2f}); the sdp override costs performance here[/yellow]")
    if prep_status == "failing_dynamic":
        notes.append("[red]the current environment fails at generation: torch compile cannot build the dynamic-shape input prep; numbers above were measured with the SDNQ_COMPILE_KWARGS='{\"dynamic\": false}' workaround applied, set it (verified working here) or install msvc build tools[/red]")
    elif prep_status == "failing":
        notes.append("[red]the current environment fails at generation: torch compile is broken; numbers above were measured with eager input prep, matching what you get after disabling Dequantize using torch.compile; alternatively install msvc build tools[/red]")
    if not fp8_result["qk"][0]:
        notes.append("[red]float8_e4m3fn is unsupported on this gpu: selecting it in either dropdown fails generation with a compile error[/red]")
    sage_ms, _sage_err = measured(results, "sage")
    if sage_ms and int8_ms:
        notes.append(f"sageattention comparison at {reference}: sdnq int8 {int8_ms:.2f} ms vs sage {sage_ms:.2f} ms; sdnq additionally covers masks, gqa and causal attention")
    notes.append("* marks the best config per shape: lowest error among rows within 5% of the fastest sdnq time")
    notes.append("the int8 qk row is what default settings run: MatMul type auto resolves to int8 and pv stays unquantized")
    notes.append("the prep column is time spent quantizing q/k/v before the attention kernel, included in median time; it shrinks where torch compile can fuse it")
    notes.append("[yellow]non pow2 head dims (sd 1.5): quantized matmul configs currently fail torch compile and hadamard hangs it; disable quantized matmul for sd 1.5 sessions or set SDNQ_COMPILE_KWARGS='{\"dynamic\": false}'[/yellow]")
    notes.append("the six settings above apply on the next generation; toggling the 'SDNQ attention' SDP override requires a restart")
    notes.append("errors are measured on synthetic tensors with outlier-heavy keys; real models, especially qk-normed dits, sit lower")
    emit(Panel("\n".join(notes), title="notes", box=ROUNDED_BOX))


def main():
    global bench_dtype # pylint: disable=global-statement
    args = parse_cli()
    if not load_sdnext():
        sys.exit(1)
    if args.dtype == "auto":
        # webui generation dtype: bf16 where native, fp16 on older gpus where bf16 is emulated
        if isinstance(getattr(devices, "dtype", None), torch.dtype) and devices.dtype in (torch.bfloat16, torch.float16):
            bench_dtype = devices.dtype
    else:
        bench_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    selected = list(full_run) if args.shapes.strip().lower() == "all" else [s.strip() for s in args.shapes.split(",") if s.strip()]
    unknown = [s for s in selected if s not in shape_presets]
    if unknown:
        console.print(f"[red]unknown shape preset(s): {', '.join(unknown)}; available: {', '.join(shape_presets)}[/red]")
        sys.exit(1)
    print_banner(selected, args)
    prep_status, prep_detail = probe_compiled_prep() # must run first: see the cold-start note in probe_compiled_prep
    fp8_result = probe_fp8()
    print_environment(fp8_result, prep_status, prep_detail)
    if not args.skip_checks:
        run_correctness()
    if args.skip_bench:
        if args.save:
            save_transcript(args.save)
        return
    # bench the prep mode the advice points to: compiled, static workaround, or eager
    if prep_status == "failing_dynamic":
        from modules.sdnq.kernels import triton_atten as atten_module
        inner = getattr(atten_module.get_attn_inputs, "_torchdynamo_orig_callable", None)
        atten_module.get_attn_inputs = torch.compile(inner, fullgraph=True, dynamic=False)
        emit("[yellow]dynamic-shape compile is broken here: benchmarking with the dynamic=false workaround applied, numbers match the webui after setting SDNQ_COMPILE_KWARGS='{\"dynamic\": false}'[/yellow]")
    elif prep_status == "failing":
        emit("[yellow]torch compile is broken here: benchmarking with eager input prep, numbers match the webui after disabling Dequantize using torch.compile[/yellow]")
        torch._dynamo.config.disable = True # pylint: disable=protected-access
    all_results = {}
    minimum_vram = {"wan22": 6.0, "ltx2": 3.0, "masked": 6.0}
    for index, preset in enumerate(selected, start=1):
        needed = minimum_vram.get(preset, 2.0)
        if free_vram_gb() < needed:
            emit(f"[yellow]skipping {preset}: needs about {needed:.0f} gb free vram, {free_vram_gb():.1f} gb available[/yellow]")
            continue
        all_results[preset] = bench_shape(preset, args.iters, args.warmup, position=(index, len(selected)), config_timeout=args.config_timeout, fp8_result=fp8_result)
    build_recommendations(all_results, fp8_result, prep_status)
    if args.save:
        save_transcript(args.save)


if __name__ == "__main__":
    main()
