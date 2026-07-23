#!/usr/bin/env python
"""
Benchmark and validate SDNQ attention and weight dequantization on the local GPU.

The attention section runs the kernel from modules/sdnq/kernels/triton_atten.py directly and
compares speed and numerical error against torch scaled_dot_product_attention and
sageattention when installed. Verifies mask, causal, GQA, cross-attention and padding code
paths, probes float8 hardware support and the torch.compile input prep, and prints
recommended values for the Compute Settings -> SDNQ Attention section.

The dequant section builds real SDNQ linear layers per storage dtype (int8, uint4,
float8_e4m3fn, float8_e4m3fn_sdnq, float4_e2m1fn) at Flux.1, Krea 2 and Qwen3 TE layer geometry and
measures eager vs compiled weight dequantization plus the full linear forward with and
without quantized matmul, against a bf16 nn.Linear baseline. For float storage dtypes it
also measures quantized matmul with the MatMul type set explicitly (int8, float16), since
enabled routes them to fp8 matmul, which not every gpu can run. Setting sweeps
(--dequant-sweeps) cover group size, svd rank, weight-side hadamard group size, dequantize
full precision off, dynamic quantization, cpu quantize time and conv2d quantization. Probes
whether compiled dequant of fp8 storage works on this GPU (triton before sm_89 lacks e4m3
conversions) and prints recommended values for the Compute Settings -> SDNQ section.

The block section measures complete configurations (weights dtype x matmul path x attention)
end to end through a dit-style transformer block, with output error at depth one and four
against an fp32 reference block, because component speedups and errors do not compose
multiplicatively.

Benchmarks run in the webui's configured dtype (--dtype overrides). The prep column is the
q/k/v quantization cost outside the kernel, included in the median. Recommendations come
from measured comparisons only.

Shape presets follow real model geometries: sd15, sdxl, sdxl-cross, qwen3-te (Anima TE),
anima, flux2 (Klein), krea2 (segment mask), wan22 (A14B), ltx2 (LTX 2.3), plus masked and
wan22-cfg presets. Run from the sdnext root with the venv active:
    python cli/sdnq-attention-benchmark.py
    python cli/sdnq-attention-benchmark.py --shapes all --json results.json
    python cli/sdnq-attention-benchmark.py --sections dequant
    python cli/sdnq-attention-benchmark.py --shapes wan22,ltx2 --iters 20

The first run of each shape includes triton autotune and torch.compile time; results are
cached on disk and reused by the webui for matching shapes.
"""

import io
import os
import sys
import json
import math
import time
import signal
import logging
import argparse
import tempfile
import importlib.metadata
import importlib.import_module
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
report = {} # structured results mirroring the tables, written out by --json
compiled_dequantize_weight = None # tool-owned compiled variant, built on first use

torch_device_module = torch.xpu if torch.xpu.is_available() else torch.cuda
torch_device = "xpu" if torch.xpu.is_available() else "cuda"


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
    # geometry from the model transformer and text-encoder configs;
    # optional keys: kv_tokens (cross-attention), kv_heads (gqa), causal
    "sd15": dict(batch=2, heads=8, tokens=4096, head_dim=40, desc="SD 1.5 unet self-attention at 512px, batched cfg, head dim padded 40 to 64"),
    "sdxl": dict(batch=2, heads=10, tokens=4096, head_dim=64, desc="SDXL unet self-attention at 1024px, batched cfg"),
    "sdxl-cross": dict(batch=2, heads=10, tokens=4096, kv_tokens=77, head_dim=64, desc="SDXL unet cross-attention at 1024px, 77 text tokens"),
    "qwen3-te": dict(batch=2, heads=16, kv_heads=8, tokens=512, head_dim=128, causal=True, desc="Qwen3 text encoder (Anima), causal gqa 16:8 heads, 512 token prompt"),
    "anima": dict(batch=1, heads=16, tokens=4096, head_dim=128, desc="Anima 1.0 self-attention at 1024px, one cfg pass"),
    "flux2": dict(batch=1, heads=32, tokens=4608, head_dim=128, desc="FLUX.2 Klein 9B joint attention at 1024px, 4096 image plus 512 text tokens"),
    "krea2": dict(batch=1, heads=48, tokens=4608, head_dim=128, desc="Krea 2 12B joint attention at 1024px, 4096 image plus 512 text tokens (128 real), kv expanded from gqa 48:12, segment mask"),
    "wan22": dict(batch=1, heads=40, tokens=32760, head_dim=128, desc="Wan 2.2 A14B self-attention, 832x480 81 frames, one cfg pass"),
    "wan22-cfg": dict(batch=2, heads=40, tokens=32760, head_dim=128, desc="Wan 2.2 A14B self-attention, 832x480 81 frames, batched cfg"),
    "ltx2": dict(batch=1, heads=32, tokens=13376, head_dim=128, desc="LTX 2.3 self-attention, 1216x704 121 frames, one cfg pass"),
    "masked": dict(batch=1, heads=32, tokens=4608, head_dim=128, desc="FLUX.2 Klein shape with boolean key-padding mask, 25% of keys masked"),
}
full_run = ["sd15", "sdxl", "sdxl-cross", "qwen3-te", "anima", "flux2", "krea2", "wan22", "ltx2"]
# settings advice comes from a self-attention shape with the full config set; cross-attention
# and text-encoder shapes measure the hijack's cost there but would mislead as global advice
recommendation_presets = ["flux2", "krea2", "anima", "sdxl", "wan22", "ltx2", "sd15"]
default_shapes = "sdxl,flux2"
all_sections = ["attention", "dequant", "block"]

# combined block benchmark: a dit-style transformer block at krea 2 class geometry, so each row
# measures a complete configuration of weights dtype x matmul path x attention end to end
# generic dit-block geometries from the model transformer configs: flux.1 (3072 wide,
# 24 heads, 4x gelu ff, 4096 image plus 512 text tokens) and krea 2 (6144 wide, 48 heads
# after gqa expansion, swiglu at 16384, same joint sequence)
block_geometries = {
    "flux1": dict(hidden=3072, heads=24, mlp_dim=12288, tokens=4608),
    "krea2": dict(hidden=6144, heads=48, mlp_dim=16384, tokens=4608),
}
block_geometry = block_geometries["flux1"] # active geometry; bench_block_section iterates
block_attention_specs = {
    "sdpa": None, # stock torch sdpa
    "atten int8": dict(matmul_dtype="auto", pv_matmul_dtype="auto"),
    "atten int8 smooth": dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True),
    "atten int8 hadamard": dict(matmul_dtype="auto", pv_matmul_dtype="auto", use_hadamard=True),
    "atten full": dict(matmul_dtype="auto", pv_matmul_dtype="int8", smooth_k=True, use_hadamard=True),
    "sage": "sage", # external baselines, resolved to the sage wrappers in build_bench_block
    "sage fp16 accum": "sagefp16",
}
# id, weights config (None = bf16), use quantized matmul, attention spec; fp8/fp4 rows use the
# dequant path: quantized matmul auto-selects fp8 for float dtypes, unsupported before sm_89
block_configs = [
    ("bf16", None, False, "sdpa"),
    ("bf16-atten", None, False, "atten int8"),
    ("bf16-sage", None, False, "sage"),
    ("bf16-sagefp16", None, False, "sage fp16 accum"),
    ("int8", dict(weights_dtype="int8"), False, "sdpa"),
    ("int8-mm", dict(weights_dtype="int8"), True, "sdpa"),
    ("int8-mm-atten", dict(weights_dtype="int8"), True, "atten int8"),
    ("int8-mm-smooth", dict(weights_dtype="int8"), True, "atten int8 smooth"),
    ("int8-mm-hadamard", dict(weights_dtype="int8"), True, "atten int8 hadamard"),
    ("int8-mm-atten-full", dict(weights_dtype="int8"), True, "atten full"),
    ("int8-mm-sage", dict(weights_dtype="int8"), True, "sage"),
    ("int8-mm-sagefp16", dict(weights_dtype="int8"), True, "sage fp16 accum"),
    ("int6", dict(weights_dtype="int6"), False, "sdpa"),
    ("uint4-mm", dict(weights_dtype="uint4"), True, "sdpa"),
    ("uint4-mm-atten", dict(weights_dtype="uint4"), True, "atten int8"),
    ("fp8sdnq", dict(weights_dtype="float8_e4m3fn_sdnq"), False, "sdpa"),
    ("fp8sdnq-atten", dict(weights_dtype="float8_e4m3fn_sdnq"), False, "atten int8"),
    ("nvfp4", dict(weights_dtype="float4_e2m1fn", group_size=16), False, "sdpa"),
    ("nvfp4-atten", dict(weights_dtype="float4_e2m1fn", group_size=16), False, "atten int8"),
]

# benchmark configs: id, label, kwargs for sdnq_triton_atten (None = external baseline);
# fp8 configs run only on gpus where the float8 probe passes
bench_configs = [
    ("base", "torch sdpa", None),
    ("sage", "sageattention", None), # label resolved to the dispatched kernel by sage_kernel_label
    ("sagefp16", "sage int8 qk + fp16 pv, fp16 accum", None), # sm86 only
    ("amdflash", "triton flash (amd)", None),
    ("noquant", "sdnq, quantized matmul off", dict(do_quantize=False)),
    ("int8", "sdnq int8 qk", dict(matmul_dtype="auto", pv_matmul_dtype="auto")),
    ("smooth", "sdnq int8 qk + smooth k", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True)),
    ("hadamard", "sdnq int8 qk + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", use_hadamard=True)),
    ("smooth_hadamard", "sdnq int8 qk + smooth + hadamard", dict(matmul_dtype="auto", pv_matmul_dtype="auto", smooth_k=True, use_hadamard=True)),
    ("fp16pv", "sdnq int8 qk + fp16 pv", dict(matmul_dtype="auto", pv_matmul_dtype="float16")),
    ("int8pv", "sdnq int8 qk + int8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="int8")),
    ("fp8pv", "sdnq int8 qk + fp8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="float8_e4m3fn")),
    ("full", "sdnq int8 qk + smooth + hadamard + int8 pv", dict(matmul_dtype="auto", pv_matmul_dtype="int8", smooth_k=True, use_hadamard=True)),
    ("fp16qk", "sdnq fp16 qk", dict(matmul_dtype="float16", pv_matmul_dtype="auto")),
    ("fp8qk", "sdnq fp8 qk", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="auto")),
    ("fp8full", "sdnq fp8 qk + fp8 pv", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="float8_e4m3fn")),
]
# external baselines are compared against but never starred or recommended as sdnq configs
external_config_ids = ("base", "sage", "sagefp16", "amdflash")
# every preset runs the full config list (availability gates still apply per config); only
# hard technical exclusions live here, never runtime trims. sd15: compiling hadamard with
# a non pow2 head dim currently hangs torch inductor
preset_excluded_configs = {"sd15": {"hadamard", "smooth_hadamard", "full"}}

# weight dequant benchmark geometry from the model transformer configs: flux.1 attention
# and feed-forward linears (3072 wide), the qwen3 1.7b (anima te) gate/up projection for
# text-encoder geometry, and krea 2 12b linears (6144 wide: standalone wq, swiglu gate)
dequant_shapes = [
    ("flux1 attn 3072x3072", 3072, 3072),
    ("flux1 mlp 12288x3072", 12288, 3072),
    ("te mlp 6144x2048", 6144, 2048),
    ("krea2 wq 6144x6144", 6144, 6144),
    ("krea2 mlp 16384x6144", 16384, 6144),
]
dequant_forward_tokens = 4096 # rows of the forward-bench input, one 1024px image worth of tokens
# id, label, sdnq config kwargs; int8 first: the ui default and the dominant pre-quantized format
dequant_dtype_configs = [
    ("int8", "int8", dict(weights_dtype="int8")),
    ("uint8", "uint8", dict(weights_dtype="uint8")),
    ("int6", "int6", dict(weights_dtype="int6")),
    ("uint6", "uint6", dict(weights_dtype="uint6")),
    ("int4", "int4", dict(weights_dtype="int4")),
    ("uint4", "uint4", dict(weights_dtype="uint4")),
    ("int2", "int2", dict(weights_dtype="int2")),
    ("uint2", "uint2", dict(weights_dtype="uint2")),
    ("fp8", "float8_e4m3fn", dict(weights_dtype="float8_e4m3fn")),
    ("fp8sdnq", "float8_e4m3fn_sdnq", dict(weights_dtype="float8_e4m3fn_sdnq")),
    ("nvfp4", "float4_e2m1fn g16", dict(weights_dtype="float4_e2m1fn", group_size=16)),
]
# svd/hadamard variants measured on top of these dtypes at the first dequant shape; low bits are
# where rotation is expected to pay, int8 is the control
dequant_variant_dtypes = ["int8", "uint4", "uint2"]
dequant_variant_configs = [
    ("hadamard", dict(use_hadamard=True)),
    ("svd", dict(use_svd=True)),
    ("svd+hadamard", dict(use_svd=True, use_hadamard=True)),
]
# float storage dtypes route quantized matmul to fp8 under auto; measure the explicit MatMul
# type alternatives so advice on gpus without fp8 matmul cites numbers instead of escape hatches
float_mm_dtypes = ["fp8", "fp8sdnq", "nvfp4"]
float_mm_alternative_dtypes = ["int8", "float16"]
# setting sweeps for the remaining measurable SDNQ options, each at the first dequant shape:
# group size 0 = auto, -1 = row-wise; explicit values snap down to a divisor of in_features
# and grouped weights force a per-forward re-quantize when quantized matmul is on
group_sweep_dtypes = ["int8", "uint4"]
group_sweep_values = [0, -1, 32, 64, 128, 256]
svd_rank_sweep_dtypes = ["int8", "uint4"]
svd_rank_sweep_values = [16, 32, 64, 128]
hadamard_group_values = [32, 64, 128, 256] # weight-side rotation group, swept on int8
toggle_dtypes = ["int8", "uint4"] # dequantize using full precision off is measured on these
dynamic_quant_requests = ["uint2", "uint4", "int6"] # dynamic quantization escalates until loss passes
# conv geometry: sdxl unet mid-block and vae decoder convs, the two conv-heavy model classes
conv_shapes = [
    ("unet 1280x1280 3x3 @32px", 1280, 1280, 3, 32),
    ("vae 512x512 3x3 @256px", 512, 512, 3, 256),
]
conv_configs = [ # id, weights config (None = bf16 baseline), use conv quantized matmul
    ("bf16", None, False),
    ("int8", dict(weights_dtype="int8"), False),
    ("int8-mm", dict(weights_dtype="int8"), True),
    ("uint8-mm", dict(weights_dtype="uint8"), True),
]
all_dequant_sweeps = ["groups", "svd", "hgroups", "toggles", "conv"]
all_mm_backends = ["torch", "triton"]
recommend_error_cap = 2.0 # a faster config is not recommended when it multiplies measured output error beyond this
recommend_speed_margin = 0.90 # an on/off verdict needs new_ms <= old_ms * this; sub-margin wins sit inside 12-iter run noise and never justify added error

atten_settings = [
    ("sdnq_attention_matmul_type", "MatMul type"),
    ("sdnq_attention_pv_matmul_type", "PV MatMul type"),
    ("sdnq_attention_smooth_k", "Use Smooth K"),
    ("sdnq_attention_use_hadamard", "Use Hadamard"),
    ("sdnq_attention_hadamard_group_size", "Hadamard Group Size"),
]


def parse_cli():
    parser = argparse.ArgumentParser(description="benchmark and validate sdnq attention and weight dequantization on the local gpu")
    parser.add_argument("--sections", type=str, default=",".join(all_sections), help=f"comma-separated benchmark sections: {', '.join(all_sections)} (default: %(default)s)")
    parser.add_argument("--dequant-dtypes", type=str, default="all", help=f"comma-separated dequant dtype configs: {', '.join(dtype_id for dtype_id, _label, _cfg in dequant_dtype_configs)}; 'all' runs every one (default: %(default)s)")
    parser.add_argument("--dequant-variants", type=str, default="all", help=f"comma-separated svd/hadamard variants benched on {', '.join(dequant_variant_dtypes)}: {', '.join(variant_id for variant_id, _cfg in dequant_variant_configs)}; 'all' or 'none' (default: %(default)s)")
    parser.add_argument("--dequant-sweeps", type=str, default="all", help=f"comma-separated setting sweeps in the dequant section: {', '.join(all_dequant_sweeps)}; 'all' or 'none' (default: %(default)s)")
    parser.add_argument("--mm-backends", type=str, default="none", help=f"comma-separated quantized-matmul backends to compare in one run: {', '.join(all_mm_backends)}; 'none' benches only the backend this device selects (default: %(default)s)")
    parser.add_argument("--mm-rounds", type=int, default=2, help="alternating rounds per matmul backend, fastest kept, so clock drift cancels instead of favouring one backend (default: %(default)s)")
    parser.add_argument("--block-configs", type=str, default="all", help=f"comma-separated combined block configs: {', '.join(config_id for config_id, _w, _mm, _a in block_configs)}; 'all' runs every one (default: %(default)s)")
    parser.add_argument("--shapes", type=str, default=default_shapes, help=f"comma-separated attention shape presets: {', '.join(shape_presets)}; 'all' runs {', '.join(full_run)} (default: %(default)s)")
    parser.add_argument("--iters", type=int, default=12, help="minimum timed iterations per config, scaled up for fast kernels (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=4, help="minimum warmup iterations per config, scaled up for fast kernels (default: %(default)s)")
    parser.add_argument("--skip-checks", action="store_true", help="skip kernel correctness checks")
    parser.add_argument("--skip-bench", action="store_true", help="skip benchmarks, run checks and the fp8 and compile probes only")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16"], help="tensor dtype for benchmarks; auto uses the dtype the webui selected for this gpu (default: %(default)s)")
    parser.add_argument("--config-timeout", type=int, default=300, help="best effort: abort a config whose compile plus first call exceeds this many seconds, 0 disables; cannot interrupt native-level hangs (default: %(default)s)")
    parser.add_argument("--save", type=str, default=None, help="write a plain-text copy of all tables and notes to this file; keeps colors and live progress on the terminal, unlike piping through tee")
    parser.add_argument("--json", type=str, default=None, help="write structured results (environment, probes, per-shape and dequant timings, recommendations) to this file")
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


@contextmanager
def capture_console_output():
    # gate stdout and stderr at the fd level: covers python loggers regardless of their
    # handler plumbing plus native-code writes (onnxruntime device discovery on wsl).
    # capture to a file rather than devnull: the sdnext bootstrap calls sys.exit on fatal
    # startup errors (modules/loader.py, installer.py), so discarding this stream turns a
    # startup failure into a silent process exit with nothing to debug.
    # the python-level streams must move with the fds: on a windows console sys.stdout writes
    # through the console api using the handle behind fd 1, so once fd 1 is a file the next
    # print raises OSError 'the handle is invalid' and, with stderr equally broken, kills the
    # interpreter before any handler or finally can run. buffer is deliberately left open:
    # library loggers built during the import (transformers, torch) capture sys.stderr at
    # handler construction and would raise on a closed stream long after the window closes
    captured = {"text": ""}
    buffer = io.StringIO()
    saved_stdout_fd, saved_stderr_fd = os.dup(1), os.dup(2)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sink_fd, sink_path = tempfile.mkstemp(prefix="sdnq-bench-startup-", suffix=".log")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)
        sys.stdout, sys.stderr = buffer, buffer
        yield captured
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        os.close(sink_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        native_text = ""
        try:
            with open(sink_path, "r", encoding="utf-8", errors="replace") as fh:
                native_text = fh.read()
            os.unlink(sink_path)
        except OSError:
            pass
        captured["text"] = buffer.getvalue() + native_text


def load_sdnext():
    global shared, devices, sdnq_triton_atten # pylint: disable=global-statement
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if not torch_device_module.is_available():
        console.print("[red]no cuda, rocm or xpu device available: sdnq attention requires a gpu with triton[/red]")
        return False
    stock_sdpa = torch.nn.functional.scaled_dot_product_attention
    # the webui startup log (device detect, packages, settings validation) is noise here: the
    # environment panel reports the stack. the bootstrap reconfigures its loggers during
    # import and onnxruntime's device discovery warns from c++, so gate the os-level fds for
    # the import window; on failure the captured log is replayed after the gate lifts
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        with capture_console_output() as startup_log:
            from modules import shared as shared_module
            from modules import devices as devices_module
            from modules.sdnq.kernels.triton_atten import sdnq_triton_atten as atten
    except BaseException as e: # pylint: disable=broad-exception-caught # SystemExit is not an Exception: the bootstrap exits on a failed torch or library import
        if isinstance(e, KeyboardInterrupt):
            raise
        detail = f"exited with code {e.code}" if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        console.print(f"[red]sdnext failed to start: {detail}[/red]")
        text = startup_log["text"].strip()
        if text:
            console.print(Panel(escape(text[-4000:]), title="sdnext startup log", box=ROUNDED_BOX))
        console.print("run from the sdnext root with the venv active; triton is required")
        return False
    # keep the sdnext loggers quiet after the bootstrap too, so stray log lines cannot tear
    # the live tables mid-bench
    try:
        import installer
        installer.log.setLevel(logging.CRITICAL)
        from modules.logger import log as sdnext_log
        sdnext_log.setLevel(logging.CRITICAL)
    except Exception:
        pass
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
        if torch_device_module.get_device_capability(torch.device(torch_device)) == (8, 6):
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


def sage_kernel_label():
    # "sage" is a different kernel per gpu: sageattention's dispatch (core.py sageattn) picks the
    # pv dtype and accumulator by arch and cuda version, and modules/attention.py forces the cuda
    # fp16-pv kernel on sm86. name what ran so reports from different gpus stay comparable
    try:
        capability = torch_device_module.get_device_capability(torch.device(torch_device))
        cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    except Exception:
        return "sageattention"
    fp8_accum = "fp32+fp16" if cuda_version >= (12, 8) else "fp32+fp32" # sageattention2++ needs cuda 12.8
    if capability == (7, 5):
        return "sage int8 qk + fp16 pv, triton"
    if capability in {(8, 0), (8, 6), (8, 7)}:
        return "sage int8 qk + fp16 pv, fp32 accum"
    if capability == (9, 0):
        return "sage int8 qk + fp8 pv, fp32+fp32 accum"
    if capability in {(8, 9), (10, 0), (12, 0), (12, 1)}:
        return f"sage int8 qk + fp8 pv, {fp8_accum} accum"
    return "sageattention"


def config_label(config_id, label):
    return sage_kernel_label() if config_id == "sage" else label


def amd_triton_flash():
    # sdnext's vendored flash attention for rocm/zluda (modules/flash_attn_triton_amd): the
    # relevant baseline on amd, where stock sdpa has no flash kernel and sage is unavailable
    try:
        if getattr(devices, "backend", None) not in {"rocm", "zluda"}:
            return None
        from modules.flash_attn_triton_amd import interface_fa
        def flash_fn(q, k, v, scale, is_causal=False):
            head_size = q.size(3)
            if head_size % 8 != 0:
                pad = 8 - head_size % 8
                q = torch.nn.functional.pad(q, [0, pad])
                k = torch.nn.functional.pad(k, [0, pad])
                v = torch.nn.functional.pad(v, [0, pad])
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = torch.zeros_like(q)
            interface_fa.fwd(q, k, v, out, 0.0, scale, is_causal)
            return out[..., :head_size].transpose(1, 2)
        return flash_fn
    except Exception:
        return None


def sage_attention_fp16_accum():
    # sage's fast sm86 mode: fp16 pv accumulation, which sdnext deliberately does not ship
    # (accumulator overflow risk on extreme activations); benched to quantify what that
    # choice costs in speed and buys in error
    try:
        if torch_device_module.get_device_capability(torch.device(torch_device)) != (8, 6):
            return None
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_fn(q, k, v, scale):
            return sageattn_qk_int8_pv_fp16_cuda(q=q, k=k, v=v, tensor_layout="HND", is_causal=False, sm_scale=scale, return_lse=False, pv_accum_dtype="fp16")
        return sage_fn
    except Exception:
        return None


def make_qkv(batch, heads, tokens, head_dim, structured=True, kv_heads=None, kv_tokens=None):
    # structured keys carry a shared per-channel bias plus a few outlier channels, mimicking the
    # key statistics that motivate smoothing and rotation; absolute error varies by model
    # architecture while the relative ordering of configs holds
    generator = torch.Generator(device=torch_device).manual_seed(1234)
    kv_heads = kv_heads or heads
    kv_tokens = kv_tokens or tokens
    q = torch.randn(batch, heads, tokens, head_dim, device=torch_device, dtype=bench_dtype, generator=generator)
    k = torch.randn(batch, kv_heads, kv_tokens, head_dim, device=torch_device, dtype=bench_dtype, generator=generator)
    v = torch.randn(batch, kv_heads, kv_tokens, head_dim, device=torch_device, dtype=bench_dtype, generator=generator)
    if structured:
        k = k + torch.randn(1, kv_heads, 1, head_dim, device=torch_device, dtype=bench_dtype, generator=generator) * 3.0
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


def max_token_err(out, ref, eps=1e-6):
    # worst single-token relative error over the feature axis: catches localized corruption
    # (one garbage image region, one dead token) that a global norm averages away
    out = out.to(torch.float32)
    numerator = (out - ref).norm(dim=-1)
    denominator = ref.norm(dim=-1).clamp_min(eps)
    return (numerator / denominator).max().item()


def err_cell(value, fmt="{:.5f}"):
    # non-finite outputs are the black-image failure class; label them instead of printing nan
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "[red]non-finite[/red]"
    return fmt.format(value)


def fp32_linear_reference(x, weight_fp32):
    # sdnext enables tf32 globally; compute the linear reference in true fp32, matching fp32_reference
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.nn.functional.linear(x.to(torch.float32), weight_fp32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


def cuda_context_alive():
    # a kernel fault (misaligned address, illegal memory access) is sticky: every later cuda
    # call in the process fails; detect it so sections abort cleanly instead of cascading
    try:
        torch.zeros(1, device=torch_device)
        torch_device_module.synchronize()
        return True
    except Exception:
        return False


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


def bench_stats(fn, warmup, iters, on_phase=None):
    """Time fn and return (median ms, relative sigma of that median).

    The sigma covers within-sample dispersion only (IQR-based standard error of a
    median); slow clock and thermal drift across a run is tracked separately by the
    per-shape sentinel re-measurements and folded in at verdict time."""
    if on_phase:
        on_phase("warmup")
    fn()
    torch_device_module.synchronize()
    started = time.perf_counter()
    fn()
    torch_device_module.synchronize()
    estimate = max(time.perf_counter() - started, 1e-6)
    # scale counts to a time budget so short kernels get enough activity to ramp gpu clocks
    warmup = min(max(warmup, int(0.2 / estimate)), 500)
    iters = min(max(iters, int(0.5 / estimate)), 500)
    for _ in range(warmup):
        fn()
    torch_device_module.synchronize()
    if on_phase:
        on_phase(f"timing {iters} iterations")
    events = [(torch_device_module.Event(enable_timing=True), torch_device_module.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch_device_module.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in events)
    n = len(times)
    median = times[n // 2]
    iqr = times[(3 * n) // 4] - times[n // 4]
    # standard error of a median: 1.2533 * sigma / sqrt(n), robust sigma from IQR / 1.349
    sigma_rel = (1.2533 * (iqr / 1.349) / math.sqrt(n)) / median if median > 0 else 0.0
    return median, sigma_rel


def bench(fn, warmup, iters, on_phase=None):
    return bench_stats(fn, warmup, iters, on_phase)[0]


drift_samples = [] # |ln ratio| of per-shape sentinel re-measurements; tracks run-level clock drift


def record_drift(first_ms, repeat_ms):
    if first_ms and repeat_ms:
        drift_samples.append(abs(math.log(repeat_ms / first_ms)))


def run_drift_sigma():
    # rms of the sentinel deltas, floored so a lucky pair of samples cannot claim a
    # noiseless run; replays of json files inject the stored value via drift_override
    if drift_override is not None:
        return drift_override
    if not drift_samples:
        return 0.0
    rms = math.sqrt(sum(d * d for d in drift_samples) / len(drift_samples))
    return min(max(rms, 0.005), 0.10)


drift_override = None
verdict_z = 1.28 # one-sided 90%: an on/off verdict is only stated when its margin test clears this
# per-test z holding the family-wise confidence at 90% across k tested candidates (sidak)
sidak_z = {1: 1.28, 2: 1.63, 3: 1.82, 4: 1.95}

# the synthetic-tensor errors are seed-fixed math and land within a few percent across
# nvidia, amd and intel gpus; a value outside its band means the measurement itself is
# off (wrong dtype, broken kernel, degraded reference), not an interesting gpu
error_sanity_bands = { # (shape, config) -> (low, high), bfloat16 runs only
    ("anima", "smooth"): (0.0159, 0.0215),
    ("sdxl", "smooth"): (0.0185, 0.0250),
    ("anima", "noquant"): (0.0021, 0.0043),
    ("sdxl", "noquant"): (0.0025, 0.0050),
}


def row_sigma(entry, key="ms"):
    # relative sigma of one measured row: within-sample dispersion plus the run-level
    # drift sampled by the sentinels; None when the row predates sigma capture, which
    # drops the verdict back to a point-estimate test (old json replays)
    boot = entry.get(f"{key}_sigma") if isinstance(entry, dict) else None
    drift = run_drift_sigma()
    if boot is None:
        return drift or None
    return math.sqrt(boot * boot + drift * drift)


def pair_sigma(entry_a, key_a, entry_b, key_b):
    sa, sb = row_sigma(entry_a, key_a), row_sigma(entry_b, key_b)
    if sa is None and sb is None:
        return None
    return math.sqrt((sa or 0.0) ** 2 + (sb or 0.0) ** 2)


def speed_verdict(new_ms, old_ms, sigma=None, margin=None, z=verdict_z):
    """Three-zone test of "new beats old by the speed margin": returns 'faster',
    'not_faster', or 'inconclusive' when the gap sits inside z*sigma of the margin.
    Without a sigma the test degrades to the plain point-estimate threshold."""
    if not (new_ms and old_ms):
        return None
    margin = margin if margin is not None else recommend_speed_margin
    gap = math.log(old_ms / new_ms) + math.log(margin) # > 0 clears the margin
    if sigma:
        if gap > z * sigma:
            return "faster"
        if gap < -z * sigma:
            return "not_faster"
        return "inconclusive"
    return "faster" if gap > 0 else "not_faster"


def free_vram_gb():
    free, _total = torch_device_module.mem_get_info()
    return free / 1024**3


def fp8_compile_gate_flag():
    # False on gpus where sdnq upcasts e4m3 storage to the scale dtype before the compiled
    # dequant, because triton cannot convert e4m3 there; absent on builds without the gate
    try:
        from modules.sdnq import kernel_wrappers as sdnq_kernel_wrappers
        return getattr(sdnq_kernel_wrappers, "is_fp8_compile_supported", None)
    except Exception:
        return None


def fp8_failure_is_capability(detail):
    # the triton pre-sm_89 signature; anything else is an environment failure where a torch
    # or triton issue is more likely than a missing hardware capability
    detail = detail or ""
    return "fp8e4nv" in detail or "not supported in this architecture" in detail


def print_environment(fp8_result, prep_status, prep_detail, weight_dequant_result=None):
    device = torch.device(torch_device)
    capability = torch_device_module.get_device_capability(device)
    # backend runtime versions (cuda/cudnn/driver, hip, ipex, openvino, directml) so a shared
    # report identifies the stack without inferring it from the torch version string
    try:
        gpu_info = devices.get_gpu_info() or {}
    except Exception:
        gpu_info = {}
    runtime_versions = {key: gpu_info[key] for key in ("cuda", "hip", "cudnn", "driver", "ipex", "openvino", "directml") if gpu_info.get(key)}
    runtime_line = f"python: {sys.version.split()[0]} ({sys.platform})  backend: {getattr(devices, 'backend', 'unknown')}"
    if runtime_versions:
        runtime_line += "  " + "  ".join(f"{key}: {value}" for key, value in runtime_versions.items())
    lines = [
        f"device: [cyan]{torch_device_module.get_device_name(device)}[/cyan] capability={capability}",
        f"torch: {torch.__version__}  triton: {triton_version() or 'not installed'}  sageattention: {package_version('sageattention') or 'not installed'}  flash-attn: {package_version('flash-attn') or 'not installed'}",
        runtime_line,
        f"benchmark dtype: {dtype_label()}",
    ]
    if fp8_result is not None:
        if fp8_result["qk"][0]:
            lines.append("float8_e4m3fn matmul: [green]supported[/green]")
        elif fp8_failure_is_capability(fp8_result["qk"][1]):
            lines.append(f"float8_e4m3fn matmul: [red]not supported on this gpu, selecting it fails generation[/red] [dim]({escape(fp8_result['qk'][1])})[/dim]")
        else:
            lines.append(f"float8_e4m3fn matmul: [red]failed to compile in this environment, selecting it fails generation[/red]; the error is not the hardware-capability signature, a torch or triton issue is more likely than the gpu [dim]({escape(fp8_result['qk'][1])})[/dim]")
    lines.append(f"sdnq attention enabled in current config: {'[green]yes[/green]' if 'SDNQ attention' in shared.opts.sdp_overrides else '[yellow]no, enable via Compute Settings -> SDP overrides (requires restart)[/yellow]'}")
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
    if weight_dequant_result is not None:
        gate_flag = fp8_compile_gate_flag()
        e4m3_ok, e4m3_detail = weight_dequant_result["float8_e4m3fn"]
        e5m2_ok, e5m2_detail = weight_dequant_result["float8_e5m2"]
        if e4m3_ok:
            lines.append("compiled weight dequant, float8_e4m3fn storage: [green]supported[/green]")
        else:
            lines.append(f"compiled weight dequant, raw float8_e4m3fn storage: [red]fails on this gpu[/red] [dim]({escape(e4m3_detail)})[/dim]")
            if gate_flag is False:
                lines.append(f"  sdnq upcasts e4m3 weights to {dtype_label()} before the compiled dequant, so generation is safe and the fp8 rows below are measured that way (SDNQ_ALLOW_FP8_COMPILE overrides)")
            elif gate_flag is None:
                lines.append("  [red]this sdnq build has no fp8 compile gate: loading fp8 storage weights with Dequantize using torch.compile enabled fails at generation[/red]")
        e5m2_verdict = "[green]supported[/green]" if e5m2_ok else f"[red]fails[/red] [dim]({escape(e5m2_detail)})[/dim]"
        lines.append(f"compiled weight dequant, float8_e5m2 storage: {e5m2_verdict}")
    overrides = [f"{key}={value}" for key, value in os.environ.items() if key.startswith("SDNQ_TRITON_ATTEN") or key.startswith("SDNQ_ALLOW_FP8") or key.startswith("SDNQ_COMPILE")]
    if overrides:
        lines.append(f"env overrides: {' '.join(overrides)}")
    emit(Panel("\n".join(lines), title="environment", box=ROUNDED_BOX))
    report["environment"] = dict(
        device=torch_device_module.get_device_name(device),
        capability=f"{capability[0]}.{capability[1]}",
        python=sys.version.split()[0],
        platform=sys.platform,
        backend=str(getattr(devices, "backend", None)),
        torch=str(torch.__version__),
        triton=triton_version(),
        sageattention=package_version("sageattention"),
        dtype=dtype_label(),
        **runtime_versions,
        fp8_attention_matmul=fp8_result["qk"][0] if fp8_result is not None else None,
        compiled_input_prep=prep_status,
        fp8_compile_gate=fp8_compile_gate_flag(),
        weight_dequant_compile={name: dict(ok=ok, detail=detail) for name, (ok, detail) in (weight_dequant_result or {}).items()},
    )


def print_banner(selected, sections, args):
    lines = [
        "measures sdnq attention and weight dequantization speed and accuracy on this gpu and recommends values for [cyan]Compute Settings -> SDNQ / SDNQ Attention[/cyan]",
        f"sections: [cyan]{', '.join(sections)}[/cyan]",
    ]
    if args.skip_bench:
        lines.append("running correctness checks and the float8 and compile probes only (--skip-bench)")
    elif "attention" in sections:
        lines.append(f"attention shapes: [cyan]{', '.join(selected)}[/cyan]  (available: {', '.join(shape_presets)}; pass --shapes to match the models you use)")
        lines.append("first run compiles triton kernels per shape and can take several minutes; repeat runs are much faster")
    if args.mm_backends.strip().lower() not in {"none", ""} and "dequant" in sections:
        lines.append(f"matmul backends compared in one run: [cyan]{args.mm_backends}[/cyan]  (swapped in-process, {args.mm_rounds} alternating rounds per dtype)")
    if args.save:
        lines.append(f"a plain-text copy of the results will be saved to [cyan]{args.save}[/cyan]")
    if args.json:
        lines.append(f"structured results will be saved to [cyan]{args.json}[/cyan]")
    console.print(Panel("\n".join(lines), title="sdnq benchmark", box=ROUNDED_BOX))


def probe_fp8():
    # hardware probe, prep runs eager via the module-global swap so the verdict isolates the
    # attention kernel: fp8 kernel compile fails on pre-ada nvidia and pre-rdna4/cdna3 amd;
    # prep failures are dtype-independent and probed separately. do not force eager by
    # toggling torch._dynamo.config.disable: torch 2.13+ raises "found no compiled frames"
    # for fullgraph-compiled functions called inside a disable window
    try:
        from modules.sdnq import kernel_wrappers as sdnq_kernel_wrappers
        is_fp8_mm_supported = getattr(sdnq_kernel_wrappers, "is_fp8_mm_supported", True)
    except Exception:
        is_fp8_mm_supported = True
    if not is_fp8_mm_supported:
        return dict(qk=(False, "FP8 matmul is not supported in this architecture"), pv=(False, "FP8 matmul is not supported in this architecture"))
    from modules.sdnq.kernels import triton_atten as atten_module
    q, k, v = make_qkv(1, 2, 256, 64, structured=False)
    result = {}
    compiled_prep = atten_module.get_attn_inputs
    inner_prep = getattr(compiled_prep, "_torchdynamo_orig_callable", None)
    if inner_prep is not None:
        atten_module.get_attn_inputs = inner_prep
    try:
        with console.status("probing float8 support") as status:
            for name, kwargs in [("qk", dict(matmul_dtype="float8_e4m3fn", pv_matmul_dtype="auto")), ("pv", dict(matmul_dtype="auto", pv_matmul_dtype="float8_e4m3fn"))]:
                status.update(f"probing float8 support: {name} matmul (a compile failure here is normal on older gpus)")
                try:
                    out = sdnq_triton_atten(q, k, v, **kwargs)
                    torch_device_module.synchronize()
                    result[name] = (not bool(torch.isnan(out).any().item()), "compiles and runs")
                except Exception as e:
                    result[name] = (False, f"{type(e).__name__}: {error_summary(e, 120)}")
    finally:
        atten_module.get_attn_inputs = compiled_prep
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
            torch_device_module.synchronize()
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
            torch_device_module.synchronize()
            static_ok = True
        except Exception:
            pass
        finally:
            atten_module.get_attn_inputs = compiled_prep
            torch._dynamo.reset() # pylint: disable=protected-access
    return ("failing_dynamic" if static_ok else "failing"), detail


def make_source_weight(out_features, in_features, seed=1234):
    # structured weights with a few high-magnitude input channels, mimicking the outlier
    # statistics of real dit linears; keeps quantization error columns in a realistic range
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    weight = torch.randn(out_features, in_features, device=torch_device, dtype=bench_dtype, generator=generator) * 0.02
    weight[:, [3, in_features // 2, in_features - 5]] *= 8.0
    return weight


class BenchBlock(torch.nn.Module):
    # dit-style block: fused qkv self-attention plus a gelu mlp, both with residuals; the
    # attention_fn attribute is set per benchmark config (stock sdpa or sdnq attention)
    def __init__(self, hidden, heads, mlp_dim, device=None, dtype=None):
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.LayerNorm(hidden, elementwise_affine=False, device=device, dtype=dtype)
        self.norm2 = torch.nn.LayerNorm(hidden, elementwise_affine=False, device=device, dtype=dtype)
        self.qkv = torch.nn.Linear(hidden, hidden * 3, bias=False, device=device, dtype=dtype)
        self.proj = torch.nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(hidden, mlp_dim, bias=False, device=device, dtype=dtype)
        self.down = torch.nn.Linear(mlp_dim, hidden, bias=False, device=device, dtype=dtype)
        self.attention_fn = None

    def forward(self, x):
        batch, tokens, channels = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(batch, tokens, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        attn = self.attention_fn(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(batch, tokens, channels)
        x = x + self.proj(attn)
        h = self.norm2(x)
        return x + self.down(torch.nn.functional.gelu(self.up(h)))


def make_block_master():
    # one master weight set shared by every block config, so all rows quantize identical weights
    hidden, heads, mlp_dim = block_geometry["hidden"], block_geometry["heads"], block_geometry["mlp_dim"]
    block = BenchBlock(hidden, heads, mlp_dim, device=torch_device, dtype=bench_dtype)
    with torch.no_grad():
        for seed, linear in enumerate((block.qkv, block.proj, block.up, block.down), start=1):
            linear.weight.copy_(make_source_weight(linear.out_features, linear.in_features, seed=seed))
    return {key: value.clone() for key, value in block.state_dict().items()}


def build_bench_block(master_sd, weights_cfg, use_mm, attention_spec):
    from modules.sdnq import SDNQConfig
    from modules.sdnq.quantizer import apply_sdnq_to_module
    hidden, heads, mlp_dim = block_geometry["hidden"], block_geometry["heads"], block_geometry["mlp_dim"]
    block = BenchBlock(hidden, heads, mlp_dim, device=torch_device, dtype=bench_dtype)
    block.load_state_dict(master_sd)
    block.eval()
    for param in block.parameters():
        param.requires_grad_(False)
    if weights_cfg is not None:
        config = SDNQConfig(use_quantized_matmul=use_mm, add_skip_keys=False, **weights_cfg)
        block, _config = apply_sdnq_to_module(block, config, torch_dtype=bench_dtype)
    atten_kwargs = block_attention_specs[attention_spec]
    if atten_kwargs is None:
        def attention_fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elif isinstance(atten_kwargs, str):
        sage_fn = sage_attention() if atten_kwargs == "sage" else sage_attention_fp16_accum()
        if sage_fn is None:
            raise RuntimeError("sageattention unavailable")
        def attention_fn(q, k, v, fn=sage_fn):
            return fn(q, k, v, q.shape[-1] ** -0.5)
    else:
        def attention_fn(q, k, v, kw=atten_kwargs):
            return sdnq_triton_atten(q, k, v, **kw)
    block.attention_fn = attention_fn
    return block


def block_storage_bytes(block):
    total = 0
    for module in block.modules():
        if hasattr(module, "sdnq_dequantizer"):
            total += layer_storage_bytes(module)
        elif isinstance(module, torch.nn.Linear):
            total += module.weight.numel() * module.weight.element_size()
    return total


def make_quantized_linear(weight, weights_dtype, group_size=0, use_quantized_matmul=False, use_svd=False, use_hadamard=False, quantized_matmul_dtype=None, device=torch_device, **config_kwargs):
    # quantize through the same entry point model loading uses, so forward benches measure
    # the production wrapper classes and dequantizer configuration; returns the quantize wall
    # time, a one-shot measurement of what on-the-fly quantization pays per layer at load
    from modules.sdnq import SDNQConfig
    from modules.sdnq.quantizer import sdnq_quantize_layer
    out_features, in_features = weight.shape
    linear = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=bench_dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    config = SDNQConfig(weights_dtype=weights_dtype, group_size=group_size, use_quantized_matmul=use_quantized_matmul, use_svd=use_svd, use_hadamard=use_hadamard, quantized_matmul_dtype=quantized_matmul_dtype, add_skip_keys=False, **config_kwargs)
    torch_device_module.synchronize()
    started = time.perf_counter()
    layer, _config = sdnq_quantize_layer(linear, config, torch_dtype=bench_dtype, param_name="bench.weight")
    torch_device_module.synchronize()
    quant_seconds = time.perf_counter() - started
    if not hasattr(layer, "sdnq_dequantizer"):
        raise RuntimeError(f"sdnq did not quantize the layer to {weights_dtype}")
    return layer, quant_seconds


def layer_storage_bytes(layer):
    # measured storage of everything the quantized layer keeps: packed weights, scales,
    # zero points and svd factors; sizes on disk and in vram follow this, not the nominal bits
    total = 0
    for name in ("weight", "scale", "zero_point", "svd_up", "svd_down"):
        tensor = getattr(layer, name, None)
        if isinstance(tensor, (torch.Tensor, torch.nn.Parameter)):
            total += tensor.numel() * tensor.element_size()
    return total


def size_cell(size_bytes):
    return f"{size_bytes / 1024**2:6.1f}mb"


def quant_cell(quant_seconds):
    return f"{quant_seconds * 1000.0:5.0f}ms" if quant_seconds is not None else "-"


def dequant_args(layer, upcast_fp8=True):
    # mirror SDNQDequantizer.__call__'s marshaling so the direct compiled call below measures
    # the same graph the instance would compile, including the e4m3 upcast sdnq applies before
    # the compiled dequant on gpus where triton cannot convert it (pre sm_89).
    # upcast_fp8=False keeps the hardware probe honest: it must compile the raw e4m3 weight
    deq = layer.sdnq_dequantizer
    weight = layer.weight
    if upcast_fp8 and weight.dtype == torch.float8_e4m3fn and fp8_compile_gate_flag() is False:
        weight = weight.to(dtype=layer.scale.dtype)
    kwargs = dict(
        zero_point=getattr(layer, "zero_point", None),
        svd_up=getattr(layer, "svd_up", None),
        svd_down=getattr(layer, "svd_down", None),
        dtype=deq.result_dtype,
        result_shape=deq.result_shape,
        quantized_weight_shape=deq.quantized_weight_shape,
        re_quantize_for_matmul=deq.re_quantize_for_matmul or deq.is_packed,
    )
    return (deq.weights_dtype, weight, layer.scale), kwargs


def get_compiled_dequantize_weight():
    # sdnq's own dequantize_weight_compiled is a passthrough when the config has Dequantize
    # using torch.compile off; compile a tool-owned variant with the same kwargs so compiled
    # rows always measure what the webui runs with the option on. sdnq only raises the dynamo
    # recompile limits when the option is on; raise them here too or the per-dtype-per-shape
    # specializations exceed the default limit of 8 and silently fall back to eager
    global compiled_dequantize_weight # pylint: disable=global-statement
    if compiled_dequantize_weight is None:
        for limit_name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(torch._dynamo.config, limit_name): # pylint: disable=protected-access
                setattr(torch._dynamo.config, limit_name, max(8192, getattr(torch._dynamo.config, limit_name) or 0)) # pylint: disable=protected-access
        from modules.sdnq.dequantizer import dequantize_weight
        compiled_dequantize_weight = torch.compile(dequantize_weight, fullgraph=True, dynamic=False)
    return compiled_dequantize_weight


def probe_weight_dequant_compile():
    # compiled weight dequantization per fp8 storage dtype: triton before sm_89 lacks e4m3
    # conversions, so compiled dequant of raw float8_e4m3fn storage crashes on ampere while
    # e5m2 is expected to compile; the verdicts show whether sdnq's e4m3 upcast is load-bearing
    # on this gpu, so probe the raw weight rather than the upcast one the bench rows use
    result = {}
    for name in ("float8_e4m3fn", "float8_e5m2"):
        with console.status(f"probing compiled weight dequant: {name} storage (a compile failure here is expected for e4m3 on pre-ada nvidia)"):
            try:
                layer, _quant_seconds = make_quantized_linear(make_source_weight(256, 256), name)
                args, kwargs = dequant_args(layer, upcast_fp8=False)
                out = get_compiled_dequantize_weight()(*args, **kwargs)
                torch_device_module.synchronize()
                result[name] = (bool(torch.isfinite(out).all().item()), "compiles and runs")
            except Exception as e:
                result[name] = (False, f"{type(e).__name__}: {error_summary(e, 120)}")
                reset_compiled_dequant() # drop failed compile state so later rows compile fresh
    return result


def reset_compiled_dequant():
    global compiled_dequantize_weight # pylint: disable=global-statement
    compiled_dequantize_weight = None
    torch._dynamo.reset() # pylint: disable=protected-access


def run_correctness():
    checks = []
    q, k, v = make_qkv(2, 8, 512, 64, structured=False)
    checks.append(("plain", (q, k, v), {}, {}))
    checks.append(("causal", (q, k, v), dict(is_causal=True), {}))
    qg, kg, vg = make_qkv(2, 8, 512, 64, structured=False, kv_heads=2)
    checks.append(("gqa 8:2 heads", (qg, kg, vg), dict(enable_gqa=True), {}))
    bool_mask = torch.zeros(2, 1, 1, 512, device=torch_device, dtype=torch.bool)
    bool_mask[..., :384] = True
    checks.append(("bool key-padding mask", (q, k, v), dict(attn_mask=bool_mask), {}))
    float_mask = torch.zeros(2, 1, 1, 512, device=torch_device, dtype=bench_dtype)
    float_mask[..., 384:] = float("-inf")
    checks.append(("float additive mask", (q, k, v), dict(attn_mask=float_mask), {}))
    checks.append(("smooth k + hadamard", (q, k, v), {}, dict(smooth_k=True, use_hadamard=True)))
    checks.append(("smooth + hadamard + int8 pv", (q, k, v), {}, dict(smooth_k=True, use_hadamard=True, pv_matmul_dtype="int8")))
    qp, kp, vp = make_qkv(2, 8, 512, 40, structured=False)
    checks.append(("head dim 40 (padded)", (qp, kp, vp), {}, {}))
    checks.append(("head dim 40 + hadamard", (qp, kp, vp), {}, dict(use_hadamard=True)))
    qn, kn, vn = make_qkv(2, 8, 1000, 64, structured=False)
    checks.append(("non-pow2 sequence 1000", (qn, kn, vn), {}, {}))
    qx, kx, vx = make_qkv(2, 8, 512, 64, structured=False, kv_tokens=77)
    checks.append(("cross-attention 512q 77kv", (qx, kx, vx), {}, {}))
    # stress rows model the extreme-activation class (qwen-image, z-image base) where quantized
    # attention paths have produced non-finite outputs in the wild; they pass on finite output,
    # errors are informational since saturated softmax legitimately degrades accuracy
    qs, ks, vs = make_qkv(2, 8, 512, 64, structured=False)
    checks.append(("stress: 100x activations", (qs * 100.0, ks * 100.0, vs * 100.0), {}, {}))
    ko = ks.clone()
    ko[:, :, :2, :] *= 1000.0
    checks.append(("stress: 1000x outlier keys", (qs, ko, vs), {}, {}))
    checks.append(("stress: fp16 100x activations", ((qs * 100.0).to(torch.float16), (ks * 100.0).to(torch.float16), (vs * 100.0).to(torch.float16)), {}, {}))

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("code path")
    table.add_column("kernel error", justify="right")
    table.add_column("int8 error", justify="right")
    table.add_column("int8 max token", justify="right")
    table.add_column("result", justify="center")
    panel = Panel(table, title="kernel correctness", subtitle="[dim]small shapes, vs fp32 sdpa reference; stress rows pass on finite output[/dim]", box=ROUNDED_BOX, expand=False)
    failed = []
    details = {}
    # checks target kernel behavior, so the input prep runs eager via a module-global swap.
    # do not toggle torch._dynamo.config.disable for this: newer torch raises "found no
    # compiled frames" when a fullgraph-compiled function is called inside a disable window,
    # failing every check and poisoning the first compiled call afterwards
    from modules.sdnq.kernels import triton_atten as atten_module
    compiled_prep = atten_module.get_attn_inputs
    inner_prep = getattr(compiled_prep, "_torchdynamo_orig_callable", None)
    if inner_prep is not None:
        atten_module.get_attn_inputs = inner_prep
    aborted_after = None
    progress, task = live_progress()
    try:
        with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
            for index, (name, (cq, ck, cv), kwargs, sdnq_kwargs) in enumerate(checks, start=1):
                progress.reset(task, description=f"check {index}/{len(checks)}  {name}")
                try:
                    ref_kwargs = dict(kwargs)
                    if ref_kwargs.get("attn_mask", None) is not None and torch.is_floating_point(ref_kwargs["attn_mask"]):
                        ref_kwargs["attn_mask"] = ref_kwargs["attn_mask"].to(torch.float32) # stock sdpa requires the additive mask dtype to match query
                    ref = fp32_reference(cq, ck, cv, **ref_kwargs)
                    plain = sdnq_triton_atten(cq, ck, cv, do_quantize=False, **kwargs)
                    quant_kwargs = dict(matmul_dtype="auto", pv_matmul_dtype="auto", **kwargs)
                    quant_kwargs.update(sdnq_kwargs)
                    quant = sdnq_triton_atten(cq, ck, cv, **quant_kwargs)
                    err_plain = rel_err(plain, ref)
                    err_quant = rel_err(quant, ref)
                    max_err = max_token_err(quant, ref)
                    finite = bool(torch.isfinite(plain).all().item() and torch.isfinite(quant).all().item())
                    if name.startswith("stress:"):
                        ok = finite
                    else:
                        ok = finite and err_plain < 0.01 and err_quant < 0.2
                    if not ok:
                        failed.append(name)
                    details[name] = dict(kernel_err=err_plain, int8_err=err_quant, max_token_err=max_err, finite=finite, ok=ok)
                    verdict = "[green]pass[/green]" if ok else "[red]fail[/red]"
                    if not finite:
                        verdict = "[red]non-finite[/red]"
                    table.add_row(name, err_cell(err_plain), err_cell(err_quant), err_cell(max_err), verdict)
                except Exception as e:
                    failed.append(name)
                    details[name] = dict(error=error_summary(e, 200))
                    table.add_row(name, "-", "-", "-", failure_text(e))
                    if not cuda_context_alive():
                        aborted_after = name
                        break
            live.update(panel)
    finally:
        atten_module.get_attn_inputs = compiled_prep
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    if aborted_after is not None:
        emit(f"[red]the {aborted_after!r} check faulted the gpu and cuda errors are sticky: remaining checks and benchmarks cannot run in this process. rerun with --skip-checks and shapes that avoid the faulting geometry to collect the rest[/red]")
        report["correctness_aborted_after"] = aborted_after
    if failed:
        emit(f"[red]failed checks: {', '.join(failed)}[/red]")
    if "head dim 40 + hadamard" in failed and shared.opts.sdnq_attention_use_hadamard:
        emit("[yellow]current config has Use Hadamard enabled: models with non power-of-2 head dims (SD 1.5) fail at generation with it[/yellow]")
    report["correctness"] = dict(failed=failed, total=len(checks), checks=details)
    return failed


def make_prep_fn(q, k, v, attn_mask, kwargs, is_causal=False, enable_gqa=False):
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
            dropout_p=0.0, is_causal=is_causal, scale=None, enable_gqa=enable_gqa,
            smooth_k=kwargs.get("smooth_k", False), hadamard_group_size=hadamard_group_size,
            matmul_dtype=matmul_dtype, pv_matmul_dtype=kwargs.get("pv_matmul_dtype", None),
            do_quantize=do_quantize, out_dtype=None,
        )
    return prep


def bench_shape(preset, iters, warmup, position=None, config_timeout=300, fp8_result=None):
    preset_cfg = shape_presets[preset]
    batch, heads, tokens, head_dim = preset_cfg["batch"], preset_cfg["heads"], preset_cfg["tokens"], preset_cfg["head_dim"]
    kv_tokens = preset_cfg.get("kv_tokens", tokens)
    kv_heads = preset_cfg.get("kv_heads", heads)
    causal = preset_cfg.get("causal", False)
    gqa = kv_heads != heads
    description = preset_cfg["desc"]
    excluded_configs = preset_excluded_configs.get(preset, set())
    if preset == "sd15":
        emit("[yellow]sd15: hadamard configs skipped, compiling hadamard with a non pow2 head dim currently hangs torch inductor[/yellow]")
    attn_mask = None
    mask_nan_guard = False
    if preset == "masked":
        attn_mask = torch.zeros(batch, 1, 1, tokens, device=torch_device, dtype=torch.bool)
        attn_mask[..., :int(tokens * 0.75)] = True
    elif preset == "krea2":
        # the transformer's segment_mask: text is padded to a fixed 512 tokens ahead of the
        # image tokens and the padded tail is masked for queries and keys both, so padding
        # query rows are fully masked and yield nan under sdpa (the model nan_to_num's them)
        valid = torch.ones(batch, tokens, device=torch_device, dtype=torch.bool)
        valid[:, 128:512] = False
        attn_mask = valid.unsqueeze(1).unsqueeze(2) * valid.unsqueeze(1).unsqueeze(3)
        mask_nan_guard = True
    sage = sage_attention()
    sage_fp16 = sage_attention_fp16_accum()
    amd_flash = amd_triton_flash()
    selected_configs = []
    for config_id, label, kwargs in bench_configs:
        if config_id in excluded_configs:
            continue
        if config_id == "sage" and (sage is None or attn_mask is not None or head_dim not in {64, 96, 128} or kv_tokens != tokens or gqa or causal):
            continue
        if config_id == "sagefp16" and (sage_fp16 is None or attn_mask is not None or head_dim not in {64, 96, 128} or kv_tokens != tokens or gqa or causal):
            continue
        if config_id == "amdflash" and (amd_flash is None or attn_mask is not None or head_dim > 128 or gqa):
            continue
        if config_id == "fp8qk" and not (fp8_result and fp8_result["qk"][0]):
            continue
        if config_id == "fp8pv" and not (fp8_result and fp8_result["pv"][0]):
            continue
        if config_id == "fp8full" and not (fp8_result and fp8_result["qk"][0] and fp8_result["pv"][0]):
            continue
        selected_configs.append((config_id, config_label(config_id, label), kwargs))

    def make_table():
        shape_table = Table(box=box.SIMPLE_HEAVY)
        shape_table.add_column("config")
        shape_table.add_column("median time", justify="right")
        shape_table.add_column("prep", justify="right")
        shape_table.add_column("speedup", justify="right")
        shape_table.add_column("error", justify="right")
        return shape_table

    geometry = f"batch={batch} heads={heads} tokens={tokens} head_dim={head_dim}"
    if kv_tokens != tokens:
        geometry += f" kv_tokens={kv_tokens}"
    if gqa:
        geometry += f" kv_heads={kv_heads}"
    if causal:
        geometry += " causal"

    def make_panel(content):
        return Panel(content, title=f"{preset}: {geometry} {dtype_label()}", subtitle=f"[dim]{description}[/dim]", box=ROUNDED_BOX, expand=False)

    table = make_table()
    panel = make_panel(table)
    results = {}
    rows = []
    base_ms = None
    prefix = f"shape {position[0]}/{position[1]}  " if position else ""
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        progress.update(task, description=f"{prefix}{preset}: preparing inputs and fp32 reference")
        q, k, v = make_qkv(batch, heads, tokens, head_dim, kv_heads=kv_heads, kv_tokens=kv_tokens)
        scale = head_dim ** -0.5
        ref = fp32_reference(q, k, v, attn_mask=attn_mask, is_causal=causal, enable_gqa=gqa)
        if mask_nan_guard:
            ref = torch.nan_to_num(ref)
        anchor_fn = None
        for index, (config_id, label, kwargs) in enumerate(selected_configs, start=1):
            def phase(step, current_label=label, current_index=index):
                progress.update(task, description=f"{prefix}config {current_index}/{len(selected_configs)}  {current_label}: {step}")
            if config_id == "base":
                def fn(mask=attn_mask):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=gqa)
            elif config_id == "sage":
                def fn(sm=scale):
                    return sage(q, k, v, sm)
            elif config_id == "sagefp16":
                def fn(sm=scale):
                    return sage_fp16(q, k, v, sm)
            elif config_id == "amdflash":
                def fn(sm=scale):
                    return amd_flash(q, k, v, sm, is_causal=causal)
            else:
                def fn(kw=kwargs, mask=attn_mask):
                    return sdnq_triton_atten(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=gqa, **kw)
            try:
                torch._dynamo.reset() # pylint: disable=protected-access # one compiled input-prep specialization per config, matching how the webui runs a fixed config
                progress.reset(task) # restart the elapsed clock so it times the current config
                phase("compiling")
                with time_limit(config_timeout, label):
                    err = rel_err(torch.nan_to_num(fn()) if mask_nan_guard else fn(), ref)
                ms, ms_sigma = bench_stats(fn, warmup, iters, on_phase=phase)
                prep_cell = "-"
                prep_ms = None
                if kwargs is not None: # time the input prep alone; included in the median
                    try:
                        phase("timing input prep")
                        prep_ms = bench(make_prep_fn(q, k, v, attn_mask, kwargs, is_causal=causal, enable_gqa=gqa), warmup, iters)
                        prep_cell = f"[dim]{prep_ms:8.3f} ms[/dim]"
                    except Exception:
                        pass
                if base_ms is None:
                    base_ms = ms
                if config_id == "int8":
                    anchor_fn = fn
                results[config_id] = dict(ms=ms, ms_sigma=ms_sigma, err=err, prep_ms=prep_ms)
                row = (label, f"{ms:8.3f} ms", prep_cell, speedup_cell(base_ms, ms), f"{err:.5f}")
                rows.append((config_id, ms, row))
                table.add_row(*row)
            except Exception as e:
                results[config_id] = dict(ms=None, err=None, prep_ms=None, error=error_summary(e, 200))
                row = (label, "-", "-", "-", failure_text(e))
                rows.append((config_id, None, row))
                table.add_row(*row)
        # sentinel: re-time the anchor config minutes after its first measurement; the
        # delta samples run-level clock drift, which feeds the verdict noise floor
        anchor_first = (results.get("int8") or {}).get("ms")
        if anchor_fn is not None and anchor_first:
            try:
                progress.update(task, description=f"{prefix}{preset}: drift sentinel, re-timing int8")
                record_drift(anchor_first, bench(anchor_fn, warmup, iters))
            except Exception:
                pass
        # rebuild the table to star the best sdnq config when it beats the baseline
        best_id = best_config(results)
        if base_ms and best_id is not None and results[best_id]["ms"] < base_ms:
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
    if dtype_label() == "bfloat16":
        for (band_shape, band_config), (band_low, band_high) in error_sanity_bands.items():
            band_err = (results.get(band_config) or {}).get("err") if band_shape == preset else None
            if band_err and not band_low <= band_err <= band_high:
                emit(f"[yellow]measurement sanity: {preset} {band_config} error {band_err:.5f} sits outside the cross-gpu band [{band_low:.4f}, {band_high:.4f}]; treat this run's numbers with suspicion[/yellow]")
    report.setdefault("attention", {})[preset] = dict(geometry=geometry, results=results)
    return results


def bench_dequant_shape(shape_label, out_features, in_features, iters, warmup, position=None, config_timeout=300, selected_dtypes=None):
    from modules.sdnq.common import check_torch_compile
    compile_on = check_torch_compile()
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if selected_dtypes is None or dtype_id in selected_dtypes]

    def make_table():
        shape_table = Table(box=box.SIMPLE_HEAVY)
        shape_table.add_column("config")
        shape_table.add_column("quant", justify="right")
        shape_table.add_column("size", justify="right")
        shape_table.add_column("deq eager", justify="right")
        shape_table.add_column("deq compiled", justify="right")
        shape_table.add_column("weight err", justify="right")
        shape_table.add_column("fwd eager", justify="right")
        shape_table.add_column("fwd compiled", justify="right")
        shape_table.add_column("quantized mm", justify="right")
        shape_table.add_column("mm err", justify="right")
        shape_table.add_column("speedup", justify="right")
        return shape_table

    def make_panel(content):
        subtitle = f"[dim]{dequant_forward_tokens} token input, bias-free linears, errors vs fp32; quantized mm and speedup follow the current config (compile {'on' if compile_on else 'off'})[/dim]"
        return Panel(content, title=f"weight dequant: {shape_label} {dtype_label()}", subtitle=subtitle, box=ROUNDED_BOX, expand=False)

    table = make_table()
    panel = make_panel(table)
    results = {}
    notes = []
    prefix = f"shape {position[0]}/{position[1]}  " if position else ""
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        progress.update(task, description=f"{prefix}{shape_label}: preparing weights and bf16 baseline")
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        baseline = torch.nn.Linear(in_features, out_features, bias=False, device=torch_device, dtype=bench_dtype)
        with torch.no_grad():
            baseline.weight.copy_(weight)

        def baseline_fn():
            return baseline(x)

        ref_out = fp32_linear_reference(x, weight_fp32)
        base_ms, base_ms_sigma = bench_stats(baseline_fn, warmup, iters)
        base_err = rel_err(baseline_fn(), ref_out)
        base_bytes = baseline.weight.numel() * baseline.weight.element_size()
        results["bf16"] = dict(fwd_ms=base_ms, fwd_ms_sigma=base_ms_sigma, fwd_err=base_err, size_bytes=base_bytes)
        table.add_row(f"{dtype_label()} nn.Linear", "-", size_cell(base_bytes), "-", "-", "-", f"{base_ms:8.3f} ms", "-", "-", "-", "x1.00")

        for index, (dtype_id, label, cfg) in enumerate(dtype_configs, start=1):
            def phase(step, current_label=label, current_index=index):
                progress.update(task, description=f"{prefix}config {current_index}/{len(dtype_configs)}  {current_label}: {step}")
            entry = dict(eager_ms=None, compiled_ms=None, fwd_eager_ms=None, fwd_compiled_ms=None, fwd_err=None, mm_ms=None, mm_err=None, mm_dtype=None, weight_err=None, quant_s=None, size_bytes=None)
            results[dtype_id] = entry
            progress.reset(task)
            phase("quantizing")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, **cfg)
                entry["size_bytes"] = layer_storage_bytes(layer)
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(label, "-", "-", "-", "-", "-", "-", "-", "-", "-", failure_text(e))
                continue

            def eager_fn(current=layer):
                return current.sdnq_dequantizer(
                    current.weight, current.scale,
                    zero_point=getattr(current, "zero_point", None),
                    svd_up=getattr(current, "svd_up", None), svd_down=getattr(current, "svd_down", None),
                    skip_compile=True,
                )

            try:
                phase("timing eager dequant")
                eager_out = eager_fn().to(torch.float32)
                entry["eager_ms"], entry["eager_ms_sigma"] = bench_stats(eager_fn, warmup, iters)
                entry["weight_err"] = rel_err(eager_out, weight_fp32)
                eager_cell = f"{entry['eager_ms']:8.3f} ms"
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), failure_text(e), "-", "-", "-", "-", "-", "-", "-")
                del layer
                continue

            try:
                args, kwargs = dequant_args(layer)
                def compiled_fn(current_args=args, current_kwargs=kwargs):
                    return get_compiled_dequantize_weight()(*current_args, **current_kwargs)
                phase("compiling dequant")
                with time_limit(config_timeout, label):
                    compiled_out = compiled_fn()
                    torch_device_module.synchronize()
                entry["compiled_ms"], entry["compiled_ms_sigma"] = bench_stats(compiled_fn, warmup, iters, on_phase=phase)
                compiled_cell = f"{entry['compiled_ms']:8.3f} ms"
                drift = rel_err(compiled_out, eager_out)
                if drift > 1e-3:
                    notes.append(f"[yellow]{label}: compiled vs eager dequant drift {drift:.2e}[/yellow]")
                    entry["compiled_drift"] = drift
                del compiled_out
            except Exception as e:
                entry["compiled_error"] = error_summary(e, 200)
                compiled_cell = failure_text(e)
                reset_compiled_dequant()
            del eager_out

            def fwd_fn(current=layer):
                return current(x)

            # eager-mode forward: the dequantizer's __call__ resolves dequantize_weight_compiled
            # as a module global at call time, so pointing it at the eager function for the
            # bench matches the webui with Dequantize using torch.compile off. do not toggle
            # torch._dynamo.config.disable instead: code objects called during a disable window
            # keep their skip marking and never compile again in this process
            from modules.sdnq import dequantizer as dequantizer_module
            saved_compiled_fn = dequantizer_module.dequantize_weight_compiled
            try:
                phase("timing linear forward, eager dequant")
                dequantizer_module.dequantize_weight_compiled = dequantizer_module.dequantize_weight
                fwd_fn()
                torch_device_module.synchronize()
                entry["fwd_eager_ms"], entry["fwd_eager_ms_sigma"] = bench_stats(fwd_fn, warmup, iters)
                entry["fwd_err"] = rel_err(fwd_fn(), ref_out)
                fwd_eager_cell = f"{entry['fwd_eager_ms']:8.3f} ms"
            except Exception as e:
                entry["fwd_eager_error"] = error_summary(e, 200)
                fwd_eager_cell = failure_text(e)
            finally:
                dequantizer_module.dequantize_weight_compiled = saved_compiled_fn

            try:
                phase("compiling linear forward")
                with time_limit(config_timeout, label):
                    fwd_fn()
                    torch_device_module.synchronize()
                phase("timing linear forward, compiled dequant")
                entry["fwd_compiled_ms"], entry["fwd_compiled_ms_sigma"] = bench_stats(fwd_fn, warmup, iters)
                fwd_compiled_cell = f"{entry['fwd_compiled_ms']:8.3f} ms"
            except Exception as e:
                entry["fwd_compiled_error"] = error_summary(e, 200)
                fwd_compiled_cell = failure_text(e)
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            del layer

            fwd_ms = entry["fwd_compiled_ms"] if compile_on else entry["fwd_eager_ms"]
            entry["fwd_ms"] = fwd_ms # the mode the current config runs; recommendations and speedup key off it
            entry["fwd_ms_sigma"] = entry.get("fwd_compiled_ms_sigma") if compile_on else entry.get("fwd_eager_ms_sigma")
            speedup = speedup_cell(base_ms, fwd_ms) if fwd_ms else "-"

            try:
                phase("quantizing for quantized matmul")
                mm_layer, _mm_quant_seconds = make_quantized_linear(weight, use_quantized_matmul=True, **cfg)
                entry["mm_dtype"] = mm_layer.sdnq_dequantizer.quantized_matmul_dtype
                def mm_fn(current=mm_layer):
                    return current(x)
                phase("timing quantized matmul forward")
                with time_limit(config_timeout, label):
                    mm_fn()
                    torch_device_module.synchronize()
                entry["mm_ms"], entry["mm_ms_sigma"] = bench_stats(mm_fn, warmup, iters)
                entry["mm_err"] = rel_err(mm_fn(), ref_out)
                mm_cell = f"{entry['mm_ms']:8.3f} ms [dim]{entry['mm_dtype']}[/dim]"
                del mm_layer
            except Exception as e:
                entry["mm_error"] = error_summary(e, 200)
                mm_cell = failure_text(e)

            if entry["fwd_err"] and entry["weight_err"] and abs(entry["fwd_err"] - entry["weight_err"]) > 0.2 * entry["weight_err"]:
                notes.append(f"[yellow]{label}: fwd err {entry['fwd_err']:.5f} diverges from weight err {entry['weight_err']:.5f}[/yellow]")
            table.add_row(label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), eager_cell, compiled_cell, err_cell(entry["weight_err"]), fwd_eager_cell, fwd_compiled_cell, mm_cell, err_cell(entry["mm_err"]), speedup)
        # sentinel: re-time the bf16 baseline to sample run-level clock drift for this shape window
        try:
            progress.update(task, description=f"{prefix}{shape_label}: drift sentinel, re-timing the {dtype_label()} baseline")
            record_drift(base_ms, bench(baseline_fn, warmup, iters))
        except Exception:
            pass
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    for note in notes:
        emit(note)
    report.setdefault("dequant", {})[shape_label] = dict(out_features=out_features, in_features=in_features, base_ms=base_ms, results=results)
    return results


def bench_dequant_variants(shape_label, out_features, in_features, plain_results, selected_dtypes, selected_variants, iters, warmup, config_timeout=300):
    # svd and hadamard on top of the base dtypes: the quantize-time, size and error cost of
    # each option, benched through the production dequantizer path at one layer shape
    variant_configs = [(variant_id, cfg) for variant_id, cfg in dequant_variant_configs if variant_id in selected_variants]
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if dtype_id in dequant_variant_dtypes and (selected_dtypes is None or dtype_id in selected_dtypes)]
    if not variant_configs or not dtype_configs:
        return {}

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("size", justify="right")
    table.add_column("deq compiled", justify="right")
    table.add_column("weight err", justify="right")
    table.add_column("err vs plain", justify="right")
    table.add_column("fwd compiled", justify="right")
    table.add_column("speedup", justify="right")
    panel = Panel(table, title=f"svd / hadamard variants: {shape_label} {dtype_label()}", subtitle="[dim]plain rows repeated dimmed for comparison; err vs plain below x1.00 = better reconstruction[/dim]", box=ROUNDED_BOX, expand=False)

    results = {}
    base_ms = plain_results.get("bf16", {}).get("fwd_ms")
    runs = [(dtype_id, label, cfg, variant_id, variant_cfg) for dtype_id, label, cfg in dtype_configs for variant_id, variant_cfg in variant_configs]
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        added_plain = set()
        for index, (dtype_id, label, cfg, variant_id, variant_cfg) in enumerate(runs, start=1):
            row_label = f"{label} + {variant_id}"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"variant {current_index}/{len(runs)}  {current_label}: {step}")
            if dtype_id not in added_plain:
                added_plain.add(dtype_id)
                plain = plain_results.get(dtype_id) or {}
                if plain.get("weight_err") is not None:
                    plain_fwd = plain.get("fwd_compiled_ms")
                    table.add_row(
                        f"[dim]{label}[/dim]", f"[dim]{quant_cell(plain.get('quant_s'))}[/dim]", f"[dim]{size_cell(plain['size_bytes'])}[/dim]",
                        f"[dim]{plain['compiled_ms']:8.3f} ms[/dim]" if plain.get("compiled_ms") else "-",
                        f"[dim]{plain['weight_err']:.5f}[/dim]", "[dim]x1.00[/dim]",
                        f"[dim]{plain_fwd:8.3f} ms[/dim]" if plain_fwd else "-",
                        f"[dim]{speedup_cell(base_ms, plain_fwd)}[/dim]" if base_ms and plain_fwd else "-",
                    )
            entry = dict(quant_s=None, size_bytes=None, compiled_ms=None, weight_err=None, err_ratio=None, fwd_ms=None)
            results[f"{dtype_id}+{variant_id}"] = entry
            progress.reset(task)
            phase("quantizing")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, **cfg, **variant_cfg)
                entry["size_bytes"] = layer_storage_bytes(layer)
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, "-", "-", "-", "-", "-", "-", failure_text(e))
                continue

            def deq_fn(current=layer):
                return current.sdnq_dequantizer(
                    current.weight, current.scale,
                    zero_point=getattr(current, "zero_point", None),
                    svd_up=getattr(current, "svd_up", None), svd_down=getattr(current, "svd_down", None),
                )

            def fwd_fn(current=layer):
                return current(x)

            try:
                phase("compiling dequant")
                with time_limit(config_timeout, row_label):
                    deq_out = deq_fn().to(torch.float32)
                    torch_device_module.synchronize()
                entry["compiled_ms"] = bench(deq_fn, warmup, iters, on_phase=phase)
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                plain_err = (plain_results.get(dtype_id) or {}).get("weight_err")
                entry["err_ratio"] = entry["weight_err"] / plain_err if plain_err else None
                del deq_out
                phase("timing linear forward")
                with time_limit(config_timeout, row_label):
                    fwd_fn()
                    torch_device_module.synchronize()
                entry["fwd_ms"] = bench(fwd_fn, warmup, iters)
                table.add_row(
                    row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]),
                    f"{entry['compiled_ms']:8.3f} ms", err_cell(entry["weight_err"]),
                    f"x{entry['err_ratio']:.2f}" if entry["err_ratio"] else "-",
                    f"{entry['fwd_ms']:8.3f} ms",
                    speedup_cell(base_ms, entry["fwd_ms"]) if base_ms else "-",
                )
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), "-", "-", "-", "-", failure_text(e))
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            del layer
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()
    report["dequant_variants"] = dict(shape=shape_label, results=results)
    return results


def bench_float_mm_alternatives(shape_label, out_features, in_features, plain_results, selected_dtypes, iters, warmup, config_timeout=300):
    # explicit MatMul type for float storage dtypes: under enabled these route quantized matmul
    # to fp8, which not every gpu can run; measure what setting int8 or float16 costs instead
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if dtype_id in float_mm_dtypes and (selected_dtypes is None or dtype_id in selected_dtypes)]
    if not dtype_configs:
        return {}

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("mm fwd", justify="right")
    table.add_column("out err", justify="right")
    table.add_column("vs dequant", justify="right")
    panel = Panel(table, title=f"float weights, explicit MatMul type: {shape_label} {dtype_label()}", subtitle="[dim]quantized matmul with the MatMul type set explicitly; dequant path and enabled rows repeated dimmed; vs dequant above x1.00 = faster than the dequant-path forward[/dim]", box=ROUNDED_BOX, expand=False)

    results = {}
    runs = [(dtype_id, label, cfg, mm_dtype) for dtype_id, label, cfg in dtype_configs for mm_dtype in float_mm_alternative_dtypes]
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        ref_out = fp32_linear_reference(x, weight_fp32)
        added_plain = set()
        for index, (dtype_id, label, cfg, mm_dtype) in enumerate(runs, start=1):
            row_label = f"{label} + {mm_dtype} mm"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"float mm {current_index}/{len(runs)}  {current_label}: {step}")
            plain = plain_results.get(dtype_id) or {}
            fwd_ms = plain.get("fwd_ms")
            if dtype_id not in added_plain:
                added_plain.add(dtype_id)
                if fwd_ms:
                    table.add_row(f"[dim]{label} dequant path[/dim]", "-", f"[dim]{fwd_ms:8.3f} ms[/dim]", f"[dim]{plain['fwd_err']:.5f}[/dim]" if plain.get("fwd_err") else "-", "[dim]x1.00[/dim]")
                if plain.get("mm_ms"):
                    table.add_row(f"[dim]{label} + enabled ({plain.get('mm_dtype')})[/dim]", "-", f"[dim]{plain['mm_ms']:8.3f} ms[/dim]", f"[dim]{plain['mm_err']:.5f}[/dim]" if plain.get("mm_err") else "-", f"[dim]{speedup_cell(fwd_ms, plain['mm_ms'])}[/dim]" if fwd_ms else "-")
                elif plain.get("mm_error"):
                    table.add_row(f"[dim]{label} + enabled ({plain.get('mm_dtype') or 'float8_e4m3fn'})[/dim]", "-", failure_text(RuntimeError(plain["mm_error"])), "-", "-")
            entry = dict(quant_s=None, mm_ms=None, mm_err=None, mm_dtype=None)
            results[f"{dtype_id}+{mm_dtype}"] = entry
            progress.reset(task)
            phase("quantizing")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, use_quantized_matmul=True, quantized_matmul_dtype=mm_dtype, **cfg)
                entry["mm_dtype"] = layer.sdnq_dequantizer.quantized_matmul_dtype
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, "-", failure_text(e), "-", "-")
                continue

            def mm_fn(current=layer):
                return current(x)

            try:
                phase("compiling quantized matmul forward")
                with time_limit(config_timeout, row_label):
                    mm_fn()
                    torch_device_module.synchronize()
                phase("timing quantized matmul forward")
                entry["mm_ms"], entry["mm_ms_sigma"] = bench_stats(mm_fn, warmup, iters)
                entry["mm_err"] = rel_err(mm_fn(), ref_out)
                table.add_row(row_label, quant_cell(entry["quant_s"]), f"{entry['mm_ms']:8.3f} ms", err_cell(entry["mm_err"]), speedup_cell(fwd_ms, entry["mm_ms"]) if fwd_ms else "-")
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), failure_text(e), "-", "-")
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            del layer
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()
    report["dequant_float_mm"] = dict(shape=shape_label, results=results)
    return results


# === quantized matmul backends ===
#
# The linear layers bind their scaled-mm function at import
# (`from ...kernel_wrappers import int_scaled_mm_func`), so SDNQ_USE_TRITON_MM freezes the
# backend for the process and a cross-process A/B carries clock drift into the comparison.
# The call sites look the name up as a module global at call time, so rebinding it on the
# consuming module switches backends in-process; the layer forwards are compile_func'd, so
# every swap needs a dynamo reset or the traced graph keeps calling the previous function.
#
# The torch row is whatever kernel_wrappers bound when triton was not selected, captured
# rather than reimplemented. Run without SDNQ_USE_TRITON_MM=1 to have it available: where
# triton is the platform default (xpu, ipex, zluda, rdna2 and older) the torch fallbacks are
# never defined and the row needs SDNQ_USE_TRITON_MM=0.

mm_swap_targets = [
    ("modules.sdnq.layers.linear.linear_int8", "int_scaled_mm_func"),
    ("modules.sdnq.layers.linear.linear_uint8", "int_scaled_mm_func"),
    ("modules.sdnq.layers.linear.linear_fp16", "fp_scaled_mm_func"),
    ("modules.sdnq.layers.linear.linear_fp8", "fp8_scaled_mm_func"),
]


def mm_backend_bindings():
    # {backend: {(module, attr): func}} for the backends bindable in this process, plus a
    # note for any that are not
    bound = {}
    for module_path, attr in mm_swap_targets:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        func = getattr(module, attr, None)
        if func is not None:
            bound[(module_path, attr)] = func
    try:
        from modules.sdnq.kernels.triton_scaled_mm import sdnq_scaled_mm
    except Exception as e:
        return {}, {"triton": f"triton scaled mm unavailable: {error_summary(e, 120)}"}

    backends, unavailable = {}, {}
    if bound and all(func is sdnq_scaled_mm for func in bound.values()):
        unavailable["torch"] = "triton is the default matmul backend on this device; rerun with SDNQ_USE_TRITON_MM=0 to bind the torch fallbacks"
    elif bound:
        backends["torch"] = dict(bound)
    backends["triton"] = {target: sdnq_scaled_mm for target in bound}
    return backends, unavailable


def apply_mm_backend(binding):
    for (module_path, attr), func in binding.items():
        setattr(importlib.import_module(module_path), attr, func)
    torch._dynamo.reset() # pylint: disable=protected-access # layer forwards are compiled: the traced graph pins the previous function


def bench_mm_backends(shape_label, out_features, in_features, selected_dtypes, backends, iters, warmup, config_timeout=300, rounds=2):
    # paired same-run comparison of the quantized-matmul backends: one quantized layer per
    # dtype, benched through each backend in turn so both rows see the same weights and the
    # same clock state. Round order alternates so monotonic drift cancels instead of
    # accumulating into whichever backend runs second; each row keeps its fastest round.
    available, unavailable = mm_backend_bindings()
    selected = [name for name in backends if name in available]
    for name in backends:
        if name in unavailable:
            emit(f"[yellow]matmul backend '{name}' not benchable: {unavailable[name]}[/yellow]")
    if len(selected) < 2:
        if selected:
            emit(f"[yellow]matmul backend comparison needs two bindable backends, only '{selected[0]}' is available; skipping[/yellow]")
        return {}
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if selected_dtypes is None or dtype_id in selected_dtypes]
    if not dtype_configs:
        return {}
    # the comparison only runs when the torch row is bindable, which means the process came up on
    # it; restore it after the sweep so later sections measure the config the user actually runs
    original_binding = available["torch"]

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("weights")
    for name in selected:
        table.add_column(f"{name} mm", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("out err", justify="right")
    panel = Panel(
        table,
        title=f"quantized matmul backends, paired: {shape_label} {dtype_label()}",
        subtitle=f"[dim]same layer and clock state, {rounds} alternating rounds, fastest kept; delta = {selected[-1]} vs {selected[0]}, negative = {selected[-1]} faster[/dim]",
        box=ROUNDED_BOX, expand=False,
    )

    results = {}
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        ref_out = fp32_linear_reference(x, weight_fp32)
        for index, (dtype_id, label, cfg) in enumerate(dtype_configs, start=1):
            def phase(step, current_label=label, current_index=index):
                progress.update(task, description=f"mm backends {current_index}/{len(dtype_configs)}  {current_label}: {step}")
            entry = dict(mm_dtype=None, backends={})
            results[dtype_id] = entry
            try:
                phase("quantizing for quantized matmul")
                layer, _quant_seconds = make_quantized_linear(weight, use_quantized_matmul=True, **cfg)
                entry["mm_dtype"] = layer.sdnq_dequantizer.quantized_matmul_dtype
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(label, *["-"] * len(selected), "-", failure_text(e))
                continue

            def mm_fn(current=layer):
                return current(x)

            for round_index in range(rounds):
                order = selected if round_index % 2 == 0 else list(reversed(selected))
                for name in order:
                    slot = entry["backends"].setdefault(name, dict(ms=None, err=None))
                    if slot.get("error"):
                        continue
                    try:
                        phase(f"{name} backend, round {round_index + 1}/{rounds}")
                        apply_mm_backend(available[name])
                        with time_limit(config_timeout, f"{label} {name} mm"):
                            mm_fn()
                            torch_device_module.synchronize()
                        ms = bench(mm_fn, warmup, iters)
                        if slot["ms"] is None or ms < slot["ms"]:
                            slot["ms"] = ms
                        if slot["err"] is None:
                            slot["err"] = rel_err(mm_fn(), ref_out)
                    except Exception as e:
                        slot["error"] = error_summary(e, 200)
                        torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            del layer

            cells = []
            for name in selected:
                slot = entry["backends"].get(name, {})
                cells.append(f"{slot['ms']:8.3f} ms" if slot.get("ms") else failure_text(RuntimeError(slot.get("error", "not run"))))
            first, last = entry["backends"].get(selected[0], {}), entry["backends"].get(selected[-1], {})
            if first.get("ms") and last.get("ms"):
                delta = (last["ms"] - first["ms"]) / first["ms"] * 100
                colour = "green" if delta < -3 else ("red" if delta > 3 else "dim")
                entry["delta_pct"] = delta
                delta_cell = f"[{colour}]{delta:+.1f}%[/{colour}]"
            else:
                delta_cell = "-"
            errs = {slot.get("err") for slot in entry["backends"].values() if slot.get("err") is not None}
            err_text = err_cell(max(errs)) if errs else "-"
            if len(errs) > 1 and max(errs) - min(errs) > 1e-4:
                err_text += " [yellow]differs[/yellow]" # backends must be numerically equivalent; a split here is a kernel bug
            table.add_row(label, *cells, delta_cell, err_text)
            live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)

    apply_mm_backend(original_binding)
    torch_device_module.empty_cache()
    report["dequant_mm_backends"] = dict(shape=shape_label, backends=selected, rounds=rounds, results=results)
    return results


def resolved_group_label(layer, in_features):
    # infer the group size sdnq actually used from the stored scale shape: grouped scales are
    # [out, groups, 1], row-wise scales collapse the group axis
    scale = getattr(layer, "scale", None)
    if scale is None:
        return "-"
    if scale.ndim >= 3 and scale.shape[1] > 1:
        return f"g{in_features // scale.shape[1]}"
    return "row"


def bench_group_sizes(shape_label, out_features, in_features, plain_results, selected_dtypes, iters, warmup, config_timeout=300):
    # the Group size setting: 0 = auto, -1 = row-wise, explicit values snap to a divisor of
    # in_features; grouping forces a per-forward re-quantize when quantized matmul is on, so
    # the mm cells price that cost alongside the accuracy gain
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if dtype_id in group_sweep_dtypes and (selected_dtypes is None or dtype_id in selected_dtypes)]
    if not dtype_configs:
        return {}

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("size", justify="right")
    table.add_column("weight err", justify="right")
    table.add_column("fwd", justify="right")
    table.add_column("quantized mm", justify="right")
    table.add_column("mm err", justify="right")
    panel = Panel(table, title=f"group size sweep: {shape_label} {dtype_label()}", subtitle="[dim]Group size setting; auto and the mm path can resolve to different groups, resolved size shown per cell[/dim]", box=ROUNDED_BOX, expand=False)

    results = {}
    runs = [(dtype_id, label, cfg, group) for dtype_id, label, cfg in dtype_configs for group in group_sweep_values]
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        ref_out = fp32_linear_reference(x, weight_fp32)
        for index, (dtype_id, label, cfg, group) in enumerate(runs, start=1):
            group_label = {0: "auto", -1: "row"}.get(group, str(group))
            row_label = f"{label} group {group_label}"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"group {current_index}/{len(runs)}  {current_label}: {step}")
            entry = dict(quant_s=None, size_bytes=None, weight_err=None, fwd_ms=None, mm_ms=None, mm_err=None, group=group, resolved=None, mm_resolved=None)
            results[f"{dtype_id}@{group}"] = entry
            progress.reset(task)
            phase("quantizing")
            base_cfg = {key: value for key, value in cfg.items() if key != "group_size"}
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, group_size=group, **base_cfg)
                entry["size_bytes"] = layer_storage_bytes(layer)
                entry["resolved"] = resolved_group_label(layer, in_features)
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, "-", "-", failure_text(e), "-", "-", "-")
                continue

            def fwd_fn(current=layer):
                return current(x)

            try:
                phase("timing dequant and forward")
                with time_limit(config_timeout, row_label):
                    deq_out = layer.sdnq_dequantizer(
                        layer.weight, layer.scale,
                        zero_point=getattr(layer, "zero_point", None),
                        svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                    ).to(torch.float32)
                    fwd_fn()
                    torch_device_module.synchronize()
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                del deq_out
                entry["fwd_ms"] = bench(fwd_fn, warmup, iters)
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), "-", failure_text(e), "-", "-")
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
                del layer
                continue
            del layer

            mm_cell, mm_err_cell = "-", "-"
            try:
                phase("quantizing for quantized matmul")
                mm_layer, _mm_quant_seconds = make_quantized_linear(weight, group_size=group, use_quantized_matmul=True, **base_cfg)
                entry["mm_resolved"] = resolved_group_label(mm_layer, in_features)
                def mm_fn(current=mm_layer):
                    return current(x)
                phase("timing quantized matmul forward")
                with time_limit(config_timeout, row_label):
                    mm_fn()
                    torch_device_module.synchronize()
                entry["mm_ms"], entry["mm_ms_sigma"] = bench_stats(mm_fn, warmup, iters)
                entry["mm_err"] = rel_err(mm_fn(), ref_out)
                mm_suffix = f" [dim]{entry['mm_resolved']}[/dim]" if entry["mm_resolved"] != entry["resolved"] else ""
                mm_cell = f"{entry['mm_ms']:8.3f} ms{mm_suffix}"
                mm_err_cell = err_cell(entry["mm_err"])
                del mm_layer
            except Exception as e:
                entry["mm_error"] = error_summary(e, 200)
                mm_cell = failure_text(e)
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state

            shown_label = f"{row_label} [dim]({entry['resolved']})[/dim]" if group == 0 else row_label
            table.add_row(shown_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), err_cell(entry["weight_err"]), f"{entry['fwd_ms']:8.3f} ms", mm_cell, mm_err_cell)
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()
    report["dequant_group_sizes"] = dict(shape=shape_label, results=results)
    return results


def bench_svd_ranks(shape_label, out_features, in_features, plain_results, selected_dtypes, iters, warmup, config_timeout=300):
    # the SVD rank size setting: rank drives both outlier absorption and the rank x (in + out)
    # fp16 size overhead; plain no-svd rows repeated dimmed for comparison
    dtype_configs = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if dtype_id in svd_rank_sweep_dtypes and (selected_dtypes is None or dtype_id in selected_dtypes)]
    if not dtype_configs:
        return {}

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("size", justify="right")
    table.add_column("weight err", justify="right")
    table.add_column("err vs plain", justify="right")
    table.add_column("fwd", justify="right")
    panel = Panel(table, title=f"svd rank sweep: {shape_label} {dtype_label()}", subtitle="[dim]SVD rank size setting at svd steps 8; err vs plain below x1.00 = better reconstruction[/dim]", box=ROUNDED_BOX, expand=False)

    results = {}
    runs = [(dtype_id, label, cfg, rank) for dtype_id, label, cfg in dtype_configs for rank in svd_rank_sweep_values]
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        added_plain = set()
        for index, (dtype_id, label, cfg, rank) in enumerate(runs, start=1):
            row_label = f"{label} + svd rank {rank}"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"svd {current_index}/{len(runs)}  {current_label}: {step}")
            plain = plain_results.get(dtype_id) or {}
            if dtype_id not in added_plain:
                added_plain.add(dtype_id)
                if plain.get("weight_err") is not None:
                    plain_fwd = plain.get("fwd_ms")
                    table.add_row(f"[dim]{label}[/dim]", f"[dim]{quant_cell(plain.get('quant_s'))}[/dim]", f"[dim]{size_cell(plain['size_bytes'])}[/dim]", f"[dim]{plain['weight_err']:.5f}[/dim]", "[dim]x1.00[/dim]", f"[dim]{plain_fwd:8.3f} ms[/dim]" if plain_fwd else "-")
            entry = dict(quant_s=None, size_bytes=None, weight_err=None, err_ratio=None, fwd_ms=None, rank=rank)
            results[f"{dtype_id}@{rank}"] = entry
            progress.reset(task)
            phase("quantizing (svd)")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, use_svd=True, svd_rank=rank, **cfg)
                entry["size_bytes"] = layer_storage_bytes(layer)
                def fwd_fn(current=layer):
                    return current(x)
                phase("timing dequant and forward")
                with time_limit(config_timeout, row_label):
                    deq_out = layer.sdnq_dequantizer(
                        layer.weight, layer.scale,
                        zero_point=getattr(layer, "zero_point", None),
                        svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                    ).to(torch.float32)
                    fwd_fn()
                    torch_device_module.synchronize()
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                del deq_out
                plain_err = plain.get("weight_err")
                entry["err_ratio"] = entry["weight_err"] / plain_err if plain_err else None
                entry["fwd_ms"] = bench(fwd_fn, warmup, iters)
                table.add_row(row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), err_cell(entry["weight_err"]), f"x{entry['err_ratio']:.2f}" if entry["err_ratio"] else "-", f"{entry['fwd_ms']:8.3f} ms")
                del layer
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), "-", "-", "-", failure_text(e))
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()
    report["dequant_svd_ranks"] = dict(shape=shape_label, results=results)
    return results


def bench_hadamard_groups(shape_label, out_features, in_features, plain_results, selected_dtypes, iters, warmup, config_timeout=300):
    # the weight-side Hadamard group size setting, swept on int8 where rotation measured the
    # largest gain; unlike the attention slider it is not clamped to head dim
    if selected_dtypes is not None and "int8" not in selected_dtypes:
        return {}

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("size", justify="right")
    table.add_column("weight err", justify="right")
    table.add_column("err vs plain", justify="right")
    table.add_column("fwd", justify="right")
    panel = Panel(table, title=f"hadamard group size sweep: int8 {shape_label} {dtype_label()}", subtitle="[dim]weight-side Hadamard group size setting; err vs plain below x1.00 = better reconstruction[/dim]", box=ROUNDED_BOX, expand=False)

    results = {}
    plain = plain_results.get("int8") or {}
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        if plain.get("weight_err") is not None:
            plain_fwd = plain.get("fwd_ms")
            table.add_row("[dim]int8[/dim]", f"[dim]{quant_cell(plain.get('quant_s'))}[/dim]", f"[dim]{size_cell(plain['size_bytes'])}[/dim]", f"[dim]{plain['weight_err']:.5f}[/dim]", "[dim]x1.00[/dim]", f"[dim]{plain_fwd:8.3f} ms[/dim]" if plain_fwd else "-")
        for index, hgroup in enumerate(hadamard_group_values, start=1):
            row_label = f"int8 + hadamard group {hgroup}"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"hadamard group {current_index}/{len(hadamard_group_values)}  {current_label}: {step}")
            entry = dict(quant_s=None, size_bytes=None, weight_err=None, err_ratio=None, fwd_ms=None, hgroup=hgroup)
            results[str(hgroup)] = entry
            progress.reset(task)
            phase("quantizing (hadamard)")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, "int8", use_hadamard=True, hadamard_group_size=hgroup)
                entry["size_bytes"] = layer_storage_bytes(layer)
                def fwd_fn(current=layer):
                    return current(x)
                phase("timing dequant and forward")
                with time_limit(config_timeout, row_label):
                    deq_out = layer.sdnq_dequantizer(
                        layer.weight, layer.scale,
                        zero_point=getattr(layer, "zero_point", None),
                        svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                    ).to(torch.float32)
                    fwd_fn()
                    torch_device_module.synchronize()
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                del deq_out
                plain_err = plain.get("weight_err")
                entry["err_ratio"] = entry["weight_err"] / plain_err if plain_err else None
                entry["fwd_ms"] = bench(fwd_fn, warmup, iters)
                table.add_row(row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), err_cell(entry["weight_err"]), f"x{entry['err_ratio']:.2f}" if entry["err_ratio"] else "-", f"{entry['fwd_ms']:8.3f} ms")
                del layer
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), "-", "-", "-", failure_text(e))
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()
    report["dequant_hadamard_groups"] = dict(shape=shape_label, results=results)
    return results


def bench_quant_toggles(shape_label, out_features, in_features, plain_results, selected_dtypes, iters, warmup, config_timeout=300):
    # remaining checkbox-level settings: Dequantize using full precision off (scales kept in
    # the model dtype instead of fp32), Dynamic quantization (per-layer dtype escalation until
    # the loss threshold passes), and Quantize using GPU (quantization wall time on cpu)
    fp32_dtypes = [(dtype_id, label, cfg) for dtype_id, label, cfg in dequant_dtype_configs if dtype_id in toggle_dtypes and (selected_dtypes is None or dtype_id in selected_dtypes)]
    results = dict(fp32_off={}, dynamic={}, cpu_quant_s=None, gpu_quant_s=None)

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("quant", justify="right")
    table.add_column("size", justify="right")
    table.add_column("weight err", justify="right")
    table.add_column("fwd", justify="right")
    panel = Panel(table, title=f"dequantize full precision off: {shape_label} {dtype_label()}", subtitle="[dim]Dequantize using full precision unchecked; full-precision rows repeated dimmed[/dim]", box=ROUNDED_BOX, expand=False)
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        generator = torch.Generator(device=torch_device).manual_seed(42)
        x = torch.randn(dequant_forward_tokens, in_features, device=torch_device, dtype=bench_dtype, generator=generator)
        for index, (dtype_id, label, cfg) in enumerate(fp32_dtypes, start=1):
            row_label = f"{label} + fp32 dequant off"
            def phase(step, current_label=row_label, current_index=index):
                progress.update(task, description=f"toggle {current_index}/{len(fp32_dtypes)}  {current_label}: {step}")
            plain = plain_results.get(dtype_id) or {}
            if plain.get("weight_err") is not None:
                plain_fwd = plain.get("fwd_ms")
                table.add_row(f"[dim]{label}[/dim]", f"[dim]{quant_cell(plain.get('quant_s'))}[/dim]", f"[dim]{size_cell(plain['size_bytes'])}[/dim]", f"[dim]{plain['weight_err']:.5f}[/dim]", f"[dim]{plain_fwd:8.3f} ms[/dim]" if plain_fwd else "-")
            entry = dict(quant_s=None, size_bytes=None, weight_err=None, fwd_ms=None)
            results["fp32_off"][dtype_id] = entry
            progress.reset(task)
            phase("quantizing")
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, dequantize_fp32=False, **cfg)
                entry["size_bytes"] = layer_storage_bytes(layer)
                def fwd_fn(current=layer):
                    return current(x)
                phase("timing dequant and forward")
                with time_limit(config_timeout, row_label):
                    deq_out = layer.sdnq_dequantizer(
                        layer.weight, layer.scale,
                        zero_point=getattr(layer, "zero_point", None),
                        svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                    ).to(torch.float32)
                    fwd_fn()
                    torch_device_module.synchronize()
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                del deq_out
                entry["fwd_ms"] = bench(fwd_fn, warmup, iters)
                table.add_row(row_label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), err_cell(entry["weight_err"]), f"{entry['fwd_ms']:8.3f} ms")
                del layer
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(row_label, quant_cell(entry["quant_s"]), "-", "-", failure_text(e))
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)

    dyn_table = Table(box=box.SIMPLE_HEAVY)
    dyn_table.add_column("requested")
    dyn_table.add_column("chosen", justify="center")
    dyn_table.add_column("quant", justify="right")
    dyn_table.add_column("size", justify="right")
    dyn_table.add_column("weight err", justify="right")
    dyn_panel = Panel(dyn_table, title=f"dynamic quantization: {shape_label} {dtype_label()}", subtitle="[dim]Use Dynamic quantization: escalates to wider dtypes until normalized mse passes the loss threshold (default 10^-(bits/2))[/dim]", box=ROUNDED_BOX, expand=False)
    progress, task = live_progress()
    with Live(Group(dyn_panel, progress), console=console, refresh_per_second=4) as live:
        weight = make_source_weight(out_features, in_features)
        weight_fp32 = weight.to(torch.float32)
        for index, requested in enumerate(dynamic_quant_requests, start=1):
            progress.reset(task)
            progress.update(task, description=f"dynamic {index}/{len(dynamic_quant_requests)}  requested {requested}")
            entry = dict(quant_s=None, chosen=None, size_bytes=None, weight_err=None)
            results["dynamic"][requested] = entry
            try:
                layer, entry["quant_s"] = make_quantized_linear(weight, requested, use_dynamic_quantization=True)
                entry["chosen"] = layer.sdnq_dequantizer.weights_dtype
                entry["size_bytes"] = layer_storage_bytes(layer)
                deq_out = layer.sdnq_dequantizer(
                    layer.weight, layer.scale,
                    zero_point=getattr(layer, "zero_point", None),
                    svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                    skip_compile=True,
                ).to(torch.float32)
                entry["weight_err"] = rel_err(deq_out, weight_fp32)
                del deq_out, layer
                chosen_cell = entry["chosen"] if entry["chosen"] == requested else f"[yellow]{entry['chosen']}[/yellow]"
                dyn_table.add_row(requested, chosen_cell, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), err_cell(entry["weight_err"]))
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                dyn_table.add_row(requested, "[red]unquantized[/red]", "-", "-", failure_text(e))
        live.update(dyn_panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(dyn_panel)

    # Quantize using GPU: one int8 layer quantized on cpu for the wall-time comparison
    try:
        weight_cpu = make_source_weight(out_features, in_features).cpu()
        _cpu_layer, results["cpu_quant_s"] = make_quantized_linear(weight_cpu, "int8", device="cpu")
        del _cpu_layer, weight_cpu
        results["gpu_quant_s"] = (plain_results.get("int8") or {}).get("quant_s")
    except Exception as e:
        results["cpu_quant_error"] = error_summary(e, 200)
    torch_device_module.empty_cache()
    report["dequant_toggles"] = dict(shape=shape_label, results=results)
    return results


def make_source_conv_weight(out_channels, in_channels, kernel, seed=1234):
    # conv analogue of make_source_weight: a few high-magnitude input channels
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    weight = torch.randn(out_channels, in_channels, kernel, kernel, device=torch_device, dtype=bench_dtype, generator=generator) * 0.02
    weight[:, [1, in_channels // 2, in_channels - 2]] *= 8.0
    return weight


def fp32_conv_reference(x, weight_fp32, padding):
    # convs route through cudnn, which has its own tf32 switch on top of the matmul one
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return torch.nn.functional.conv2d(x.to(torch.float32), weight_fp32, padding=padding)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn


def make_quantized_conv(weight, weights_dtype, use_quantized_matmul=False):
    from modules.sdnq import SDNQConfig
    from modules.sdnq.quantizer import sdnq_quantize_layer
    out_channels, in_channels, kh, kw = weight.shape
    conv = torch.nn.Conv2d(in_channels, out_channels, (kh, kw), padding=(kh // 2, kw // 2), bias=False, device=torch_device, dtype=bench_dtype)
    with torch.no_grad():
        conv.weight.copy_(weight)
    config = SDNQConfig(weights_dtype=weights_dtype, quant_conv=True, use_quantized_matmul_conv=use_quantized_matmul, add_skip_keys=False)
    torch_device_module.synchronize()
    started = time.perf_counter()
    layer, _config = sdnq_quantize_layer(conv, config, torch_dtype=bench_dtype, param_name="bench.weight")
    torch_device_module.synchronize()
    quant_seconds = time.perf_counter() - started
    if not hasattr(layer, "sdnq_dequantizer"):
        raise RuntimeError(f"sdnq did not quantize the conv layer to {weights_dtype}")
    return layer, quant_seconds


def bench_conv_section(iters, warmup, config_timeout=300):
    # the Quantize convolutional layers and Use quantized MatMul with conv settings, measured
    # on real Conv2d layers; dit models have no convs, this is for unet and vae model classes
    all_results = {}
    for shape_label, out_channels, in_channels, kernel, px in conv_shapes:
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("config")
        table.add_column("quant", justify="right")
        table.add_column("size", justify="right")
        table.add_column("weight err", justify="right")
        table.add_column("fwd", justify="right")
        table.add_column("out err", justify="right")
        table.add_column("speedup", justify="right")
        panel = Panel(table, title=f"conv quantization: {shape_label} {dtype_label()}", subtitle="[dim]bias-free conv2d, batch 1, errors vs true fp32 (tf32 off); mm rows use the conv quantized matmul path[/dim]", box=ROUNDED_BOX, expand=False)
        results = {}
        base_ms = None
        progress, task = live_progress()
        with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
            progress.update(task, description=f"{shape_label}: preparing weights and fp32 reference")
            weight = make_source_conv_weight(out_channels, in_channels, kernel)
            weight_fp32 = weight.to(torch.float32)
            generator = torch.Generator(device=torch_device).manual_seed(42)
            x = torch.randn(1, in_channels, px, px, device=torch_device, dtype=bench_dtype, generator=generator)
            ref_out = fp32_conv_reference(x, weight_fp32, kernel // 2)
            for index, (config_id, weights_cfg, use_mm) in enumerate(conv_configs, start=1):
                label = f"{dtype_label()} conv2d" if weights_cfg is None else f"{weights_cfg['weights_dtype']}{' + conv mm' if use_mm else ''}"
                def phase(step, current_label=label, current_index=index, current_progress=progress, current_task=task, current_shape=shape_label):
                    current_progress.update(current_task, description=f"conv {current_index}/{len(conv_configs)}  {current_shape} {current_label}: {step}")
                entry = dict(quant_s=None, size_bytes=None, weight_err=None, fwd_ms=None, out_err=None)
                results[config_id] = entry
                progress.reset(task)
                try:
                    if weights_cfg is None:
                        layer = torch.nn.Conv2d(in_channels, out_channels, (kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False, device=torch_device, dtype=bench_dtype)
                        with torch.no_grad():
                            layer.weight.copy_(weight)
                        entry["size_bytes"] = layer.weight.numel() * layer.weight.element_size()
                        weight_err_cell = "-"
                    else:
                        phase("quantizing")
                        layer, entry["quant_s"] = make_quantized_conv(weight, use_quantized_matmul=use_mm, **weights_cfg)
                        entry["size_bytes"] = layer_storage_bytes(layer)
                        deq_out = layer.sdnq_dequantizer(
                            layer.weight, layer.scale,
                            zero_point=getattr(layer, "zero_point", None),
                            svd_up=getattr(layer, "svd_up", None), svd_down=getattr(layer, "svd_down", None),
                            skip_quantized_matmul=use_mm, skip_compile=True,
                        ).to(torch.float32)
                        entry["weight_err"] = rel_err(deq_out, weight_fp32)
                        weight_err_cell = err_cell(entry["weight_err"])
                        del deq_out
                    def fwd_fn(current=layer, current_x=x):
                        return current(current_x)
                    phase("timing conv forward")
                    with time_limit(config_timeout, label):
                        fwd_fn()
                        torch_device_module.synchronize()
                    entry["fwd_ms"], entry["fwd_ms_sigma"] = bench_stats(fwd_fn, warmup, iters)
                    entry["out_err"] = rel_err(fwd_fn(), ref_out)
                    if base_ms is None:
                        base_ms = entry["fwd_ms"]
                    table.add_row(label, quant_cell(entry["quant_s"]), size_cell(entry["size_bytes"]), weight_err_cell, f"{entry['fwd_ms']:8.3f} ms", err_cell(entry["out_err"]), speedup_cell(base_ms, entry["fwd_ms"]))
                    del layer
                except Exception as e:
                    entry["error"] = error_summary(e, 200)
                    table.add_row(label, quant_cell(entry["quant_s"]), "-", "-", failure_text(e), "-", "-")
                    torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
            live.update(panel)
        console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
        transcript.append(panel)
        torch_device_module.empty_cache()
        all_results[shape_label] = results
    report["dequant_conv"] = all_results
    return all_results


def block_label(weights_cfg, use_mm, attention_spec):
    if weights_cfg is None:
        weights_part = dtype_label()
    else:
        weights_part = weights_cfg["weights_dtype"] + (" g16" if weights_cfg.get("group_size") == 16 else "")
        if use_mm:
            weights_part += " mm"
    return f"{weights_part} + {attention_spec}"


def bench_block_section(iters, warmup, config_timeout=300, selected=None):
    global block_geometry # pylint: disable=global-statement
    all_results = {}
    for family, geometry in block_geometries.items():
        block_geometry = geometry
        results = bench_block_geometry(iters, warmup, config_timeout=config_timeout, selected=selected)
        all_results[family] = results
        report.setdefault("blocks", {})[family] = dict(geometry=dict(geometry), results=results)
    # the first family also lands at the flat block key, which replays and the buyback
    # veto fall back to when no family matches the reference shape
    primary = next(iter(block_geometries))
    report["block"] = report["blocks"][primary]
    return all_results.get(primary, {})


def bench_block_geometry(iters, warmup, config_timeout=300, selected=None):
    # each row is a complete configuration measured end to end through a dit block, because
    # component speedups and errors do not compose multiplicatively
    configs = [c for c in block_configs if selected is None or c[0] in selected]
    sage_missing = {"sage": sage_attention() is None, "sage fp16 accum": sage_attention_fp16_accum() is None}
    skipped = [config_id for config_id, _w, _mm, spec in configs if sage_missing.get(spec, False)]
    if skipped:
        configs = [c for c in configs if c[0] not in skipped]
        emit(f"[dim]block: skipping {', '.join(skipped)}, sageattention (or this accumulation mode) is unavailable here[/dim]")
    if not configs:
        return {}
    hidden, heads, mlp_dim, tokens = block_geometry["hidden"], block_geometry["heads"], block_geometry["mlp_dim"], block_geometry["tokens"]

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("config")
    table.add_column("size", justify="right")
    table.add_column("block ms", justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("out err", justify="right")
    table.add_column("max tok err", justify="right")
    table.add_column("err x4 blocks", justify="right")
    panel = Panel(table, title=f"combined block: hidden={hidden} heads={heads} mlp={mlp_dim} tokens={tokens} {dtype_label()}", subtitle="[dim]dit block, fused qkv + gelu mlp with residuals; err vs an fp32 reference block, max tok = worst single token, x4 = four stacked blocks[/dim]", box=ROUNDED_BOX, expand=False)

    def run_depth(block, x0, depth):
        h = x0
        for _ in range(depth):
            h = block(h)
        return h

    results = {}
    base_ms = None
    progress, task = live_progress()
    with Live(Group(panel, progress), console=console, refresh_per_second=4) as live:
        progress.update(task, description="block: building master weights and the fp32 reference")
        master = make_block_master()
        generator = torch.Generator(device=torch_device).manual_seed(7)
        x = torch.randn(1, tokens, hidden, device=torch_device, dtype=bench_dtype, generator=generator)
        ref_block = BenchBlock(hidden, heads, mlp_dim, device=torch_device, dtype=torch.float32)
        ref_block.load_state_dict(master)
        ref_block.eval()
        def ref_attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        ref_block.attention_fn = ref_attention
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad():
                x_fp32 = x.to(torch.float32)
                ref_out = ref_block(x_fp32)
                ref_out4 = run_depth(ref_block, x_fp32, 4)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
            torch.backends.cudnn.allow_tf32 = tf32_cudnn
        del ref_block, x_fp32
        torch_device_module.empty_cache()

        for index, (config_id, weights_cfg, use_mm, attention_spec) in enumerate(configs, start=1):
            label = block_label(weights_cfg, use_mm, attention_spec)
            def phase(step, current_label=label, current_index=index):
                progress.update(task, description=f"block config {current_index}/{len(configs)}  {current_label}: {step}")
            entry = dict(ms=None, err=None, max_err=None, err4=None, size_bytes=None, label=label)
            results[config_id] = entry
            progress.reset(task)
            phase("quantizing block")
            try:
                block = build_bench_block(master, weights_cfg, use_mm, attention_spec)
                entry["size_bytes"] = block_storage_bytes(block)

                def fn(current=block):
                    with torch.no_grad():
                        return current(x)

                phase("compiling")
                with time_limit(config_timeout, label):
                    out = fn()
                    torch_device_module.synchronize()
                entry["ms"], entry["ms_sigma"] = bench_stats(fn, warmup, iters, on_phase=phase)
                entry["err"] = rel_err(out, ref_out)
                entry["max_err"] = max_token_err(out, ref_out)
                phase("measuring depth-4 error")
                with torch.no_grad():
                    entry["err4"] = rel_err(run_depth(block, x, 4), ref_out4)
                del block, out
                if base_ms is None:
                    base_ms = entry["ms"]
                table.add_row(label, size_cell(entry["size_bytes"]), f"{entry['ms']:8.3f} ms", speedup_cell(base_ms, entry["ms"]), err_cell(entry["err"]), err_cell(entry["max_err"]), err_cell(entry["err4"]))
            except Exception as e:
                entry["error"] = error_summary(e, 200)
                table.add_row(label, size_cell(entry["size_bytes"]) if entry["size_bytes"] else "-", "-", "-", "-", "-", failure_text(e))
                torch._dynamo.reset() # pylint: disable=protected-access # drop failed compile state
        live.update(panel)
    console.line() # separate sections; Live's final frame also lacks a trailing newline when output is piped
    transcript.append(panel)
    torch_device_module.empty_cache()

    notes = []
    sized = [(entry["label"], entry["ms"], entry["err"]) for entry in results.values() if entry.get("ms") and entry.get("err") is not None and math.isfinite(entry["err"])]
    frontier = sorted((c for c in sized if not any(o is not c and o[1] <= c[1] and o[2] <= c[2] for o in sized)), key=lambda c: c[1])
    if len(frontier) > 1:
        notes.append("speed/error frontier: " + ", ".join(f"{label} {ms:.2f}ms err {err:.4f}" for label, ms, err in frontier))
    growth = [entry["err4"] / entry["err"] for entry in results.values() if entry.get("err") and entry.get("err4") and math.isfinite(entry["err4"] / entry["err"])]
    if growth:
        notes.append(f"four stacked blocks: error grows x{sum(growth) / len(growth):.2f} avg, residuals dampen compounding")
    weights_mode = str(getattr(shared.opts, "sdnq_quantize_weights_mode", ""))
    current_id = None
    if weights_mode == "int8" and getattr(shared.opts, "sdnq_quantize_matmul_mode", "disabled") != "disabled":
        current_id = "int8-mm-atten" if "SDNQ attention" in shared.opts.sdp_overrides else "int8-mm"
    if current_id and results.get(current_id, {}).get("ms"):
        notes.append(f"current config runs the {results[current_id]['label']} row for int8-quantized models")
    if any(entry.get("ms") for config_id, entry in results.items() if config_id.endswith("sagefp16")):
        notes.append("sage fp16 accum: benchmark-only, no sdnext setting reaches it (sage pins fp32 accum on sm86)")
    if notes:
        emit(Panel("\n".join(notes), title="block notes", box=ROUNDED_BOX))
    return results


def measured(results, config_id):
    entry = results.get(config_id) or {}
    ms = entry.get("ms")
    return (ms, entry.get("err")) if ms is not None else (None, None)


def best_config(results):
    # lowest error among rows within 5% of the fastest sdnq time
    candidates = [(config_id, entry["ms"], entry["err"]) for config_id, entry in results.items() if entry.get("ms") is not None and config_id not in external_config_ids]
    if not candidates:
        return None
    fastest = min(ms for _config_id, ms, _err in candidates)
    near_fastest = [(config_id, ms, err) for config_id, ms, err in candidates if ms <= fastest * 1.05]
    return min(near_fastest, key=lambda item: item[2])[0]


def build_recommendations(all_results, fp8_result, prep_status, block_results=None, block_variants=None):
    # prefer an image dit shape with the full config set as reference, then the video shapes
    reference = None
    for preset in recommendation_presets:
        if preset in all_results and measured(all_results[preset], "int8")[0] is not None:
            reference = preset
            break
    if reference is None:
        emit(f"[yellow]attention recommendations need a successful int8 run at a self-attention reference shape ({', '.join(recommendation_presets)}); none was benchmarked[/yellow]")
        return
    results = all_results[reference]
    base_ms, _base_err = measured(results, "base")
    noquant_ms, noquant_err = measured(results, "noquant")
    int8_ms, int8_err = measured(results, "int8")

    table = Table(title=f"recommended settings (Compute Settings -> SDNQ Attention), measured at the {reference} shape", box=ROUNDED_BOX)
    table.add_column("setting")
    table.add_column("current", justify="center")
    table.add_column("recommended", justify="center")
    table.add_column("reason")

    def current(key):
        return str(getattr(shared.opts, key))

    rows = []
    # verdicts are computed before any row is built: the smooth k and hadamard buybacks
    # feed a composition check on the qk verdict, since toggles that pass individually
    # can still lose to unquantized as a stack. compare against the unquantized sdnq row:
    # quantization has to pay for its own prep and clear the shared speed margin, at this
    # run's own measured noise level; too close to call keeps the current setting
    qk_sigma = pair_sigma(results.get("int8") or {}, "ms", results.get("noquant") or {}, "ms")
    qk_test = speed_verdict(int8_ms, noquant_ms, sigma=qk_sigma)
    use_quantized = qk_test == "faster"
    qk_inconclusive = qk_test == "inconclusive"
    if int8_ms and noquant_ms:
        if use_quantized:
            quant_reason = f"int8 qk measured x{noquant_ms / int8_ms:.2f} vs unquantized sdnq attention"
            if int8_err and noquant_err:
                quant_reason += f", error {int8_err:.5f} vs {noquant_err:.5f}; smooth k and hadamard below buy error back"
        elif qk_inconclusive:
            quant_reason = f"int8 qk measured x{noquant_ms / int8_ms:.2f} vs unquantized sdnq attention, too close to the x{1 / recommend_speed_margin:.2f} margin to call at this run's noise (ratio sigma {qk_sigma:.1%}); keeping the current setting"
        elif int8_ms < noquant_ms:
            quant_reason = f"int8 qk measured only x{noquant_ms / int8_ms:.2f} vs unquantized sdnq attention, under the x{1 / recommend_speed_margin:.2f} margin the verdict requires"
            if int8_err and noquant_err:
                quant_reason += f"; unquantized keeps error {noquant_err:.5f} vs {int8_err:.5f}"
        else:
            quant_reason = f"unquantized sdnq measured x{int8_ms / noquant_ms:.2f} vs int8 qk with lower error; quantization prep outweighs the kernel gain on this gpu"
    elif int8_ms and base_ms:
        use_quantized = base_ms / int8_ms >= 1.10
        quant_reason = f"int8 qk measured x{base_ms / int8_ms:.2f} vs torch sdpa; unquantized sdnq row unavailable"
    else:
        use_quantized = False
        quant_reason = "int8 qk failed to run"

    # buyback costs are judged at the block geometry matching the reference family when
    # one was measured, so a krea2 verdict uses the krea2-width block
    block_rows = block_results or {}
    block_geo = (report.get("block") or {}).get("geometry") or {}
    if block_variants:
        block_family = reference if reference in block_variants else next(iter(block_variants))
        variant = block_variants.get(block_family) or {}
        if variant.get("results"):
            block_rows = variant["results"]
            block_geo = variant.get("geometry") or {}

    def block_ms(config_id):
        return (block_rows.get(config_id) or {}).get("ms")

    def buyback_cost(kernel_config_id, block_config_id):
        # cost of a prep toggle, judged at the most representative measured scope: the
        # kernel rows hand the prep contiguous q/k/v, real models hand it strided views
        # from the fused qkv projection, which the block section measures
        toggle_ms, toggle_err = measured(results, kernel_config_id)
        if not (toggle_ms and int8_ms and int8_err and toggle_err):
            return None, None, None
        gain = int8_err / toggle_err if toggle_err > 0 else 1.0
        kernel_cost = toggle_ms / int8_ms - 1.0
        block_base_ms, block_toggle_ms = block_ms("int8-mm-atten"), block_ms(block_config_id)
        if block_base_ms and block_toggle_ms:
            cost = block_toggle_ms / block_base_ms - 1.0
            geo = f" ({block_geo['hidden']}-wide, hd{block_geo['hidden'] // block_geo['heads']})" if block_geo.get("hidden") and block_geo.get("heads") else ""
            return gain, cost, f"{cost:+.0%} at block scope{geo} with strided qkv ({kernel_cost:+.0%} on contiguous kernel tensors)"
        return gain, kernel_cost, f"{kernel_cost:+.0%} time"

    smooth_gain, smooth_cost, smooth_cost_note = buyback_cost("smooth", "int8-mm-smooth")
    smooth_rec = smooth_reason = None
    if smooth_gain is not None:
        smooth_rec = smooth_gain >= 1.3 and smooth_cost <= 0.25 # attention-level cost is a few percent end to end
        smooth_reason = f"int8 error x{smooth_gain:.1f} lower for {smooth_cost_note}"

    hadamard_gain, hadamard_cost, hadamard_cost_note = buyback_cost("hadamard", "int8-mm-hadamard")
    hadamard_rec = hadamard_reason = None
    if hadamard_gain is not None:
        hadamard_rec = hadamard_gain >= 1.3 and hadamard_cost <= 0.15
        hadamard_reason = f"int8 error x{hadamard_gain:.1f} lower for {hadamard_cost_note}; hangs torch compile on non pow2 head dims (SD 1.5)"

    # composition check: the toggles recommended above must still beat unquantized as a
    # stack; when the exact stack was not measured, predict it additively from the single
    # toggle deltas (measured cross-vendor: additive to ~2% median with a +2% skew, so the
    # prediction carries that correction plus a model sigma alongside the measurement noise)
    additive_model_skew = 1.02
    additive_model_sigma = 0.028

    def stack_estimate(qk_ms, toggles):
        deltas, measured_row = 0.0, results.get({(True, True): "smooth_hadamard", (True, False): "smooth", (False, True): "hadamard", (False, False): "int8"}[toggles]) or {}
        if measured_row.get("ms"):
            return measured_row["ms"], row_sigma(measured_row, "ms"), False
        for on, toggle_id in ((toggles[0], "smooth"), (toggles[1], "hadamard")):
            toggle_ms, _err = measured(results, toggle_id)
            if on and toggle_ms and int8_ms:
                deltas += toggle_ms - int8_ms
        sigma = row_sigma(results.get("int8") or {}, "ms")
        sigma = math.sqrt((sigma or 0.0) ** 2 + additive_model_sigma ** 2)
        return (qk_ms + deltas) * additive_model_skew, sigma, True

    composition_flip = False
    if use_quantized and noquant_ms and int8_ms:
        toggles = (bool(smooth_rec), bool(hadamard_rec))
        combo_label = {(True, True): "int8 qk + smooth + hadamard", (True, False): "int8 qk + smooth k", (False, True): "int8 qk + hadamard", (False, False): "int8 qk"}[toggles]
        combo_ms, combo_sigma, estimated = stack_estimate(int8_ms, toggles)
        combo_sigma = math.sqrt((combo_sigma or 0.0) ** 2 + (row_sigma(results.get("noquant") or {}, "ms") or 0.0) ** 2) or None
        stack_test = speed_verdict(combo_ms, noquant_ms, sigma=combo_sigma)
        if combo_ms and stack_test != "faster":
            use_quantized = False
            composition_flip = True
            source = "estimated additively at" if estimated else "measured"
            if stack_test == "inconclusive":
                quant_reason = f"the recommended stack ({combo_label}) {source} x{noquant_ms / combo_ms:.2f} vs unquantized sdnq attention, too close to the margin to call; disabled until it clearly wins"
            else:
                quant_reason = f"the recommended stack ({combo_label}) {source} x{noquant_ms / combo_ms:.2f} vs unquantized sdnq attention; the error buybacks eat the qk gain on this gpu"

    # the verdict must not hinge on int8 alone: bare float8 qk can clear the margin on
    # gpus where int8 falls short, so it gets its own shot before disabling
    fp8qk_ms, fp8qk_err = measured(results, "fp8qk")
    fp8_rescue = False
    if not use_quantized and not composition_flip and noquant_ms and fp8qk_ms:
        fp8_sigma = pair_sigma(results.get("fp8qk") or {}, "ms", results.get("noquant") or {}, "ms")
        if speed_verdict(fp8qk_ms, noquant_ms, sigma=fp8_sigma) == "faster" and not (fp8qk_err and int8_err and fp8qk_err > int8_err * recommend_error_cap):
            use_quantized = True
            fp8_rescue = True
            qk_inconclusive = False
            quant_reason = f"float8 qk measured x{noquant_ms / fp8qk_ms:.2f} vs unquantized sdnq attention where the int8 path fell short"
            if fp8qk_err:
                quant_reason += f", error {fp8qk_err:.5f}"
            quant_reason += "; smooth k and hadamard buybacks were only measured on int8"

    qk_choice = "enabled"
    qk_reason = quant_reason + "; enabled resolves to int8, uint8 remaps to int8"
    if fp8_rescue:
        qk_choice = "float8_e4m3fn"
        qk_reason = quant_reason
    elif fp8qk_ms and int8_ms:
        qk_compare = f"float8 qk measured x{int8_ms / fp8qk_ms:.2f} vs int8"
        if fp8qk_err and int8_err:
            qk_compare += f", error {fp8qk_err:.5f} vs {int8_err:.5f}"
        if fp8qk_ms < int8_ms * 0.95 and not (fp8qk_err and int8_err and fp8qk_err > int8_err * recommend_error_cap):
            qk_choice = "float8_e4m3fn"
            qk_reason = quant_reason + "; " + qk_compare
        else:
            qk_reason += f"; {qk_compare}"
    elif fp8_result["qk"][0]:
        qk_reason += "; float8 compiles here but was not benchmarked at this shape"
    if qk_inconclusive:
        qk_choice = current("sdnq_attention_matmul_type")
        qk_reason = quant_reason
    elif not use_quantized:
        qk_choice = "disabled"
        qk_reason = quant_reason
    rows.append(("MatMul type", current("sdnq_attention_matmul_type"), qk_choice, qk_reason))

    # pv rows were measured on top of int8 qk, so a pv verdict only holds when qk
    # quantization itself is recommended; note enabled means int8 pv on this dropdown.
    # each candidate is tested against the margin independently at a sidak-adjusted z:
    # picking the fastest first and testing it afterwards would bias toward enabling,
    # since the minimum of several noisy rows sits low by selection
    pv_candidates = []
    for pv_dtype, pv_name, pv_id in (("float8_e4m3fn", "fp8", "fp8pv"), ("int8", "int8", "int8pv"), ("float16", "fp16", "fp16pv")):
        pv_ms, pv_err = measured(results, pv_id)
        if pv_ms:
            pv_candidates.append((pv_dtype, pv_name, pv_id, pv_ms, pv_err))
    if use_quantized and pv_candidates and int8_ms:
        z_sel = sidak_z.get(len(pv_candidates), sidak_z[4])
        pv_winners, pv_inconclusive = [], False
        for pv_dtype, pv_name, pv_id, pv_ms, pv_err in pv_candidates:
            pv_test = speed_verdict(pv_ms, int8_ms, sigma=pair_sigma(results.get(pv_id) or {}, "ms", results.get("int8") or {}, "ms"), z=z_sel)
            if pv_test == "faster":
                pv_winners.append((pv_dtype, pv_name, pv_ms, pv_err))
            elif pv_test == "inconclusive":
                pv_inconclusive = True
        if pv_winners:
            fastest_pv = min(ms for _d, _n, ms, _e in pv_winners)
            near_fastest = [c for c in pv_winners if c[2] <= fastest_pv * 1.05]
            pv_dtype, pv_name, pv_ms, pv_err = min(near_fastest, key=lambda c: c[3] if c[3] is not None else float("inf"))
            if pv_err and int8_err and pv_err > int8_err * recommend_error_cap:
                rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "disabled", f"{pv_name} pv is x{int8_ms / pv_ms:.2f} faster but multiplies error x{pv_err / int8_err:.1f}; disabled keeps pv unquantized"))
            else:
                pv_reason = f"{pv_name} pv measured x{int8_ms / pv_ms:.2f} over int8 qk alone"
                if pv_err and int8_err:
                    pv_reason += f", error {pv_err:.5f} vs {int8_err:.5f}"
                rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), pv_dtype, pv_reason))
        elif pv_inconclusive:
            rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), current("sdnq_attention_pv_matmul_type"), "the best pv candidate sits within this run's noise of the margin; keeping the current setting"))
        else:
            rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "disabled", f"disabled keeps pv unquantized; {' and '.join(name for _d, name, _i, _m, _e in pv_candidates)} pv measured no gain here"))
    else:
        if qk_inconclusive and pv_candidates:
            pv_note = "the qk verdict above is inconclusive; pv follows it"
        elif not use_quantized and pv_candidates:
            pv_note = "qk quantization is not recommended above; pv on an unquantized qk path was not measured"
        else:
            pv_note = "disabled keeps pv unquantized"
        rows.append(("PV MatMul type", current("sdnq_attention_pv_matmul_type"), "disabled" if not qk_inconclusive else current("sdnq_attention_pv_matmul_type"), pv_note))

    if smooth_rec is not None:
        rows.append(("Use Smooth K", current("sdnq_attention_smooth_k"), str(smooth_rec), smooth_reason))
    if hadamard_rec is not None:
        rows.append(("Use Hadamard", current("sdnq_attention_use_hadamard"), str(hadamard_rec), hadamard_reason))

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
        emit("[dim]qk quantization is not worth it here, so MatMul type recommends disabled; the smooth k and hadamard rows show what to pick if it is enabled anyway[/dim]")

    notes = []
    # per-shape qk verdicts: the table above judges one reference shape, but the settings
    # are global and workloads differ; a split gpu (video wins, image loses) shows here
    # rather than being averaged away. cross and te shapes are prep-dominated, skip them
    shape_verdicts = []
    for shape_label in recommendation_presets:
        shape_results = all_results.get(shape_label)
        if not shape_results:
            continue
        shape_noquant_ms, _err = measured(shape_results, "noquant")
        shape_int8_ms, _err = measured(shape_results, "int8")
        if shape_noquant_ms and shape_int8_ms:
            shape_test = speed_verdict(shape_int8_ms, shape_noquant_ms, sigma=pair_sigma(shape_results.get("int8") or {}, "ms", shape_results.get("noquant") or {}, "ms"))
            word = {"faster": "enabled", "not_faster": "disabled", "inconclusive": "inconclusive"}[shape_test]
            shape_verdicts.append((word, f"{shape_label} {word} (x{shape_noquant_ms / shape_int8_ms:.2f})"))
    if len(shape_verdicts) > 1:
        split = len({word for word, _text in shape_verdicts}) > 1
        line = f"qk verdict by shape: {', '.join(text for _word, text in shape_verdicts)}"
        if split:
            notes.append(f"[yellow]{line}; settings are global, pick for the shapes you generate at[/yellow]")
        else:
            notes.append(line)
    labels = {config_id: config_label(config_id, label) for config_id, label, _kwargs in bench_configs}
    best_id = best_config(results)
    if base_ms and best_id is not None:
        best_ms = results[best_id]["ms"]
        if best_ms < base_ms * 0.95:
            notes.append(f"worth enabling: {labels[best_id]} measured x{base_ms / best_ms:.2f} vs torch sdpa at {reference}")
        else:
            notes.append(f"[yellow]not worth enabling: best is {labels[best_id]} at x{base_ms / best_ms:.2f} vs torch sdpa, the sdp override costs performance here[/yellow]")
    # the individual setting rows above compose into one config; cite it as actually measured
    full_ms, full_err = measured(results, "full")
    if full_ms and base_ms and int8_ms:
        stack_note = f"all options on (smooth k + hadamard + int8 pv over int8 qk): x{base_ms / full_ms:.2f} vs torch sdpa"
        if full_err is not None:
            stack_note += f", error {full_err:.5f}"
        if int8_err is not None:
            stack_note += f"; int8 qk alone x{base_ms / int8_ms:.2f}, error {int8_err:.5f}"
        notes.append(stack_note)
    if prep_status == "failing_dynamic":
        notes.append("[red]generation fails here: compile cannot build the dynamic-shape prep. fix: set SDNQ_COMPILE_KWARGS='{\"dynamic\": false}' (verified on this machine) or install msvc build tools; numbers above use that workaround[/red]")
    elif prep_status == "failing":
        notes.append("[red]generation fails here: torch compile is broken. fix: disable Dequantize using torch.compile or install msvc build tools; numbers above use eager prep[/red]")
    if not fp8_result["qk"][0]:
        if fp8_failure_is_capability(fp8_result["qk"][1]):
            notes.append("[red]float8_e4m3fn unsupported on this gpu: selecting it in either dropdown fails generation[/red]")
        else:
            notes.append("[red]float8_e4m3fn failed to compile here: selecting it in either dropdown fails generation[/red]; not the hardware signature, so a torch or triton issue rather than the gpu")
    sage_ms, _sage_err = measured(results, "sage")
    if sage_ms and int8_ms:
        notes.append(f"vs sage at {reference}: sdnq int8 {int8_ms:.2f} ms, sage {sage_ms:.2f} ms; sdnq also covers masks, gqa, causal")
    sagefp16_ms, _sagefp16_err = measured(results, "sagefp16")
    if sagefp16_ms:
        notes.append("sage fp16 accum: benchmark-only, no sdnext setting reaches it (sage pins fp32 accum on sm86)")
    # compare flash against the config the verdict above actually recommends
    amdflash_ms, _amdflash_err = measured(results, "amdflash")
    if amdflash_ms:
        if use_quantized and int8_ms:
            notes.append(f"vs triton flash at {reference}: sdnq int8 {int8_ms:.2f} ms, flash {amdflash_ms:.2f} ms; sdnq also covers masks, its gain tracks this gpu's int8 throughput")
        elif noquant_ms:
            notes.append(f"vs triton flash at {reference}: unquantized sdnq {noquant_ms:.2f} ms, flash {amdflash_ms:.2f} ms; sdnq also covers masks")
    # cross-reference the combined block section when it ran in this invocation: kernel-scope
    # error buybacks can vanish at block output, and per-call costs differ inside a real block
    block_results = report.get("block", {}).get("results", {})
    block_base = block_results.get("int8-mm-atten") or {}
    if block_base.get("err") and block_base.get("ms"):
        cross_parts = []
        buyback_seen = False
        for variant_key, setting_name in (("int8-mm-smooth", "smooth k"), ("int8-mm-hadamard", "hadamard"), ("int8-mm-atten-full", "full stack (smooth + hadamard + int8 pv)")):
            variant_entry = block_results.get(variant_key) or {}
            if variant_entry.get("err") and variant_entry.get("ms"):
                cross_parts.append(f"{setting_name} output error {variant_entry['err']:.5f} vs {block_base['err']:.5f} at {variant_entry['ms'] - block_base['ms']:+.2f} ms per block")
                if variant_entry["err"] < block_base["err"] * 0.98:
                    buyback_seen = True
        if cross_parts:
            cross_note = f"block-level cross-check vs plain int8 attention: {'; '.join(cross_parts)}"
            if not buyback_seen:
                cross_note += "; the kernel-scope error buybacks above did not change block output error at this image shape"
            notes.append(cross_note)
    notes.append("*: best config per shape, lowest error among rows within 5% of the fastest sdnq time")
    notes.append("default settings run the int8 qk row: MatMul type enabled resolves to int8, pv stays disabled")
    notes.append("prep: q/k/v quantization before the kernel, included in the median; shrinks where compile fuses it")
    notes.append("[yellow]non pow2 head dims (sd 1.5): quantized matmul fails compile, hadamard hangs it; disable quantized matmul there or set SDNQ_COMPILE_KWARGS='{\"dynamic\": false}'[/yellow]")
    notes.append("settings above apply on the next generation; the SDNQ attention SDP override needs a restart")
    notes.append("errors use synthetic outlier-heavy keys; real models, especially qk-normed dits, sit lower")
    emit(Panel("\n".join(notes), title="notes", box=ROUNDED_BOX))
    report["recommendations_attention"] = dict(reference=reference, rows=[list(row) for row in rows], notes=notes)


def ratio_text(numerator_ms, denominator_ms):
    if not numerator_ms or not denominator_ms:
        return None
    return f"x{numerator_ms / denominator_ms:.2f}"


def build_dequant_recommendations(dequant_results, weight_dequant_result, variant_results=None, float_mm_results=None, sweep_results=None):
    reference = None
    for shape_label, _out_features, _in_features in dequant_shapes:
        results = dequant_results.get(shape_label)
        if results and results.get("int8", {}).get("eager_ms") is not None:
            reference = shape_label
            break
    if reference is None:
        emit("[yellow]no successful dequant benchmark, sdnq recommendations unavailable[/yellow]")
        return
    results = dequant_results[reference]
    base_ms = results.get("bf16", {}).get("fwd_ms")

    def entry(dtype_id):
        return results.get(dtype_id) or {}

    def current(key):
        return str(getattr(shared.opts, key))

    table = Table(title=f"recommended settings (Compute Settings -> SDNQ), measured at the {reference} layer", box=ROUNDED_BOX)
    table.add_column("setting")
    table.add_column("current", justify="center")
    table.add_column("recommended", justify="center")
    table.add_column("reason")

    rows = []
    weights_mode = str(getattr(shared.opts, "sdnq_quantize_weights_mode", "int8"))
    mode_ids = {cfg["weights_dtype"]: dtype_id for dtype_id, _label, cfg in dequant_dtype_configs}
    mm_id = mode_ids.get(weights_mode, "int8")

    # the compile toggle is output-neutral (dequant drift is checked separately), so its verdict
    # is a symmetric faster/slower test (margin 1.0) at the layer-forward scope the option gates,
    # voted across every measured shape; the standalone kernel can lose to eager on small layers
    # from launch overhead alone, which is why it never decides this row
    compile_id = mm_id if entry(mm_id).get("fwd_eager_ms") and entry(mm_id).get("fwd_compiled_ms") else "int8"
    fwd_votes = []
    for shape_label in dequant_results:
        vote_row = (dequant_results.get(shape_label) or {}).get(compile_id) or {}
        fwd_eager, fwd_compiled = vote_row.get("fwd_eager_ms"), vote_row.get("fwd_compiled_ms")
        if fwd_eager and fwd_compiled:
            test = speed_verdict(fwd_compiled, fwd_eager, sigma=pair_sigma(vote_row, "fwd_compiled_ms", vote_row, "fwd_eager_ms"), margin=1.0)
            fwd_votes.append((fwd_eager / fwd_compiled, test))
    int8_eager, int8_compiled = entry("int8").get("eager_ms"), entry("int8").get("compiled_ms")
    if fwd_votes:
        ratios = sorted(ratio for ratio, _test in fwd_votes)
        span = f"x{ratios[0]:.2f}" if len(ratios) == 1 else f"x{ratios[0]:.2f}-x{ratios[-1]:.2f}"
        scope_text = f"{compile_id} layer forward measured {span} compiled vs eager across {len(fwd_votes)} shape{'s' if len(fwd_votes) > 1 else ''}"
        kinds = {test for _ratio, test in fwd_votes}
        if "faster" in kinds and "not_faster" not in kinds:
            compile_choice, compile_reason = "True", scope_text
        elif "not_faster" in kinds and "faster" not in kinds:
            compile_choice, compile_reason = "False", scope_text
        elif "faster" in kinds:
            n_faster = sum(1 for _ratio, test in fwd_votes if test == "faster")
            n_slower = sum(1 for _ratio, test in fwd_votes if test == "not_faster")
            compile_choice = current("sdnq_dequantize_compile")
            compile_reason = f"{scope_text}; split verdict ({n_faster} faster, {n_slower} slower), keeping the current setting"
        else:
            compile_choice = current("sdnq_dequantize_compile")
            compile_reason = f"{scope_text}, within this run's noise; keeping the current setting"
        rows.append(("Dequantize using torch.compile", current("sdnq_dequantize_compile"), compile_choice, compile_reason))
    elif int8_eager and int8_compiled:
        compile_test = speed_verdict(int8_compiled, int8_eager, sigma=pair_sigma(entry("int8"), "compiled_ms", entry("int8"), "eager_ms"), margin=1.0)
        compile_reason = f"int8 dequant kernel measured {ratio_text(int8_eager, int8_compiled)} compiled vs eager, no layer forward data"
        if compile_test == "inconclusive":
            compile_choice = current("sdnq_dequantize_compile")
            compile_reason += "; within this run's noise, keeping the current setting"
        else:
            compile_choice = str(compile_test == "faster")
        rows.append(("Dequantize using torch.compile", current("sdnq_dequantize_compile"), compile_choice, compile_reason))
    elif int8_eager:
        rows.append(("Dequantize using torch.compile", current("sdnq_dequantize_compile"), current("sdnq_dequantize_compile"), "compiled int8 dequant unavailable, keeping the current value"))

    # judge quantized matmul on the dtype the current config would quantize with, falling back to int8;
    # speed never wins alone: the faster path must also hold output error within recommend_error_cap
    mm_entry = entry(mm_id)
    # explicit MatMul type rows are measured at the first dequant shape only; ignore them when
    # the recommendation reference fell back to another shape
    float_mm = float_mm_results if (float_mm_results and reference == dequant_shapes[0][0]) else {}

    def float_mm_entry(dtype_id, mm_dtype):
        return float_mm.get(f"{dtype_id}+{mm_dtype}") or {}

    # the enable checkbox is gone: one row on the type dropdown carries the verdict;
    # candidates are enabled (auto-resolved dtype) plus any measured explicit alternative
    mm_candidates = []
    if mm_entry.get("mm_ms"):
        mm_candidates.append(("enabled", mm_entry.get("mm_dtype"), mm_entry["mm_ms"], mm_entry.get("mm_err")))
    if mm_id in float_mm_dtypes:
        for alt_dtype in float_mm_alternative_dtypes:
            alt = float_mm_entry(mm_id, alt_dtype)
            if alt.get("mm_ms"):
                mm_candidates.append((alt_dtype, alt.get("mm_dtype"), alt["mm_ms"], alt.get("mm_err")))
    if mm_candidates and mm_entry.get("fwd_ms"):
        fastest_mm = min(ms for _sel, _res, ms, _err in mm_candidates)
        near_fastest = [c for c in mm_candidates if c[2] <= fastest_mm * 1.05]
        best_sel, best_resolved, best_ms, best_err = min(near_fastest, key=lambda c: c[3] if c[3] is not None else float("inf"))
        fwd_err = mm_entry.get("fwd_err")
        err_ok = not (fwd_err and best_err) or best_err <= fwd_err * recommend_error_cap
        best_entry = mm_entry if best_sel == "enabled" else float_mm_entry(mm_id, best_sel)
        mm_test = speed_verdict(best_ms, mm_entry["fwd_ms"], sigma=pair_sigma(best_entry, "mm_ms", mm_entry, "fwd_ms"), z=sidak_z.get(len(mm_candidates), sidak_z[4]))
        recommend_mm = mm_test == "faster" and err_ok
        mm_reason = f"{mm_id} quantized matmul ({best_resolved}) measured {ratio_text(mm_entry['fwd_ms'], best_ms)} vs the dequant path"
        if fwd_err and best_err:
            mm_reason += f", output error {best_err:.5f} vs {fwd_err:.5f}"
        if mm_test == "faster" and not err_ok:
            mm_reason += f"; not recommended, the speedup multiplies output error beyond x{recommend_error_cap:.0f}"
        if weights_mode not in mode_ids:
            mm_reason += f"; current quantization type {weights_mode} was not benchmarked, judged on int8"
        type_parts = []
        for sel, resolved, ms, err in mm_candidates:
            sel_label = f"enabled ({resolved})" if sel == "enabled" else sel
            type_parts.append(f"{sel_label} {ms:.3f} ms err {err:.5f}" if err is not None else f"{sel_label} {ms:.3f} ms")
        if len(mm_candidates) > 1:
            mm_reason += f"; lowest error within 5% of the fastest: {', '.join(type_parts)}"
        elif best_sel == "enabled":
            mm_reason += f"; enabled resolves to {best_resolved} for {mm_id} weights"
        else:
            mm_reason += f"; enabled (float8_e4m3fn) failed on this gpu for {mm_id} weights"
        if mm_test == "inconclusive" and err_ok:
            mm_choice = current("sdnq_quantize_matmul_mode")
            mm_reason += "; too close to the margin to call at this run's noise, keeping the current setting"
        else:
            mm_choice = best_sel if recommend_mm else "disabled"
        rows.append(("Quantized MatMul type", current("sdnq_quantize_matmul_mode"), mm_choice, mm_reason))

    # the te override is independent now, with its own dtype option; judge it at
    # text-encoder geometry using the dtype the te config would quantize with
    te_shape = next((label for label, _out, _in in dequant_shapes if label.startswith("te ")), None)
    te_weights_mode = str(getattr(shared.opts, "sdnq_quantize_weights_mode_te", "Same as model"))
    if te_weights_mode in {"Same as model", "default"}:
        te_weights_mode = weights_mode
    te_mm_id = mode_ids.get(te_weights_mode, "int8")
    te_entry = ((dequant_results.get(te_shape) or {}).get(te_mm_id) or {}) if te_shape else {}
    if te_entry.get("fwd_ms") and te_entry.get("mm_ms"):
        te_fwd_err, te_mm_err = te_entry.get("fwd_err"), te_entry.get("mm_err")
        te_err_ok = not (te_fwd_err and te_mm_err) or te_mm_err <= te_fwd_err * recommend_error_cap
        te_test = speed_verdict(te_entry["mm_ms"], te_entry["fwd_ms"], sigma=pair_sigma(te_entry, "mm_ms", te_entry, "fwd_ms"))
        te_reason = f"{te_mm_id} quantized matmul measured {ratio_text(te_entry['fwd_ms'], te_entry['mm_ms'])} vs the dequant path at {te_shape}"
        if te_fwd_err and te_mm_err:
            te_reason += f", output error {te_mm_err:.5f} vs {te_fwd_err:.5f}"
        if te_test == "faster" and not te_err_ok:
            te_reason += f"; not recommended, the speedup multiplies output error beyond x{recommend_error_cap:.0f}"
        if te_test == "inconclusive" and te_err_ok:
            te_choice = current("sdnq_quantize_matmul_mode_te")
            te_reason += "; too close to the margin to call at this run's noise, keeping the current setting"
        else:
            te_choice = "enabled" if te_test == "faster" and te_err_ok else "disabled"
        te_reason += "; applies when text encoders are sdnq-quantized"
        rows.append(("Quantized MatMul type for Text Encoders", current("sdnq_quantize_matmul_mode_te"), te_choice, te_reason))

    sweeps = sweep_results or {}
    sweeps_at_reference = reference == dequant_shapes[0][0]
    mm_on = getattr(shared.opts, "sdnq_quantize_matmul_mode", "disabled") != "disabled"

    # Group size: judged on the path the current config runs (mm cells when quantized matmul
    # is on); an explicit size must cut error meaningfully without real speed or size cost
    groups = sweeps.get("groups") or {}
    group_id = mm_id if mm_id in group_sweep_dtypes else "int8"
    auto_group = groups.get(f"{group_id}@0") or {}
    if sweeps_at_reference and auto_group:
        def group_metrics(g_entry):
            ms = g_entry.get("mm_ms") if mm_on else g_entry.get("fwd_ms")
            err = g_entry.get("mm_err") if mm_on else g_entry.get("weight_err")
            return ms, err
        auto_ms, auto_err = group_metrics(auto_group)
        candidates = []
        for group in group_sweep_values:
            if group == 0:
                continue
            g_entry = groups.get(f"{group_id}@{group}") or {}
            g_ms, g_err = group_metrics(g_entry)
            if g_ms and g_err and g_entry.get("size_bytes"):
                candidates.append((group, g_ms, g_err, g_entry["size_bytes"]))
        if auto_ms and auto_err and auto_group.get("size_bytes"):
            qualifying = [c for c in candidates if c[2] < auto_err * 0.80 and c[1] <= auto_ms * 1.05 and c[3] <= auto_group["size_bytes"] * 1.10]
            resolved = auto_group.get("mm_resolved") if mm_on else auto_group.get("resolved")
            if qualifying:
                best_group, best_gms, best_gerr, best_gsize = min(qualifying, key=lambda c: c[2])
                group_reason = f"{group_id} error {best_gerr:.5f} vs {auto_err:.5f} at auto ({resolved}), {best_gms:.3f} vs {auto_ms:.3f} ms, {size_cell(best_gsize).strip()} vs {size_cell(auto_group['size_bytes']).strip()}"
                rows.append(("Group size", current("sdnq_group_size"), str(best_group), group_reason))
            else:
                explicit_best = min(candidates, key=lambda c: c[2]) if candidates else None
                group_reason = f"auto resolves to {resolved} for {group_id} weights"
                if mm_on and explicit_best and auto_err and abs(explicit_best[2] - auto_err) <= auto_err * 0.10:
                    group_reason += f"; grouped storage is re-quantized per forward for quantized matmul, so no swept group changed mm error (best {explicit_best[2]:.5f} vs {auto_err:.5f})"
                elif explicit_best:
                    group_reason += f"; best explicit ({explicit_best[0]}) measured error {explicit_best[2]:.5f} vs {auto_err:.5f} at {explicit_best[1]:.3f} vs {auto_ms:.3f} ms and {size_cell(explicit_best[3]).strip()} vs {size_cell(auto_group['size_bytes']).strip()}"
                rows.append(("Group size", current("sdnq_group_size"), "0", group_reason))

    # Dequantize using full precision: off keeps scales in the model dtype; the speed gain
    # must hold error within the cap to be recommended
    toggles = sweeps.get("toggles") or {}
    fp32_off = toggles.get("fp32_off") or {}
    toggle_id = mm_id if mm_id in toggle_dtypes else "int8"
    off_entry = fp32_off.get(toggle_id) or {}
    on_entry = entry(toggle_id)
    if sweeps_at_reference and off_entry.get("fwd_ms") and on_entry.get("fwd_ms") and off_entry.get("weight_err") and on_entry.get("weight_err"):
        off_faster = off_entry["fwd_ms"] <= on_entry["fwd_ms"] * 0.95
        off_err_ok = off_entry["weight_err"] <= on_entry["weight_err"] * recommend_error_cap
        recommend_fp32 = not (off_faster and off_err_ok)
        fp32_reason = f"{toggle_id} with full precision off: {off_entry['fwd_ms']:.3f} vs {on_entry['fwd_ms']:.3f} ms, weight error {off_entry['weight_err']:.5f} vs {on_entry['weight_err']:.5f}"
        rows.append(("Dequantize using full precision", current("sdnq_dequantize_fp32"), str(recommend_fp32), fp32_reason))

    # conv settings: the conv matmul verdict is spatial-size dependent, so every measured conv
    # shape weighs in; the recommended value follows the first (unet-class) shape, with the
    # other shape's number in the reason so vae-quantizing sessions can differ
    conv_all = sweeps.get("conv") or {}
    conv_per_shape = {}
    for conv_shape_label, _oc, _ic, _k, _px in conv_shapes:
        shape_conv = conv_all.get(conv_shape_label) or {}
        mm_rows_here = [(config_id, shape_conv.get(config_id) or {}) for config_id in ("int8-mm", "uint8-mm")]
        conv_per_shape[conv_shape_label] = (shape_conv.get("bf16") or {}, shape_conv.get("int8") or {}, [(config_id, row) for config_id, row in mm_rows_here if row.get("fwd_ms")])
    conv_base, conv_int8, conv_mm_rows = conv_per_shape.get(conv_shapes[0][0], ({}, {}, []))
    if conv_base.get("fwd_ms"):
        quant_rows = [(config_id, row) for config_id, row in [("int8", conv_int8)] + conv_mm_rows if row.get("fwd_ms")]
        if quant_rows:
            best_conv_id, best_conv = min(quant_rows, key=lambda item: item[1]["fwd_ms"])
            conv_free = best_conv["fwd_ms"] <= conv_base["fwd_ms"] * 1.05
            conv_reason = f"{best_conv_id} conv measured {best_conv['fwd_ms']:.3f} vs {conv_base['fwd_ms']:.3f} ms {dtype_label()} at {conv_shapes[0][0]}, output error {best_conv.get('out_err', 0):.5f} vs {conv_base.get('out_err', 0):.5f}, size {size_cell(best_conv.get('size_bytes', 0)).strip()} vs {size_cell(conv_base.get('size_bytes', 0)).strip()}"
            if not conv_free:
                conv_reason += "; enable for the size saving, not speed"
            rows.append(("Quantize convolutional layers", current("sdnq_quantize_conv_layers"), str(conv_free), conv_reason))
        if conv_int8.get("fwd_ms") and conv_mm_rows:
            best_mm_id, best_mm = min(conv_mm_rows, key=lambda item: item[1]["fwd_ms"])
            conv_mm_test = speed_verdict(best_mm["fwd_ms"], conv_int8["fwd_ms"], sigma=pair_sigma(best_mm, "fwd_ms", conv_int8, "fwd_ms"), z=sidak_z.get(len(conv_mm_rows), sidak_z[4]))
            mm_err_ok = not (best_mm.get("out_err") and conv_int8.get("out_err")) or best_mm["out_err"] <= conv_int8["out_err"] * recommend_error_cap
            conv_mm_reason = f"{best_mm_id} measured {best_mm['fwd_ms']:.3f} vs {conv_int8['fwd_ms']:.3f} ms for the int8 conv dequant path at {conv_shapes[0][0]}, output error {best_mm.get('out_err', 0):.5f} vs {conv_int8.get('out_err', 0):.5f}"
            other_base, _other_int8, other_mm_rows = conv_per_shape.get(conv_shapes[1][0], ({}, {}, [])) if len(conv_shapes) > 1 else ({}, {}, [])
            if other_base.get("fwd_ms") and other_mm_rows:
                other_best_id, other_best = min(other_mm_rows, key=lambda item: item[1]["fwd_ms"])
                conv_mm_reason += f"; at {conv_shapes[1][0]} {other_best_id} measured {ratio_text(other_base['fwd_ms'], other_best['fwd_ms'])} vs the {dtype_label()} conv"
                if other_best["fwd_ms"] > other_base["fwd_ms"] * 1.05:
                    conv_mm_reason += ", a slowdown; disable when quantizing vae-class convs"
            if conv_mm_test == "inconclusive" and mm_err_ok:
                conv_mm_choice = current("sdnq_use_quantized_matmul_conv")
                conv_mm_reason += "; too close to the margin to call at this run's noise, keeping the current setting"
            else:
                conv_mm_choice = str(conv_mm_test == "faster" and mm_err_ok)
            rows.append(("Use quantized MatMul with conv", current("sdnq_use_quantized_matmul_conv"), conv_mm_choice, conv_mm_reason))

    differing = 0
    for setting, current_value, recommended, reason in rows:
        if current_value == recommended:
            table.add_row(setting, f"[green]{current_value}[/green]", recommended, reason)
        else:
            differing += 1
            table.add_row(setting, f"[yellow]{current_value}[/yellow]", recommended, reason)
    emit(table)
    if differing:
        emit(f"[yellow]{differing} setting{'s' if differing > 1 else ''} differ{'' if differing > 1 else 's'} from the recommended value, change in Compute Settings -> SDNQ[/yellow]")
    elif rows:
        emit("[green]current settings already match the recommendations[/green]")

    notes = []
    speedups = []
    for dtype_id, label, _cfg in dequant_dtype_configs:
        text = ratio_text(entry(dtype_id).get("eager_ms"), entry(dtype_id).get("compiled_ms"))
        if text:
            speedups.append(f"{label} {text}")
    if speedups:
        notes.append(f"compiled dequant speedup vs eager at {reference}: {', '.join(speedups)}; standalone kernel scope, the compile verdict uses the layer forward")
    fp8_entry, fp8sdnq_entry = entry("fp8"), entry("fp8sdnq")
    if fp8_entry.get("compiled_ms") and fp8sdnq_entry.get("compiled_ms"):
        notes.append(f"fp8 storage: native e4m3 compiled {ratio_text(fp8sdnq_entry['compiled_ms'], fp8_entry['compiled_ms'])} vs the float8_e4m3fn_sdnq uint8 view")
    elif fp8sdnq_entry.get("compiled_ms") and fp8_entry.get("eager_ms"):
        notes.append(f"fp8 storage: float8_e4m3fn_sdnq (uint8 view) compiled {ratio_text(fp8_entry['eager_ms'], fp8sdnq_entry['compiled_ms'])} vs eager e4m3 dequant")
    if weight_dequant_result is not None:
        e5m2_ok = weight_dequant_result["float8_e5m2"][0]
        e4m3_ok = weight_dequant_result["float8_e4m3fn"][0]
        if not e4m3_ok and e5m2_ok:
            notes.append("compiled dequant: e5m2 compiles raw, e4m3 does not; sdnq's upcast covers e4m3")
        elif not e4m3_ok and not e5m2_ok:
            notes.append("[yellow]compiled dequant: neither fp8 dtype compiles raw; sdnq upcasts e4m3, e5m2 has no such path[/yellow]")
    fp8_mm_failed = [dtype_id for dtype_id, _label, _cfg in dequant_dtype_configs if entry(dtype_id).get("mm_error") and entry(dtype_id).get("mm_dtype") == "float8_e4m3fn"]
    if fp8_mm_failed:
        notes.append(f"[yellow]enabled quantized matmul auto-selects float8_e4m3fn for {', '.join(fp8_mm_failed)} and failed; set an explicit MatMul type for float weight dtypes[/yellow]")
        for dtype_id in fp8_mm_failed:
            alt_parts = []
            for alt_dtype in float_mm_alternative_dtypes:
                alt = float_mm_entry(dtype_id, alt_dtype)
                if alt.get("mm_ms"):
                    part = f"{alt_dtype} mm {alt['mm_ms']:.3f} ms"
                    if alt.get("mm_err") is not None:
                        part += f" err {alt['mm_err']:.5f}"
                    alt_parts.append(part)
            dequant_fwd, dequant_err = entry(dtype_id).get("fwd_ms"), entry(dtype_id).get("fwd_err")
            if alt_parts and dequant_fwd:
                base_part = f"dequant path {dequant_fwd:.3f} ms"
                if dequant_err is not None:
                    base_part += f" err {dequant_err:.5f}"
                notes.append(f"{dtype_id} with MatMul type set explicitly: {', '.join(alt_parts)}; {base_part}")
    if base_ms and entry(mm_id).get("fwd_ms"):
        notes.append(f"current quantization type {weights_mode}: dequant-path forward {ratio_text(entry(mm_id)['fwd_ms'], base_ms)} vs a {dtype_label()} nn.Linear at {reference}")
    # size/error frontier: the dtypes no other measured dtype beats on both axes at once
    sized = [(dtype_id, e["size_bytes"], e["weight_err"]) for dtype_id, _label, _cfg in dequant_dtype_configs for e in [entry(dtype_id)] if e.get("size_bytes") and e.get("weight_err")]
    frontier = sorted((c for c in sized if not any(o is not c and o[1] <= c[1] and o[2] <= c[2] for o in sized)), key=lambda c: c[1])
    if len(frontier) > 1:
        notes.append(f"size/error frontier at {reference}: " + ", ".join(f"{dtype_id} {size_cell(size).strip()} err {err:.3f}" for dtype_id, size, err in frontier))
    if variant_results:
        for variant_id, _cfg in dequant_variant_configs:
            parts = []
            for dtype_id in dequant_variant_dtypes:
                variant_entry = variant_results.get(f"{dtype_id}+{variant_id}") or {}
                if variant_entry.get("err_ratio"):
                    parts.append(f"{dtype_id} err x{variant_entry['err_ratio']:.2f} for {quant_cell(variant_entry['quant_s']).strip()} quant")
            if parts:
                notes.append(f"{variant_id} vs plain quantization: {', '.join(parts)}")
    groups_sweep = sweeps.get("groups") or {}
    if groups_sweep and sweeps_at_reference:
        group_parts = []
        mm_flat = True
        for dtype_id in group_sweep_dtypes:
            auto_g = groups_sweep.get(f"{dtype_id}@0") or {}
            swept = [(group, groups_sweep.get(f"{dtype_id}@{group}") or {}) for group in group_sweep_values if group != 0]
            swept = [(group, g_entry) for group, g_entry in swept if g_entry.get("weight_err")]
            if auto_g.get("weight_err") and swept:
                best_group, best_entry = min(swept, key=lambda item: item[1]["weight_err"])
                if best_entry.get("size_bytes") and auto_g.get("size_bytes"):
                    group_parts.append(f"{dtype_id} group {best_group} weight err x{best_entry['weight_err'] / auto_g['weight_err']:.2f} of auto for {size_cell(best_entry['size_bytes']).strip()} vs {size_cell(auto_g['size_bytes']).strip()}")
                mm_errs = [g_entry["mm_err"] for _group, g_entry in swept if g_entry.get("mm_err")] + ([auto_g["mm_err"]] if auto_g.get("mm_err") else [])
                if len(mm_errs) > 1 and max(mm_errs) > min(mm_errs) * 1.10:
                    mm_flat = False
        if group_parts:
            groups_note = "group size storage tradeoff (dequant path): " + ", ".join(group_parts)
            if mm_flat:
                groups_note += "; grouped storage is re-quantized per forward for quantized matmul, no swept group changed mm error"
            notes.append(groups_note)
    conv_note_all = sweeps.get("conv") or {}
    conv_note_parts = []
    for conv_shape_label, _oc, _ic, _k, _px in conv_shapes:
        shape_conv = conv_note_all.get(conv_shape_label) or {}
        base_row = shape_conv.get("bf16") or {}
        mm_here = [row for config_id in ("int8-mm", "uint8-mm") for row in [shape_conv.get(config_id) or {}] if row.get("fwd_ms")]
        if base_row.get("fwd_ms") and mm_here:
            best_here = min(mm_here, key=lambda row: row["fwd_ms"])
            conv_note_parts.append(f"{conv_shape_label} best conv mm {ratio_text(base_row['fwd_ms'], best_here['fwd_ms'])}")
    if len(conv_note_parts) > 1:
        notes.append(f"conv quantized matmul vs the {dtype_label()} conv by shape: " + "; ".join(conv_note_parts))
    te_shape = next((label for label, _out, _in in dequant_shapes if label.startswith("te ")), None)
    te_results = dequant_results.get(te_shape) or {} if te_shape else {}
    te_int8 = te_results.get("int8") or {}
    if te_int8.get("fwd_ms") and te_int8.get("mm_ms") and entry("int8").get("fwd_ms") and entry("int8").get("mm_ms"):
        notes.append(f"text-encoder geometry ({te_shape}): int8 quantized mm {ratio_text(te_int8['fwd_ms'], te_int8['mm_ms'])} vs the dequant path, {ratio_text(entry('int8')['fwd_ms'], entry('int8')['mm_ms'])} at {reference}")
    krea2_parts = []
    for krea2_label, _out, _in in dequant_shapes:
        if not krea2_label.startswith("krea2 "):
            continue
        krea2_int8 = (dequant_results.get(krea2_label) or {}).get("int8") or {}
        if krea2_int8.get("fwd_ms") and krea2_int8.get("mm_ms"):
            krea2_parts.append(f"{krea2_label} {ratio_text(krea2_int8['fwd_ms'], krea2_int8['mm_ms'])}")
    if krea2_parts and entry("int8").get("fwd_ms") and entry("int8").get("mm_ms"):
        notes.append(f"krea 2 geometry: int8 quantized mm vs the dequant path {', '.join(krea2_parts)}; {ratio_text(entry('int8')['fwd_ms'], entry('int8')['mm_ms'])} at {reference}")
    svd_sweep = sweeps.get("svd") or {}
    if svd_sweep and sweeps_at_reference:
        for dtype_id in svd_rank_sweep_dtypes:
            parts = []
            for rank in svd_rank_sweep_values:
                rank_entry = svd_sweep.get(f"{dtype_id}@{rank}") or {}
                if rank_entry.get("err_ratio"):
                    parts.append(f"rank {rank} err x{rank_entry['err_ratio']:.2f} ({size_cell(rank_entry['size_bytes']).strip()})")
            if parts:
                notes.append(f"svd rank on {dtype_id}: {', '.join(parts)}")
    hgroup_sweep = sweeps.get("hgroups") or {}
    hgroup_parts = [f"group {hgroup} err x{hentry['err_ratio']:.2f}" for hgroup in hadamard_group_values for hentry in [hgroup_sweep.get(str(hgroup)) or {}] if hentry.get("err_ratio")]
    if hgroup_parts and sweeps_at_reference:
        notes.append(f"weight-side hadamard group size on int8: {', '.join(hgroup_parts)}")
    dynamic_sweep = (sweeps.get("toggles") or {}).get("dynamic") or {}
    dynamic_parts = []
    for requested in dynamic_quant_requests:
        dyn_entry = dynamic_sweep.get(requested) or {}
        if dyn_entry.get("chosen"):
            dynamic_parts.append(f"{requested} -> {dyn_entry['chosen']}" + (f" err {dyn_entry['weight_err']:.3f}" if dyn_entry.get("weight_err") else ""))
    if dynamic_parts:
        notes.append(f"dynamic quantization on these synthetic weights (default loss thresholds): {', '.join(dynamic_parts)}")
    cpu_quant_s = (sweeps.get("toggles") or {}).get("cpu_quant_s")
    gpu_quant_s = (sweeps.get("toggles") or {}).get("gpu_quant_s")
    if cpu_quant_s and gpu_quant_s:
        notes.append(f"Quantize using GPU: one int8 layer at {dequant_shapes[0][0]} takes {quant_cell(gpu_quant_s).strip()} on gpu vs {quant_cell(cpu_quant_s).strip()} on cpu; scales with layer count at load")
    notes.append("weight err: relative error of the dequantized weight vs fp32; forward outputs inherit it")
    notes.append("errors use synthetic outlier-heavy weights; rotation and svd gains are outlier-driven, real models gain less")
    if notes:
        emit(Panel("\n".join(notes), title="dequant notes", box=ROUNDED_BOX))
    report["recommendations_dequant"] = dict(reference=reference, rows=[list(row) for row in rows], notes=notes)


def save_report(path, args, sections, selected):
    run_info = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), sections=sections, shapes=selected, iters=args.iters, warmup=args.warmup, dtype=dtype_label())
    run_info.update(report.get("run") or {}) # keep the drift fields main writes after the bench sections
    report["run"] = run_info
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    console.print(f"structured results saved to {path}")


pending_outputs = None # set by main once arguments are validated, so a crash can still flush
outputs_written = False


def flush_outputs():
    # write --save/--json exactly once, from the normal path or the crash handler; partial
    # results beat losing an hour-long run to a kernel fault in one config
    global outputs_written # pylint: disable=global-statement
    if outputs_written or pending_outputs is None:
        return
    outputs_written = True
    if pending_outputs["save"]:
        save_transcript(pending_outputs["save"])
    if pending_outputs["json"]:
        save_report(pending_outputs["json"], pending_outputs["args"], pending_outputs["sections"], pending_outputs["selected"])


def main():
    global bench_dtype # pylint: disable=global-statement
    args = parse_cli()
    sections = [s.strip().lower() for s in args.sections.split(",") if s.strip()]
    unknown_sections = [s for s in sections if s not in all_sections]
    if unknown_sections:
        console.print(f"[red]unknown section(s): {', '.join(unknown_sections)}; available: {', '.join(all_sections)}[/red]")
        sys.exit(1)
    known_dtypes = [dtype_id for dtype_id, _label, _cfg in dequant_dtype_configs]
    selected_dtypes = None if args.dequant_dtypes.strip().lower() == "all" else [s.strip() for s in args.dequant_dtypes.split(",") if s.strip()]
    if selected_dtypes is not None:
        unknown_dtypes = [s for s in selected_dtypes if s not in known_dtypes]
        if unknown_dtypes:
            console.print(f"[red]unknown dequant dtype(s): {', '.join(unknown_dtypes)}; available: {', '.join(known_dtypes)}[/red]")
            sys.exit(1)
    known_blocks = [config_id for config_id, _w, _mm, _a in block_configs]
    selected_blocks = None if args.block_configs.strip().lower() == "all" else [s.strip() for s in args.block_configs.split(",") if s.strip()]
    if selected_blocks is not None:
        unknown_blocks = [s for s in selected_blocks if s not in known_blocks]
        if unknown_blocks:
            console.print(f"[red]unknown block config(s): {', '.join(unknown_blocks)}; available: {', '.join(known_blocks)}[/red]")
            sys.exit(1)
    known_variants = [variant_id for variant_id, _cfg in dequant_variant_configs]
    variants_arg = args.dequant_variants.strip().lower()
    if variants_arg == "all":
        selected_variants = list(known_variants)
    elif variants_arg == "none":
        selected_variants = []
    else:
        selected_variants = [s.strip() for s in args.dequant_variants.split(",") if s.strip()]
        unknown_variants = [s for s in selected_variants if s not in known_variants]
        if unknown_variants:
            console.print(f"[red]unknown dequant variant(s): {', '.join(unknown_variants)}; available: {', '.join(known_variants)}, all, none[/red]")
            sys.exit(1)
    sweeps_arg = args.dequant_sweeps.strip().lower()
    if sweeps_arg == "all":
        selected_sweeps = list(all_dequant_sweeps)
    elif sweeps_arg == "none":
        selected_sweeps = []
    else:
        selected_sweeps = [s.strip() for s in args.dequant_sweeps.split(",") if s.strip()]
        unknown_sweeps = [s for s in selected_sweeps if s not in all_dequant_sweeps]
        if unknown_sweeps:
            console.print(f"[red]unknown dequant sweep(s): {', '.join(unknown_sweeps)}; available: {', '.join(all_dequant_sweeps)}, all, none[/red]")
            sys.exit(1)
    backends_arg = args.mm_backends.strip().lower()
    if backends_arg in {"none", ""}:
        selected_mm_backends = []
    elif backends_arg == "all":
        selected_mm_backends = list(all_mm_backends)
    else:
        selected_mm_backends = [s.strip() for s in args.mm_backends.split(",") if s.strip()]
        unknown_backends = [s for s in selected_mm_backends if s not in all_mm_backends]
        if unknown_backends:
            console.print(f"[red]unknown matmul backend(s): {', '.join(unknown_backends)}; available: {', '.join(all_mm_backends)}, all, none[/red]")
            sys.exit(1)
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
    global pending_outputs # pylint: disable=global-statement
    pending_outputs = dict(save=args.save, json=args.json, args=args, sections=sections, selected=selected)
    print_banner(selected, sections, args)
    prep_status, prep_detail = probe_compiled_prep() # must run first: see the cold-start note in probe_compiled_prep
    fp8_result = probe_fp8() if "attention" in sections else None
    weight_dequant_result = probe_weight_dequant_compile() if "dequant" in sections else None
    print_environment(fp8_result, prep_status, prep_detail, weight_dequant_result=weight_dequant_result)
    if not args.skip_checks and "attention" in sections:
        run_correctness()
        if report.get("correctness_aborted_after") and not cuda_context_alive():
            flush_outputs()
            sys.exit(1)
    if args.skip_bench:
        flush_outputs()
        return

    dequant_results = {}
    if "dequant" in sections:
        if free_vram_gb() < 2.0:
            emit(f"[yellow]skipping dequant benchmarks: needs about 2 gb free vram, {free_vram_gb():.1f} gb available[/yellow]")
        else:
            from modules.sdnq.common import check_torch_compile
            if not check_torch_compile():
                # the module-level compiled dequant is a passthrough with the option off; swap in
                # a real compiled variant so the compiled fwd rows measure what enabling it gives
                from modules.sdnq import dequantizer as dequantizer_module
                dequantizer_module.dequantize_weight_compiled = get_compiled_dequantize_weight()
                emit("[yellow]Dequantize using torch.compile is off in the current config: compiled fwd rows are measured with a tool-compiled dequant, matching the webui after enabling it[/yellow]")
            for index, (shape_label, out_features, in_features) in enumerate(dequant_shapes, start=1):
                dequant_results[shape_label] = bench_dequant_shape(shape_label, out_features, in_features, args.iters, args.warmup, position=(index, len(dequant_shapes)), config_timeout=args.config_timeout, selected_dtypes=selected_dtypes)
                torch_device_module.empty_cache()
            variant_results = {}
            if selected_variants:
                first_label, first_out, first_in = dequant_shapes[0]
                variant_results = bench_dequant_variants(first_label, first_out, first_in, dequant_results.get(first_label, {}), selected_dtypes, selected_variants, args.iters, args.warmup, config_timeout=args.config_timeout)
            first_label, first_out, first_in = dequant_shapes[0]
            first_results = dequant_results.get(first_label, {})
            float_mm_results = bench_float_mm_alternatives(first_label, first_out, first_in, first_results, selected_dtypes, args.iters, args.warmup, config_timeout=args.config_timeout)
            if selected_mm_backends:
                bench_mm_backends(first_label, first_out, first_in, selected_dtypes, selected_mm_backends, args.iters, args.warmup, config_timeout=args.config_timeout, rounds=args.mm_rounds)
            sweep_results = {}
            if "groups" in selected_sweeps:
                sweep_results["groups"] = bench_group_sizes(first_label, first_out, first_in, first_results, selected_dtypes, args.iters, args.warmup, config_timeout=args.config_timeout)
            if "svd" in selected_sweeps:
                sweep_results["svd"] = bench_svd_ranks(first_label, first_out, first_in, first_results, selected_dtypes, args.iters, args.warmup, config_timeout=args.config_timeout)
            if "hgroups" in selected_sweeps:
                sweep_results["hgroups"] = bench_hadamard_groups(first_label, first_out, first_in, first_results, selected_dtypes, args.iters, args.warmup, config_timeout=args.config_timeout)
            if "toggles" in selected_sweeps:
                sweep_results["toggles"] = bench_quant_toggles(first_label, first_out, first_in, first_results, selected_dtypes, args.iters, args.warmup, config_timeout=args.config_timeout)
            if "conv" in selected_sweeps:
                sweep_results["conv"] = bench_conv_section(args.iters, args.warmup, config_timeout=args.config_timeout)
            build_dequant_recommendations(dequant_results, weight_dequant_result, variant_results, float_mm_results, sweep_results)

    # the block section runs before the attention section: a broken-compile attention
    # environment disables dynamo globally, which would poison the block compiles
    if "block" in sections:
        if free_vram_gb() < 3.0:
            emit(f"[yellow]skipping block benchmarks: needs about 3 gb free vram, {free_vram_gb():.1f} gb available[/yellow]")
        else:
            bench_block_section(args.iters, args.warmup, config_timeout=args.config_timeout, selected=selected_blocks)

    if "attention" in sections:
        # bench the prep mode the advice points to: compiled, static workaround, or eager
        if prep_status == "failing_dynamic":
            from modules.sdnq.kernels import triton_atten as atten_module
            inner = getattr(atten_module.get_attn_inputs, "_torchdynamo_orig_callable", None)
            atten_module.get_attn_inputs = torch.compile(inner, fullgraph=True, dynamic=False)
            emit("[yellow]dynamic-shape compile is broken here: benchmarking with the dynamic=false workaround applied, numbers match the webui after setting SDNQ_COMPILE_KWARGS='{\"dynamic\": false}'[/yellow]")
        elif prep_status == "failing":
            emit("[yellow]torch compile is broken here: benchmarking with eager input prep, numbers match the webui after disabling Dequantize using torch.compile[/yellow]")
            from modules.sdnq.kernels import triton_atten as atten_module
            inner = getattr(atten_module.get_attn_inputs, "_torchdynamo_orig_callable", None)
            if inner is not None: # swap in the eager prep; a disable toggle raises on torch 2.13+
                atten_module.get_attn_inputs = inner
        all_results = {}
        minimum_vram = {"wan22": 6.0, "wan22-cfg": 12.0, "ltx2": 3.0, "masked": 6.0, "krea2": 8.0}
        for index, preset in enumerate(selected, start=1):
            needed = minimum_vram.get(preset, 2.0)
            if free_vram_gb() < needed:
                emit(f"[yellow]skipping {preset}: needs about {needed:.0f} gb free vram, {free_vram_gb():.1f} gb available[/yellow]")
                continue
            all_results[preset] = bench_shape(preset, args.iters, args.warmup, position=(index, len(selected)), config_timeout=args.config_timeout, fp8_result=fp8_result)
        build_recommendations(all_results, fp8_result, prep_status, block_results=(report.get("block") or {}).get("results"), block_variants=report.get("blocks"))

    if drift_samples:
        report.setdefault("run", {})["drift_sigma"] = run_drift_sigma()
        report["run"]["drift_samples"] = len(drift_samples)
        emit(f"[dim]run drift: sentinel re-measurements moved {run_drift_sigma():.1%} rms across {len(drift_samples)} shape windows; verdicts fold this into their uncertainty[/dim]")
    flush_outputs()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[yellow]interrupted, saving partial results[/yellow]")
        flush_outputs()
        sys.exit(130)
    except Exception as main_error: # pylint: disable=broad-exception-caught
        console.print(f"[red]benchmark aborted: {error_summary(main_error, 200)}[/red]")
        if not cuda_context_alive():
            console.print("[red]the cuda context is corrupted (kernel faults are sticky); partial results saved, a rerun is required to continue[/red]")
        flush_outputs()
        raise
