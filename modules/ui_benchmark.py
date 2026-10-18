import html
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading

import gradio as gr


_benchmark_lock = threading.Lock()
_benchmark_process = None
_benchmark_running = False
_benchmark_stop_requested = False
_stop_notice = "Stop requested; waiting for the benchmark to save partial results."


def _as_csv(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _render_log(lines, stopped=False):
    rendered = "<br>".join(html.escape(line) for line in lines)
    if stopped:
        notice = html.escape(_stop_notice)
        rendered = f"{rendered}<br>{notice}" if rendered else notice
    return rendered


def benchmark_start(sections, dtypes, variants, configs, block_configs, block_geometries, shapes, iters, warmup, save_path, json_path, skip_checks, skip_bench):
    lines = []
    res = {}
    global _benchmark_running, _benchmark_stop_requested, _benchmark_process  # pylint: disable=global-statement

    with _benchmark_lock:
        already_running = _benchmark_running
        if not already_running:
            _benchmark_running = True
            _benchmark_stop_requested = False
    if already_running:
        yield "A benchmark is already running.", res
        return

    process = None
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        benchmark_path = os.path.join(repo_root, "cli", "sdnq-attention-benchmark.py")
        if json_path == "auto":
            json_path = os.path.join(tempfile.gettempdir(), "sdnq_benchmark_results.json")
        command = [
            sys.executable,
            benchmark_path,
            "--sections", _as_csv(sections),
            "--dequant-dtypes", _as_csv(dtypes),
            "--dequant-variants", _as_csv(variants),
            "--configs", _as_csv(configs),
            "--block-configs", _as_csv(block_configs),
            "--block-geometries", _as_csv(block_geometries),
            "--shapes", _as_csv(shapes),
            "--save", save_path,
            "--json", json_path,
        ]
        if iters is not None:
            command.extend(["--iters", str(int(iters))])
        if warmup is not None:
            command.extend(["--warmup", str(int(warmup))])
        if skip_checks:
            command.append("--skip-checks")
        if skip_bench:
            command.append("--skip-bench")

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        with process:
            with _benchmark_lock:
                _benchmark_process = process
                stop_requested = _benchmark_stop_requested
            if stop_requested:
                process.send_signal(signal.SIGINT)

            stdout = process.stdout
            if stdout is None:
                raise RuntimeError("Benchmark stdout pipe was not created")
            for line in stdout:
                lines.append(line.rstrip("\r\n"))
                with _benchmark_lock:
                    stop_requested = _benchmark_stop_requested
                yield _render_log(lines, stop_requested), res

            return_code = process.wait()
            if return_code:
                lines.append(f"Benchmark exited with status {return_code}.")
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as result_file:
                        res = json.load(result_file)
                except (OSError, ValueError) as error:
                    lines.append(f"Could not read benchmark results: {error}")
            else:
                lines.append("Benchmark did not produce a JSON results file.")

        with _benchmark_lock:
            stop_requested = _benchmark_stop_requested
        yield _render_log(lines, stop_requested), res
    except Exception as error:  # pylint: disable=broad-exception-caught
        lines.append(f"Could not run benchmark: {error}")
        with _benchmark_lock:
            stop_requested = _benchmark_stop_requested
        yield _render_log(lines, stop_requested), res
    finally:
        with _benchmark_lock:
            _benchmark_process = None
            _benchmark_running = False
            _benchmark_stop_requested = False


def benchmark_stop(log):
    global _benchmark_stop_requested  # pylint: disable=global-statement
    with _benchmark_lock:
        process = _benchmark_process
        running = _benchmark_running
        if running:
            _benchmark_stop_requested = True

    if not running:
        return f"{log}<br>No benchmark is running." if log else "No benchmark is running."
    if process is not None and process.poll() is None:
        try:
            process.send_signal(signal.SIGINT)
        except OSError as error:
            return f"{log}<br>Could not interrupt benchmark: {html.escape(str(error))}"
    if _stop_notice in html.unescape(log or ""):
        return log
    notice = html.escape(_stop_notice)
    return f"{log}<br>{notice}" if log else notice


def create_ui():
    gr.HTML('<h3>Benchmark</h3>')
    with gr.Row():
        btn_start = gr.Button("Run benchmark")
        btn_stop = gr.Button("Stop benchmark")
    with gr.Row():
        sections = gr.Dropdown(label="Benchmark sections", choices=['attention', 'dequant', 'block'], value=['attention', 'dequant', 'block'], multiselect=True)
        skip_checks = gr.Checkbox(label="Skip checks", value=False)
        skip_bench = gr.Checkbox(label="Skip bench", value=False)
    with gr.Row():
        dtypes = gr.Dropdown(label="Data types", choices=['all', 'int8', 'uint8', 'int6', 'uint6', 'int4', 'uint4', 'int2', 'uint2', 'fp8', 'fp8sdnq', 'nvfp4'], value=['all'], multiselect=True)
        variants = gr.Dropdown(label="Quantization variants", choices=['all', 'none', 'hadamard', 'svd', 'svd+hadamard'], value='all', multiselect=False)
    with gr.Row():
        configs = gr.Dropdown(label="Attention configs", choices=['all', 'base', 'sage', 'sagefp16', 'amdflash', 'flex', 'flex-sparse100', 'flex-sparse50', 'flex-sparse30', 'flex-sparse15', 'flex-radial30', 'noquant', 'int8', 'int8-sparse100', 'int8-sparse50', 'int8-sparse30', 'int8-sparse15', 'int8-radial30', 'smooth', 'hadamard', 'smooth_hadamard', 'fp16pv', 'fp16pv-accum', 'int8pv', 'fp8pv', 'pvaccum', 'full', 'fp16qk', 'fp16full', 'fp16full-accum', 'fp8qk', 'fp8full'], value=['all'], multiselect=True)
        block_configs = gr.Dropdown(label="Block configs", choices=['all', 'bf16', 'bf16-atten', 'bf16-sage', 'bf16-sagefp16', 'int8', 'int8-mm', 'int8-mm-atten', 'int8-mm-smooth', 'int8-mm-hadamard', 'int8-mm-atten-full', 'int8-mm-pvaccum', 'int8-mm-fp16accum', 'int8-mm-sage', 'int8-mm-sagefp16', 'int6', 'uint4-mm', 'uint4-mm-atten', 'fp8sdnq', 'fp8sdnq-atten', 'nvfp4', 'nvfp4-atten'], value=['all'], multiselect=True)
    with gr.Row():
        shapes = gr.Dropdown(label="Model shapes", choices=['all', 'sd15', 'sdxl', 'sdxl-cross', 'qwen3-te', 'anima', 'flux2', 'krea2', 'wan22', 'wan22-cfg', 'ltx2', 'h3', 'h3-long', 'masked', 'gate-2k', 'gate-4k', 'gate-8k', 'gate-16k', 'gate-32k', 'gate-64k'], value=['sdxl'], multiselect=True)
        block_geometries = gr.Dropdown(label="Model geometries", choices=['flux1', 'krea2', 'h3'], value=['flux1', 'krea2'], multiselect=True)
    with gr.Row(visible=False):
        iters = gr.Number(label="Total iterations", value=12)
        warmup = gr.Number(label="Warmup iterations", value=4)
    with gr.Row():
        save_path = gr.Textbox(label="Save log", value="none")
        json_path = gr.Textbox(label="Save data", value="auto")

    with gr.Row():
        gr.HTML('<h3>Benchmark log</h3>')
    with gr.Row():
        log = gr.HTML('', show_label=False, elem_classes='benchmark-log')
    with gr.Row():
        gr.HTML('<h3>Benchmark results</h3>')
    with gr.Row():
        res = gr.JSON({}, show_label=False, elem_classes='benchmark-log')

    btn_start.click(fn=benchmark_start,
                  inputs=[sections, dtypes, variants, configs, block_configs, block_geometries, shapes, iters, warmup, save_path, json_path, skip_checks, skip_bench],
                  outputs=[log, res],
                 )
    btn_stop.click(fn=benchmark_stop, inputs=[log], outputs=[log], queue=False)
