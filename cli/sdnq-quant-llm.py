#!/usr/bin/env python
# SDNQ LLM quantization tool

from __future__ import annotations

import os
import sys
import time
import argparse
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path

from rich.console import Console
from rich.table import Table

import torch
import transformers


os.environ.setdefault('SDNQ_USE_OPENVINO_MM', '0')
ROOT = Path(__file__).resolve().parents[1]
SDNQ_SRC = ROOT / "extensions-builtin" / "sdnq" / "src"
if SDNQ_SRC.is_dir():
    sys.path.insert(0, str(SDNQ_SRC))
import sdnq

console = Console()
QUANT_CHOICES = (
    "int32", "int16", "int8", *(f"int{bits}" for bits in range(7, 1, -1)),
    "uint32", "uint16", "uint8", *(f"uint{bits}" for bits in range(7, 0, -1)),
    "float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2",
    *(f"fp{bits}" for bits in range(16, 0, -1)),
    *(f"ufp{bits}" for bits in range(16, 0, -1)),
)


def model_size(model: torch.nn.Module) -> int:
    """Count unique tensor storage, including buffers and avoiding tied weights."""
    storages = set()
    total = 0
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if storage.nbytes() and key not in storages:
            storages.add(key)
            total += storage.nbytes()
    return total


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def quantized_layer_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in model.modules():
        dequantizer = getattr(module, "sdnq_dequantizer", None)
        if dequantizer is not None:
            dtype = dequantizer.weights_dtype
            counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def get_stale_weight_files(existing_files: list[str], current_files: set[str]) -> list[str]:
    weight_patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*-of-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*-of-*.bin",
    )
    return [
        file_name
        for file_name in existing_files
        if file_name not in current_files
        and "/" not in file_name
        and any(fnmatchcase(file_name, pattern) for pattern in weight_patterns)
    ]


def main(options: argparse.Namespace) -> None:
    try:
        processor = transformers.AutoProcessor.from_pretrained(options.repo, trust_remote_code=True)
        console.print(f"Processor(cls={processor.__class__.__name__})")
    except (OSError, ValueError):
        processor = None
        console.print("Processor(cls=None)")
    has_image_processor = processor is not None and getattr(processor, "image_processor", None) is not None
    if has_image_processor:
        model_cls = transformers.AutoModelForMultimodalLM
    else:
        model_cls = transformers.AutoModelForCausalLM
    console.print(f"Load(repo='{options.repo}' loader={model_cls.__name__})")
    model = model_cls.from_pretrained(
        options.repo,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    console.print(f"Model(cls={model.__class__.__name__})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.repo, trust_remote_code=True)
    console.print(f"Tokenizer(cls={tokenizer.__class__.__name__})")
    size_before = model_size(model)
    quantization_device = torch.device("cuda") if torch.cuda.is_available() else None

    quant_config = sdnq.SDNQConfig(
        weights_dtype=options.quant,
        use_svd=options.method == "svd",
        use_hadamard=options.method == "hadamard",
        use_codebook=options.method == "codebook",
        use_dynamic_quantization=options.dynamic,
        dynamic_loss_threshold=options.loss,
        quantization_device=quantization_device,
    )
    console.print(f'SDNQConfig({quant_config})')
    model = sdnq.sdnq_post_load_quant(model, quantization_config=quant_config, torch_dtype=model.dtype)
    size_after = model_size(model)
    saved = size_before - size_after

    table = Table(title="SDNQ Quantization Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Model", options.repo)
    table.add_row("Source dtype", str(model.dtype))
    table.add_row("Parameters", f"{sum(parameter.numel() for parameter in model.parameters()):,}")
    table.add_row("Method", options.method)
    table.add_row("Quant dtype", options.quant)
    table.add_row("Dynamic", "enabled" if options.dynamic else "disabled")
    table.add_row("Loss threshold", str(options.loss) if options.loss is not None else "auto")
    table.add_row("Quantization device", str(quantization_device or "model device"))
    table.add_row("Storage before", format_size(size_before))
    table.add_row("Storage after", format_size(size_after))
    table.add_row("Reduction", f"{saved / size_before * 100:.1f}%" if size_before else "n/a")
    for dtype, count in sorted(quantized_layer_counts(model).items()):
        table.add_row(f"Layers ({dtype})", f"{count:,}")
    table.add_row("Excluded", f"{len(quant_config.modules_to_not_convert):,}")
    console.print(table)

    if options.output:
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory(prefix="quant_llm_", dir=ROOT / "tmp") as temp_dir:
            console.print(f'Save(temp="{temp_dir}")')
            sdnq.save_sdnq_model(model, temp_dir, max_shard_size="4GB", sdnq_config=quant_config)
            tokenizer.save_pretrained(temp_dir)
            if processor is not None:
                processor.save_pretrained(temp_dir)
            api = HfApi()
            console.print(f'Upload(repo="{options.output}")')
            api.create_repo(options.output, repo_type="model", private=options.private, exist_ok=options.overwrite)
            current_files = {
                path.relative_to(temp_dir).as_posix()
                for path in Path(temp_dir).rglob("*")
                if path.is_file()
            }
            stale_weights = get_stale_weight_files(
                api.list_repo_files(options.output, repo_type="model"),
                current_files,
            )
            api.upload_folder(
                repo_id=options.output,
                repo_type="model",
                folder_path=temp_dir,
                commit_message=f"Upload SDNQ {options.quant} quantized model",
                delete_patterns=stale_weights,
            )


if __name__ == "__main__":
    console.print(f"SDNQLLMTool(torch=={torch.__version__}, transformers=={transformers.__version__}, sdnq=={sdnq.__version__})")
    parser = argparse.ArgumentParser(description="SDNQ LLM quantization tool")
    parser.add_argument("repo", help="Source Hugging Face model repository")
    parser.add_argument("--output", help="Destination Hugging Face repository; omit to only report stats")
    parser.add_argument("--private", action="store_true", help="Create the destination repository as private")
    parser.add_argument("--overwrite", action="store_true", help="Allow updating an existing destination repository")
    parser.add_argument("--method", choices=("none", "svd", "hadamard", "codebook"), default="hadamard")
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="uint4", help="SDNQ weights_dtype")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic quantization")
    parser.add_argument("--loss", type=float, default=None, help="Dynamic quantization loss threshold (default: SDNQ auto)")
    args = parser.parse_args()
    console.print(f'Args({args})')
    t0 = time.perf_counter()
    main(args)
    t1 = time.perf_counter()
    console.print(f"Done(time={t1 - t0:.2f})")
