#!/usr/bin/env python

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from rich import print # pylint: disable=redefine-builtin


def load_hf_info_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("hf_info", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_repo_id(path: str) -> str:
    if path.startswith("huggingface/"):
        return path[len("huggingface/"):]
    return path.strip()


def parse_size_bytes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def parse_released_date(released: Any) -> str | None:
    if not isinstance(released, str):
        return None
    try:
        date_obj = datetime.fromisoformat(released).date()
        return date_obj.strftime("%Y %B")
    except ValueError:
        return None


def update_entry_fields(entry: dict[str, Any], result: dict[str, Any]) -> list[str]:
    updated_fields: list[str] = []
    data = result.get("data") if isinstance(result.get("data"), dict) else {}

    size_value = parse_size_bytes(data.get("size"))
    if size_value is not None:
        size_gb = round(size_value / 1024 / 1024 / 1024, 2)
        if entry.get("size") != size_gb:
            entry["size"] = size_gb
            updated_fields.append({ 'size': size_gb })

    date_value = parse_released_date(data.get("released"))
    if date_value is not None and entry.get("date") != date_value:
        entry["date"] = date_value
        updated_fields.append({ 'date': date_value })

    gated_value = data.get("gated")
    if gated_value is True:
        if entry.get("gated") is not True:
            entry["gated"] = True
            updated_fields.append({ 'gated': True })
    else:
        if "gated" in entry:
            entry.pop("gated", None)
            updated_fields.append({ 'gated': False })

    return updated_fields


def process_reference_file(file_path: Path, hf_info_module: Any, only_missing: bool = False) -> int:
    print(f"file='{file_path}'")
    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        print(f"{file_path}: invalid file format, expected JSON object")
        return 0

    modified = False
    entries=len(payload)
    updated_entries = 0
    i = 0

    print(f'entries={entries}')
    for key, entry in payload.items():
        i += 1
        if not isinstance(entry, dict):
            continue

        path_value = entry.get("path")
        if not isinstance(path_value, str):
            continue

        if only_missing and entry.get("size") is not None and entry.get("date") is not None:
            continue

        status = ""
        updated_fields: list[str] = []

        if "safetensors" in path_value.lower():
            status = "safetensors"
        else:
            repo_id = normalize_repo_id(path_value)
            result = hf_info_module.search(repo_id)
            if isinstance(result, dict):
                if result.get("status", None) is True:
                    updated_fields = update_entry_fields(entry, result)
                    if updated_fields:
                        modified = True
                        updated_entries += 1
                else:
                    status = result.get("message", "error")
            else:
                status = "error"

        print(f"repo='{key}' path='{path_value}' status='{status}' updated={updated_fields} progress={i}/{entries}")

    if modified:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")

    return updated_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Update data/reference-*.json entries from Hugging Face metadata.")
    parser.add_argument("--data", default="data", help="Path to the reference data directory containing reference-*.json files.")
    parser.add_argument("--missing", action="store_true", help="Only update entries that are missing size or date information.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    hf_info_path = root_dir / "cli" / "hf-info.py"
    if not hf_info_path.exists():
        raise FileNotFoundError(f"hf-info.py not found at {hf_info_path}")

    hf_info_module = load_hf_info_module(hf_info_path)
    reference_dir = root_dir / args.data
    patterns = sorted(reference_dir.glob("reference-*.json"))

    if not patterns:
        print(f"No reference files found in {reference_dir}")
        return 1

    total_updates = 0
    for path in patterns:
        total_updates += process_reference_file(path, hf_info_module, only_missing=args.missing)

    print(f"Updated entries: {total_updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
