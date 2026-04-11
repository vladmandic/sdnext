#!/usr/bin/env python

import argparse
import html
import json
import math
import os
import re
from typing import Any

import huggingface_hub as hf
from huggingface_hub.utils import disable_progress_bars


WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
)

COMPONENT_CONFIG_FILES = (
    "config.json",
    "model_config.json",
)

MODEL_CONFIG_FILES = (
    "models_index.json",
    "model_index.json",
)

ALIASES = {
    "stabilityai": "Stability AI",
    "stability-ai": "Stability AI",
    "runwayml": "Runway",
    "black-forest-labs": "Black Forest Labs",
    "blackforestlabs": "Black Forest Labs",
    "openai": "OpenAI",
    "hunyuanvideo-community": "HunyuanVideo Community",
}


def normalize_author(repo_id: str, model_info: Any) -> str | None:
    owner = repo_id.split("/", 1)[0] if "/" in repo_id else None
    card_data = getattr(model_info, "card_data", None) or {}
    card_author = card_data.get("model_creator") or card_data.get("author") or card_data.get("publisher")
    source = card_author or owner
    if not source:
        return None

    text = str(source).strip()
    lowered = text.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]

    # Fallback: replace separators and title-case while preserving known acronyms.
    text = re.sub(r"[_-]+", " ", text)
    words = [w for w in text.split(" ") if w]
    normalized = " ".join(words)
    return normalized.title().replace("Ai", "AI").replace("Ml", "ML").replace("Openai", "OpenAI")


def format_millions(value: int | None, suffix: str) -> str | None:
    if value is None:
        return None
    millions = value / 1_000_000
    rounded = round(millions, 1)
    if math.isclose(rounded, round(rounded), rel_tol=0, abs_tol=1e-9):
        return f"{int(round(rounded))}{suffix}"
    return f"{rounded}{suffix}"


def sanitize_plain_text(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = text
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"!\[(.*?)\]\((.*?)\)", " ", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"_(.*?)_", r"\1", cleaned)
    cleaned = re.sub(r"^[#>\-\*\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def to_printable_ascii(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    # Remove non-ASCII (including emoji/unicode) and control characters.
    ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
    ascii_text = "".join(ch for ch in ascii_text if 32 <= ord(ch) <= 126 or ch in ("\t", " "))
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text or None


def is_navigation_like_text(text: str | None) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    low = text.lower()
    if text.count("|") >= 3:
        return True
    if low.count("http") >= 2:
        return True
    tokens = ("hugging face", "github", "discord", "wechat", "blog", "demo", "modelscope")
    hits = sum(1 for token in tokens if token in low)
    return hits >= 3


def build_error(repo_id: str, code: str, message: str, matches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
        "repo_id": repo_id,
    }
    if matches is not None:
        payload["matches"] = matches
    return payload


def load_token() -> str | None:
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.json")
    if not os.path.isfile(secrets_path):
        return None
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token = data.get("huggingface_token")
        return token if isinstance(token, str) and token.strip() else None
    except Exception:
        return None


def load_repo_json(repo_id: str, filename: str, token: str | None) -> dict[str, Any] | None:
    try:
        path = hf.hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    except Exception:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_model_card_text(repo_id: str, token: str | None) -> str | None:
    try:
        card = hf.ModelCard.load(repo_id, token=token)
    except Exception:
        return None
    text = getattr(card, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    content = getattr(card, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return None


def get_model_index(repo_id: str, token: str | None) -> dict[str, Any] | None:
    # Per task requirements, inspect models_index.json first.
    for filename in MODEL_CONFIG_FILES:
        data = load_repo_json(repo_id, filename, token)
        if isinstance(data, dict):
            return data
    return None


def get_repo_files_map(model_info: Any) -> dict[str, int]:
    files: dict[str, int] = {}
    for sibling in getattr(model_info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if isinstance(name, str):
            files[name] = int(size) if isinstance(size, int) else 0
    return files


def is_weight_file(filename: str) -> bool:
    low = filename.lower()
    return low.endswith(WEIGHT_EXTENSIONS)


def discover_components(model_index: dict[str, Any] | None, files_map: dict[str, int]) -> dict[str, Any]:
    components: dict[str, Any] = {
        "mains": [],
        "text_encoders": [],
        "ae": None,
    }

    if isinstance(model_index, dict):
        keys = list(model_index.keys())
        main_keys = sorted([k for k in keys if re.fullmatch(r"(transformer|unet)(_\d+)?", k or "")])
        components["mains"] = main_keys

        text_keys = sorted([k for k in keys if re.fullmatch(r"text_encoder(_\d+)?", k or "")])
        components["text_encoders"] = text_keys

        if "vae" in keys:
            components["ae"] = "vae"

    top_dirs = {f.split("/", 1)[0] for f in files_map if "/" in f}

    if not components["mains"]:
        components["mains"] = sorted([d for d in top_dirs if re.fullmatch(r"(transformer|unet)(_\d+)?", d or "")])

    if not components["text_encoders"]:
        components["text_encoders"] = sorted([d for d in top_dirs if re.fullmatch(r"text_encoder(_\d+)?", d or "")])

    if components["ae"] is None and "vae" in top_dirs:
        components["ae"] = "vae"

    return components


def component_weight_files(component: str | None, files_map: dict[str, int]) -> list[str]:
    if not component:
        return []
    prefix = f"{component}/"
    return [f for f in files_map if f.startswith(prefix) and is_weight_file(f)]


def component_config(component: str | None, repo_id: str, token: str | None) -> dict[str, Any] | None:
    if not component:
        return None
    for config_name in COMPONENT_CONFIG_FILES:
        data = load_repo_json(repo_id, f"{component}/{config_name}", token)
        if isinstance(data, dict):
            return data
    return None


def arch_from_config(config: dict[str, Any] | None, component_type: str = "main") -> str | None:
    if not isinstance(config, dict):
        return None
    candidates: list[str] = []
    class_name = config.get("_class_name")
    if isinstance(class_name, str):
        candidates.append(class_name)
    arches = config.get("architectures")
    if isinstance(arches, list) and len(arches) > 0 and isinstance(arches[0], str):
        candidates.append(arches[0])

    for candidate in candidates:
        low = candidate.lower()
        if component_type == "te":
            return candidate
        if component_type == "ae":
            if "autoencoder" in low or "vae" in low:
                return "AutoEncoderKL"
            return candidate
        if "mmdit" in low or "sd3transformer2dmodel" in low:
            return "MMDiT"
        if "unet" in low:
            return "UNet"
        if "dit" in low:
            return "DiT"
        if "autoencoder" in low:
            return "AutoEncoderKL"
        if "clip" in low:
            return candidate
        return candidate

    # Config-key heuristics fallback.
    keys = {str(k).lower() for k in config.keys()}
    if component_type == "main":
        if "joint_attention_dim" in keys and "caption_projection_dim" in keys:
            return "MMDiT"
        if any("down_block_types" in k for k in keys) and any("up_block_types" in k for k in keys):
            return "UNet"
        if any("transformer" in k for k in keys):
            return "DiT"

    if component_type == "ae":
        if "down_block_types" in keys and "up_block_types" in keys:
            return "AutoEncoderKL"
    return None


def class_from_config(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    class_name = config.get("_class_name")
    if isinstance(class_name, str) and class_name.strip():
        return class_name.strip()
    arches = config.get("architectures")
    if isinstance(arches, list) and len(arches) > 0 and isinstance(arches[0], str) and arches[0].strip():
        return arches[0].strip()
    return None


def class_from_model_index(model_index: dict[str, Any] | None) -> str | None:
    if not isinstance(model_index, dict):
        return None
    class_name = model_index.get("_class_name")
    if isinstance(class_name, str) and class_name.strip():
        return class_name.strip()
    arches = model_index.get("architectures")
    if isinstance(arches, list) and len(arches) > 0 and isinstance(arches[0], str) and arches[0].strip():
        return arches[0].strip()
    return None


def unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def is_excluded_tag(tag: str) -> bool:
    low = tag.lower().strip()
    if low.startswith("license:") or low.startswith("region:") or low.startswith("language:"):
        return True
    if low.startswith("diffusers") or low.startswith("safetensors"):
        return True
    # Common language code tags such as "en", "fr", "en-us".
    if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", low):
        return True
    return False


def dedupe_tags_against_fields(tags: list[str], data: dict[str, Any]) -> list[str]:
    explicit_values: set[str] = set()

    for key, value in data.items():
        if key == "tags" or value is None:
            continue
        if isinstance(value, str):
            parts = [p.strip().lower() for p in value.split(",")]
            explicit_values.update([p for p in parts if p])
        elif isinstance(value, bool):
            explicit_values.add(str(value).lower())

    filtered: list[str] = []
    for tag in tags:
        low = tag.strip().lower()
        if low in explicit_values:
            continue
        filtered.append(tag)
    return filtered


def count_safetensors_params(fs: hf.HfFileSystem, repo_id: str, filename: str) -> int | None:
    path = f"{repo_id}/{filename}"
    try:
        with fs.open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None
            header_len = int.from_bytes(header_len_bytes, "little")
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        return None

    total = 0
    for _, meta in header.items():
        if not isinstance(meta, dict):
            continue
        shape = meta.get("shape")
        if not isinstance(shape, list):
            continue
        try:
            tensor_params = 1
            for dim in shape:
                tensor_params *= int(dim)
            total += tensor_params
        except Exception:
            return None
    return total


def count_component_params(fs: hf.HfFileSystem, repo_id: str, files: list[str]) -> int | None:
    if len(files) == 0:
        return None
    total = 0
    inspectable = False
    for filename in files:
        if filename.lower().endswith(".safetensors"):
            inspectable = True
            params = count_safetensors_params(fs, repo_id, filename)
            if params is None:
                return None
            total += params
        else:
            # Non-safetensors are not reliably inspectable without full tensor load.
            return None
    return total if inspectable else None


def sum_sizes(files: list[str], files_map: dict[str, int]) -> int | None:
    if len(files) == 0:
        return None
    return sum(files_map.get(f, 0) for f in files)


def extract_description(model_info: Any, model_card_text: str | None = None) -> str | None:
    card_data = getattr(model_info, "card_data", None) or {}
    description = card_data.get("description") if isinstance(card_data, dict) else None
    if isinstance(description, str) and description.strip():
        plain = sanitize_plain_text(description.strip().splitlines()[0])
        if plain and not is_navigation_like_text(plain):
            plain = to_printable_ascii(plain)
            return plain[:300] if plain else None

    # Fallback to model card markdown summary if available.
    card_content = model_card_text or getattr(model_info, "card_content", None)
    if isinstance(card_content, str):
        lines = card_content.splitlines()
        in_frontmatter = False
        in_code_block = False
        started = False
        first_header: str | None = None
        cleaned: list[str] = []

        for line in lines:
            raw = line.rstrip()
            stripped = raw.strip()

            # Ignore YAML frontmatter block.
            if stripped == "---" and not started:
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                continue

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Skip common non-description lines.
            if not stripped:
                if cleaned:
                    break
                continue
            if stripped.startswith("#"):
                # Store first heading as fallback, but prefer body paragraph.
                if first_header is None:
                    first_header = stripped.lstrip("#").strip()
                started = True
                continue
            if stripped.startswith("[") and "](" in stripped:
                continue
            if stripped.startswith("!"):
                continue
            if len(re.findall(r"\[[^\]]+\]\([^)]+\)", stripped)) >= 2:
                continue

            started = True
            text = stripped
            text = sanitize_plain_text(text)
            if text and not is_navigation_like_text(text):
                cleaned.append(text)

        if cleaned:
            summary = sanitize_plain_text(" ".join(cleaned))
            if summary is None:
                return None
            summary = to_printable_ascii(summary)
            return summary[:300] if summary else None

        if first_header:
            plain = sanitize_plain_text(first_header)
            if plain and not is_navigation_like_text(plain):
                plain = to_printable_ascii(plain)
                return plain[:300] if plain else None
    return None


def extract_name(model_info: Any, repo_id: str) -> str | None:
    card_data = getattr(model_info, "card_data", None) or {}
    for key in ("model_name", "name", "title"):
        value = card_data.get(key) if isinstance(card_data, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return repo_id.split("/", 1)[1] if "/" in repo_id else repo_id


def extract_version(model_info: Any, repo_id: str) -> str | None:
    card_data = getattr(model_info, "card_data", None) or {}
    for key in ("version", "model_version", "revision"):
        value = card_data.get(key) if isinstance(card_data, dict) else None
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Heuristic fallback from repo name.
    model_name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id

    # Embedded dotted versions (example: Wan2.2-T2V-A14B-Diffusers -> 2.2).
    match = re.search(r"(\d+\.\d+(?:\.\d+)*)", model_name, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Prefer long numeric build-like versions (example: 2512).
    match = re.search(r"(?:^|[-_\s])v?(\d{3,})(?:$|[-_\s])", model_name, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Dotted semantic versions (example: 1.5 or 2.0.1).
    match = re.search(r"(?:^|[-_\s])v?(\d+(?:\.\d+)+)(?:$|[-_\s])", model_name, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Hyphen/underscore separated numeric versions (example: v1-5 -> 1.5).
    match = re.search(r"(?:^|[-_\s])v?(\d+)[-_](\d+)(?:[-_](\d+))?(?:$|[-_\s])", model_name, flags=re.IGNORECASE)
    if match:
        parts = [match.group(1), match.group(2)]
        if match.group(3):
            parts.append(match.group(3))
        return ".".join(parts)

    # Single-number versions as final fallback.
    match = re.search(r"(?:^|[-_\s])v?(\d+)(?:$|[-_\s])", model_name, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_license(model_info: Any) -> str | None:
    card_data = getattr(model_info, "card_data", None) or {}

    if isinstance(card_data, dict):
        license_name = card_data.get("license_name")
        license_value = card_data.get("license")
    else:
        license_name = getattr(card_data, "license_name", None)
        license_value = getattr(card_data, "license", None)

    if isinstance(license_name, str) and license_name.strip():
        return sanitize_plain_text(license_name.strip())

    if isinstance(license_value, str) and license_value.strip() and license_value.strip().lower() != "other":
        return sanitize_plain_text(license_value.strip())

    tags = getattr(model_info, "tags", None) or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            value = tag.split(":", 1)[1].strip()
            if value and value.lower() != "other":
                return sanitize_plain_text(value)

    return None


def release_date_iso(model_info: Any) -> str | None:
    created_at = getattr(model_info, "created_at", None)
    if created_at is None:
        return None
    try:
        return created_at.date().isoformat()
    except Exception:
        try:
            return str(created_at)[:10]
        except Exception:
            return None


def modified_date_iso(model_info: Any) -> str | None:
    last_modified = getattr(model_info, "last_modified", None)
    if last_modified is None:
        return None
    try:
        return last_modified.date().isoformat()
    except Exception:
        try:
            return str(last_modified)[:10]
        except Exception:
            return None


def handle_not_found(repo_id: str, api: hf.HfApi) -> dict[str, Any]:
    try:
        # Edge-case support for "multiple matches" response shape.
        candidates = list(api.list_models(search=repo_id, limit=10, full=False))
    except Exception:
        candidates = []

    matches = []
    for model in candidates:
        model_id = getattr(model, "id", None)
        if not isinstance(model_id, str):
            continue
        if repo_id.lower() in model_id.lower():
            matches.append({"repo_id": model_id, "name": model_id.split("/", 1)[-1]})

    if len(matches) > 1:
        return build_error(
            repo_id,
            "multiple_matches",
            "Multiple matching models found. Please specify a more specific repo id.",
            matches=matches,
        )

    return build_error(repo_id, "repo_not_found", "Model repo id not found or inaccessible.")


def main() -> int:
    disable_progress_bars()
    parser = argparse.ArgumentParser(description="Query Hugging Face model metadata and component stats.")
    parser.add_argument("repo_id", help="Strict Hugging Face repo id in format owner/name")
    args = parser.parse_args()

    repo_id = args.repo_id.strip()
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
        error = build_error(repo_id, "invalid_repo_id", "Expected repo id format: owner/name")
        print(json.dumps(error, indent=2, sort_keys=False))
        return 1

    token = load_token()
    api = hf.HfApi(token=token)

    try:
        model_info = api.model_info(repo_id=repo_id, files_metadata=True)
    except hf.errors.RepositoryNotFoundError:
        print(json.dumps(handle_not_found(repo_id, api), indent=2, sort_keys=False))
        return 1
    except hf.errors.HfHubHTTPError as err:
        status_code = getattr(getattr(err, "response", None), "status_code", None)
        code = "access_denied" if status_code in (401, 403) else "request_failed"
        message = "Access denied while querying repository." if code == "access_denied" else f"Hugging Face request failed: {err}"
        print(json.dumps(build_error(repo_id, code, message), indent=2, sort_keys=False))
        return 1
    except Exception as err:
        print(json.dumps(build_error(repo_id, "request_failed", f"Unexpected error: {err}"), indent=2, sort_keys=False))
        return 1

    files_map = get_repo_files_map(model_info)
    model_card_text = load_model_card_text(repo_id, token)
    model_index = get_model_index(repo_id, token)
    components = discover_components(model_index, files_map)

    main_components = components["mains"]
    text_components = components["text_encoders"]
    ae_component = components["ae"]

    main_files: list[str] = []
    for main_component in main_components:
        main_files.extend(component_weight_files(main_component, files_map))
    te_files: list[str] = []
    for te_component in text_components:
        te_files.extend(component_weight_files(te_component, files_map))
    ae_files = component_weight_files(ae_component, files_map)

    fs = hf.HfFileSystem(token=token)

    model_params_raw = count_component_params(fs, repo_id, main_files)
    te_params_raw = count_component_params(fs, repo_id, te_files)

    model_size_raw = sum_sizes(main_files, files_map)
    te_size_raw = sum_sizes(te_files, files_map)
    ae_size_raw = sum_sizes(ae_files, files_map)

    size_total_raw = None
    available_sizes = [v for v in (model_size_raw, te_size_raw, ae_size_raw) if v is not None]
    if available_sizes:
        size_total_raw = sum(available_sizes)

    main_cfgs = [component_config(main_component, repo_id, token) for main_component in main_components]
    main_component_classes = [class_from_config(cfg) for cfg in main_cfgs]
    main_arches = [arch_from_config(cfg, component_type="main") for cfg in main_cfgs]
    main_dit_entries = [
        cls if isinstance(cls, str) and cls.strip() else arch
        for cls, arch in zip(main_component_classes, main_arches)
    ]
    main_dit_entries = [entry.strip() for entry in main_dit_entries if isinstance(entry, str) and entry.strip()]

    te_arches: list[str] = []
    for te_component in text_components:
        cfg = component_config(te_component, repo_id, token)
        arch = arch_from_config(cfg, component_type="te")
        te_arches.append(arch if arch is not None else te_component)

    ae_cfg = component_config(ae_component, repo_id, token)
    ae_arch = arch_from_config(ae_cfg, component_type="ae")
    model_class = class_from_model_index(model_index)
    if model_class is None:
        first_main_class = next((c for c in main_component_classes if isinstance(c, str) and c.strip()), None)
        model_class = first_main_class

    downloads = getattr(model_info, "downloads", None)
    downloads_int = int(downloads) if isinstance(downloads, int) else None
    pipeline = getattr(model_info, "pipeline_tag", None)
    pipeline_value = str(pipeline) if isinstance(pipeline, str) and pipeline.strip() else None
    gated = getattr(model_info, "gated", None)
    gated_value = gated if isinstance(gated, (bool, str)) else None
    tags_raw = getattr(model_info, "tags", None) or []
    tags = [str(tag) for tag in tags_raw if isinstance(tag, str) and not is_excluded_tag(tag)]

    data = {
        "author": normalize_author(repo_id, model_info),
        "name": extract_name(model_info, repo_id),
        "version": extract_version(model_info, repo_id),
        "description": extract_description(model_info, model_card_text=model_card_text),
        "released": release_date_iso(model_info),
        "modified": modified_date_iso(model_info),
        "license": extract_license(model_info),
        "repo_id": repo_id,
        "pipeline": pipeline_value,
        "gated": gated_value,
        "size": format_millions(size_total_raw, "MB"),
        "class": model_class,
        "dit": ", ".join(main_dit_entries) if len(main_dit_entries) > 0 else None,
        "dit_params": format_millions(model_params_raw, "M"),
        "dit_size": format_millions(model_size_raw, "MB"),
        "te": ", ".join(te_arches) if len(te_arches) > 0 else None,
        "te_params": format_millions(te_params_raw, "M"),
        "te_size": format_millions(te_size_raw, "MB"),
        "ae": ae_arch,
        "downloads": downloads_int,
        "tags": tags,
    }

    data["tags"] = dedupe_tags_against_fields(data["tags"], data)

    output = {
        "ok": True,
        "data": data,
    }
    print(json.dumps(output, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
