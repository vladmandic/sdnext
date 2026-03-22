import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import installer
from modules.logger import log
from modules.json_helpers import readfile, writefile
from scripts.rocm.rocm_vars import ROCM_ENV_VARS, SOLVER_GROUPS  # pylint: disable=no-name-in-module
from scripts.rocm import rocm_profiles  # pylint: disable=no-name-in-module


def _check_rocm() -> bool:
    try:
        from modules import shared
        if getattr(shared.cmd_opts, 'use_rocm', False):
            return True
    except Exception:
        pass
    try:
        if installer.torch_info.get('type') == 'rocm':
            return True
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return True
    except Exception:
        pass
    return False


is_rocm = _check_rocm()


CONFIG = Path(os.path.abspath(os.path.join('data', 'rocm-config.json')))

_cache: Optional[Dict[str, str]] = None  # loaded once, invalidated on save

# Vars that must never be set — they interfere with PyTorch dtype handling
_UNSET_VARS = {
    "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16",
}


# --- venv helpers ---

def _get_venv() -> str:
    return os.environ.get("VIRTUAL_ENV", "") or sys.prefix


def _expand_venv(value: str) -> str:
    return value.replace("{VIRTUAL_ENV}", _get_venv())


def _collapse_venv(value: str) -> str:
    venv = _get_venv()
    if venv and value.startswith(venv):
        return "{VIRTUAL_ENV}" + value[len(venv):]
    return value


# --- dropdown helpers ---

def _dropdown_display(stored_val: str, options) -> str:
    if options and isinstance(options[0], tuple):
        return next((label for label, val in options if val == str(stored_val)), str(stored_val))
    return str(stored_val)


def _dropdown_stored(display_val: str, options) -> str:
    if options and isinstance(options[0], tuple):
        return next((val for label, val in options if label == str(display_val)), str(display_val))
    return str(display_val)


def _dropdown_choices(options):
    if options and isinstance(options[0], tuple):
        return [label for label, _ in options]
    return options


# --- config I/O ---

def load_config() -> Dict[str, str]:
    global _cache  # pylint: disable=global-statement
    if _cache is None:
        if CONFIG.exists():
            data = readfile(str(CONFIG), lock=True, as_type="dict")
            _cache = data if data else {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
        else:
            _cache = {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
        log.debug(f'ROCm load_config: path={CONFIG} items={len(_cache)}')
    return _cache


def save_config(config: Dict[str, str]) -> None:
    global _cache  # pylint: disable=global-statement
    writefile(config, str(CONFIG))
    _cache = config


def apply_env(config: Optional[Dict[str, str]] = None) -> None:
    if config is None:
        config = load_config()
    applied = 0
    skipped = 0
    for var in _UNSET_VARS:
        if var in os.environ:
            del os.environ[var]
    for var, value in config.items():
        if var in _UNSET_VARS:
            skipped += 1
            continue
        expanded = _expand_venv(str(value))
        if expanded == "":
            skipped += 1
            continue
        os.environ[var] = expanded
        applied += 1


def apply_all(names: list, values: list) -> None:
    config = load_config().copy()
    for name, value in zip(names, values):
        if name not in ROCM_ENV_VARS:
            log.warning(f'ROCm apply_all: unknown variable={name}')
            continue
        meta = ROCM_ENV_VARS[name]
        if meta["widget"] == "checkbox":
            if value is None:
                pass  # Gradio passed None (component not interacted with) — leave config unchanged
            else:
                config[name] = "1" if value else "0"
        elif meta["widget"] == "radio":
            stored = _dropdown_stored(str(value), meta["options"])
            valid = {v for _, v in meta["options"]} if meta["options"] and isinstance(meta["options"][0], tuple) else set(meta["options"] or [])
            if stored in valid:
                config[name] = stored
            # else: value was None/invalid — leave the existing saved value untouched
        else:
            config[name] = _collapse_venv(str(value))
    save_config(config)
    apply_env(config)


def reset_defaults() -> None:
    defaults = {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
    save_config(defaults)
    apply_env(defaults)
    log.info('ROCm reset_defaults: config reset to defaults')


def clear_env() -> None:
    """Remove all managed ROCm vars from os.environ without writing to disk."""
    cleared = 0
    for var in ROCM_ENV_VARS:
        if var in os.environ:
            del os.environ[var]
            cleared += 1
    for var in _UNSET_VARS:
        if var in os.environ:
            del os.environ[var]
    log.info(f'ROCm clear_env: cleared={cleared}')


def delete_config() -> None:
    """Delete the saved config file and clear all vars from the environment."""
    global _cache  # pylint: disable=global-statement
    clear_env()
    if CONFIG.exists():
        CONFIG.unlink()
        log.info(f'ROCm delete_config: deleted {CONFIG}')
    _cache = None


def apply_profile(name: str) -> None:
    """Merge an architecture profile on top of the current config, then save and apply."""
    profile = rocm_profiles.PROFILES.get(name)
    if profile is None:
        log.warning(f'ROCm apply_profile: unknown profile={name}')
        return
    config = load_config().copy()
    config.update(profile)
    save_config(config)
    apply_env(config)
    log.info(f'ROCm apply_profile: profile={name} overrides={len(profile)}')


def _hip_version_from_file(db_path: Path) -> str:
    """Parse HIP_VERSION_* keys from .hipVersion in the SDK bin folder."""
    hip_ver_file = db_path / ".hipVersion"
    if not hip_ver_file.exists():
        return ""
    kv = {}
    for line in hip_ver_file.read_text(errors="ignore").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    major = kv.get("HIP_VERSION_MAJOR", "")
    minor = kv.get("HIP_VERSION_MINOR", "")
    patch = kv.get("HIP_VERSION_PATCH", "")
    git   = kv.get("HIP_VERSION_GITHASH", "")
    if major:
        return f"{major}.{minor}.{patch} ({git})"
    return ""


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as _m  # pylint: disable=import-outside-toplevel
        return _m.version(name)
    except Exception:
        return "n/a"


def _db_file_summary(path: Path, patterns: list) -> dict:
    """Return {filename: 'N KB'} for files matching any of the given glob patterns."""
    out = {}
    for pat in patterns:
        for f in sorted(path.glob(pat)):
            kb = f.stat().st_size // 1024
            out[f.name] = f"{kb} KB"
    return out


def _user_db_summary(path: Path) -> dict:
    """Return {filename: 'N KB, M entries'} for user MIOpen DB txt files."""
    out = {}
    for pat in ("*.udb.txt", "*.ufdb.txt"):
        for f in sorted(path.glob(pat)):
            kb = f.stat().st_size // 1024
            try:
                lines = sum(1 for _ in f.open(errors="ignore"))
            except Exception:
                lines = 0
            out[f.name] = f"{kb} KB, {lines} entries"
    return out


def info() -> dict:
    config = load_config()
    db_path = Path(_expand_venv(config.get("MIOPEN_SYSTEM_DB_PATH", "")))

    # --- ROCm / HIP package versions ---
    rocm_pkgs = {}
    for pkg in ("rocm", "rocm-sdk-core", "rocm-sdk-devel"):
        v = _pkg_version(pkg)
        if v != "n/a":
            rocm_pkgs[pkg] = v
    libs_pkg = _pkg_version("rocm-sdk-libraries-gfx103x-dgpu")
    if libs_pkg != "n/a":
        rocm_pkgs["rocm-sdk-libraries (gfx103x)"] = libs_pkg

    hip_ver = _hip_version_from_file(db_path)
    if not hip_ver:
        try:
            import torch  # pylint: disable=import-outside-toplevel
            hip_ver = getattr(torch.version, "hip", "") or ""
        except Exception:
            pass

    rocm_section = {}
    if hip_ver:
        rocm_section["hip_version"] = hip_ver
    rocm_section.update(rocm_pkgs)

    # --- Torch ---
    torch_section = {}
    try:
        import torch  # pylint: disable=import-outside-toplevel
        torch_section["version"] = torch.__version__
        torch_section["hip"] = getattr(torch.version, "hip", None) or "n/a"
    except Exception:
        pass

    # --- GPU ---
    gpu_section = [dict(g) for g in installer.gpu_info]

    # --- System DB ---
    sdb = {"path": str(db_path)}
    if db_path.exists():
        solver_db = _db_file_summary(db_path, ["*.db.txt"])
        find_db   = _db_file_summary(db_path, ["*.HIP.fdb.txt", "*.fdb.txt"])
        kernel_db = _db_file_summary(db_path, ["*.kdb"])
        if solver_db:
            sdb["solver_db"] = solver_db
        if find_db:
            sdb["find_db"] = find_db
        if kernel_db:
            sdb["kernel_db"] = kernel_db
    else:
        sdb["exists"] = False

    # --- User DB (~/.miopen/db) ---
    user_db_path = Path.home() / ".miopen" / "db"
    udb = {"path": str(user_db_path), "exists": user_db_path.exists()}
    if user_db_path.exists():
        ufiles = _user_db_summary(user_db_path)
        if ufiles:
            udb["files"] = ufiles

    return {
        "rocm":       rocm_section,
        "torch":      torch_section,
        "gpu":        gpu_section,
        "system_db":  sdb,
        "user_db":    udb,
    }


# Apply saved config to os.environ at import time (only when ROCm is present)
if is_rocm:
    try:
        apply_env()
    except Exception as _e:
        print(f"[rocm_mgr] Warning: failed to apply env at import: {_e}", file=sys.stderr)
else:
    log.debug('ROCm is not installed — skipping rocm_mgr env apply')
