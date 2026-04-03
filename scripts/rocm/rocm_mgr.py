import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import installer
from modules.logger import log
from modules.json_helpers import readfile, writefile
from modules.shared import opts

from scripts.rocm.rocm_vars import ROCM_ENV_VARS  # pylint: disable=no-name-in-module
from scripts.rocm import rocm_profiles  # pylint: disable=no-name-in-module


CONFIG = Path(os.path.abspath(os.path.join('data', 'rocm.json')))

_cache: Optional[Dict[str, str]] = None  # loaded once, invalidated on save

# Metadata key written into rocm.json to record which architecture profile is active.
# Not an environment variable - always skipped during env application but preserved in the
# saved config so that arch-safety enforcement is consistent across restarts.
_ARCH_KEY = "_rocm_arch"

# Vars that must never appear in the process environment.
#
# _DTYPE_UNSAFE: alter FP16 inference dtype - must be cleared regardless of config
#   MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL  - DEBUG alias: routes all FP16 convs through BF16 exponent math
#   MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL        - API-level alias: same BF16-exponent effect
#   MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM - unstable experimental FP16 path
#   MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16     - changes FP16 WrW atomic accumulation
#
# SOLVER_DISABLED_BY_DEFAULT: every solver known to be incompatible with this runtime
#   (FP32-only, training-only WrW/BWD, fixed-geometry mismatches, XDLOPS/CDNA-only, arch-specific).
#   Actively unsetting these ensures no inherited shell value can re-enable them.
_DTYPE_UNSAFE = {
    "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL",
    "MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16",
}
# _UNSET_VARS: hard-blocked vars that are DELETED from the process env and never written,
# regardless of saved config. Limited to dtype-corrupting vars only.
# IMPORTANT: SOLVER_DISABLED_BY_DEFAULT is intentionally NOT included here.
#   When a solver var is absent (unset) MIOpen still calls IsApplicable() on every
#   conv-find - wasted probing overhead. When a var is explicitly "0" MIOpen skips
#   IsApplicable() immediately. Solver defaults flow through the config loop as "0"
#   (their ROCM_ENV_VARS default is "0") so they are explicitly set to "0" in the env.
_UNSET_VARS = _DTYPE_UNSAFE

# Additional environment vars that must be removed from the process before MIOpen loads.
# These are not MIOpen solver toggles but can corrupt MIOpen's runtime behaviour:
#   HIP_PATH / HIP_PATH_71  - point to the system AMD ROCm install; override the venv-bundled
#                              _rocm_sdk_devel DLLs with a potentially mismatched system version
#   QML_*/QT_*              - QtQuick shader/disk-cache flags leaked from Qt tools; harmless for
#                              PyTorch but can conflict with Gradio's embedded Qt helpers
#   PYENV_VIRTUALENV_DISABLE_PROMPT - pyenv noise that confuses venv detection
_EXTRA_CLEAR_VARS = {
    "HIP_PATH",
    "HIP_PATH_71",
    "PYENV_VIRTUALENV_DISABLE_PROMPT",
    "QML_DISABLE_DISK_CACHE",
    "QML_FORCE_DISK_CACHE",
    "QT_DISABLE_SHADER_DISK_CACHE",
    # PERF_VALS vars are NOT boolean toggles - MIOpen reads them as perf-config strings.
    # If inherited from a parent shell with value "1", MIOpen's GetPerfConfFromEnv parses
    # "1" as a degenerate config and can return dtype=float32 output from FP16 tensors.
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS",
}

# Solvers whose MIOpen IsApplicable() explicitly rejects non-FP32 tensors.
# They are safe to leave enabled in FP32 mode. When the active dtype is FP16 or BF16
# we force them OFF so MIOpen skips the IsApplicable probe entirely - avoids overhead on
# every conv shape find. These are NOT in _UNSET_VARS because they are valid in FP32.
_FP32_ONLY_SOLVERS = {
    "MIOPEN_DEBUG_CONV_FFT",           # FFT convolution - FP32 only (MIOpen source: IsFp32 check)
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3",   # Winograd 3x3 - FP32 only
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD", # Fused Winograd - FP32 only
}


def _resolve_dtype() -> str:
    """Return the resolved active compute dtype: 'FP16', 'BF16', 'FP32', or '' (not yet known).
    Prefers the resolved devices.dtype (post test_fp16/bf16) over the raw opts string."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from modules import devices as _dev  # pylint: disable=import-outside-toplevel
        if _dev.dtype is not None:
            if _dev.dtype == torch.float16:
                return 'FP16'
            if _dev.dtype == torch.bfloat16:
                return 'BF16'
            if _dev.dtype == torch.float32:
                return 'FP32'
    except Exception:
        pass
    try:
        v = getattr(opts, 'cuda_dtype', None)
        if v in ('FP16', 'BF16', 'FP32'):
            return v
    except Exception:
        pass
    return ''


# --- venv helpers ---

def _get_venv() -> str:
    return sys.prefix


def _get_root() -> str:
    from modules.paths import script_path  # pylint: disable=import-outside-toplevel
    return str(script_path)


def _expand_venv(value: str) -> str:
    return value.replace("{VIRTUAL_ENV}", _get_venv()).replace("{ROOT}", _get_root())


def _collapse_venv(value: str) -> str:
    venv = _get_venv()
    root = _get_root()
    if venv and value.startswith(venv):
        return "{VIRTUAL_ENV}" + value[len(venv):]
    if root and value.startswith(root):
        return "{ROOT}" + value[len(root):]
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
        file_existed = CONFIG.exists()
        if file_existed:
            data = readfile(str(CONFIG), lock=True, as_type="dict")
            _cache = data if data else {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
            # Purge unsafe vars from a stale saved config and re-persist only if the file existed.
            # When running without a saved config (first run / after Delete), load_config() must
            # never create the file - that only happens via save_config() on Apply or Apply Profile.
            dirty = {k for k in _cache if k in _UNSET_VARS or (k != _ARCH_KEY and k not in ROCM_ENV_VARS)}
            if dirty:
                _cache = {k: v for k, v in _cache.items() if k not in dirty}
                writefile(_cache, str(CONFIG))
                log.debug(f'ROCm load_config: purged {len(dirty)} stale/unsafe var(s) from saved config')
        else:
            _cache = {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
        log.debug(f'ROCm load_config: path={CONFIG} existed={file_existed} items={len(_cache)}')
    return _cache


def save_config(config: Dict[str, str]) -> None:
    global _cache  # pylint: disable=global-statement
    sanitized = {k: v for k, v in config.items() if k not in _UNSET_VARS}
    # Enforce arch-incompatible solvers to "0" before writing.
    # Prevents malformed edits (UI or JSON hand-edit) from persisting incompatible "1" values.
    arch = sanitized.get(_ARCH_KEY, "")
    unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
    for var in unavailable:
        if var in sanitized and sanitized[var] != "0":
            sanitized[var] = "0"
            log.debug(f'ROCm save_config: clamped arch-incompatible var={var} arch={arch}')
    writefile(sanitized, str(CONFIG))
    _cache = sanitized


def apply_env(config: Optional[Dict[str, str]] = None) -> None:
    if config is None:
        config = load_config()
    for var in _UNSET_VARS | _EXTRA_CLEAR_VARS:
        if var in os.environ:
            del os.environ[var]
    for var, value in config.items():
        if var == _ARCH_KEY:
            continue
        if var in _UNSET_VARS:
            continue
        if var not in ROCM_ENV_VARS:
            continue
        meta = ROCM_ENV_VARS.get(var, {})
        if meta.get("options"):
            value = _dropdown_stored(str(value), meta["options"])
        expanded = _expand_venv(str(value))
        if expanded == "":
            continue
        os.environ[var] = expanded
    # Arch safety net: hard-force all hardware-incompatible vars to "0" in the env.
    # This runs *after* the config loop so it overrides any stale "1" that survived in the JSON.
    # Source of truth: rocm_profiles.UNAVAILABLE[arch] - vars with no supporting hardware.
    arch = config.get(_ARCH_KEY, "")
    unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
    if unavailable:
        for var in unavailable:
            os.environ[var] = "0"
    dtype_str = _resolve_dtype()
    if dtype_str in ('FP16', 'BF16'):
        for var in _FP32_ONLY_SOLVERS:
            os.environ[var] = "0"


def apply_all(names: list, values: list) -> None:
    config = load_config().copy()
    arch = config.get(_ARCH_KEY, "")
    unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
    for name, value in zip(names, values):
        if name not in ROCM_ENV_VARS:
            log.warning(f'ROCm apply_all: unknown variable={name}')
            continue
        # Arch safety net: silently clamp incompatible solvers back to "0".
        # The UI may send the current checkbox state even for greyed-out vars.
        if name in unavailable:
            config[name] = "0"
            continue
        meta = ROCM_ENV_VARS[name]
        if meta["widget"] == "checkbox":
            if value is None:
                pass  # Gradio passed None (component not interacted with) - leave config unchanged
            else:
                config[name] = "1" if value else "0"
        elif meta["widget"] == "radio":
            stored = _dropdown_stored(str(value), meta["options"])
            valid = {v for _, v in meta["options"]} if meta["options"] and isinstance(meta["options"][0], tuple) else set(meta["options"] or [])
            if stored in valid:
                config[name] = stored
            # else: value was None/invalid - leave the existing saved value untouched
        else:
            if meta.get("options"):
                value = _dropdown_stored(str(value), meta["options"])
            config[name] = _collapse_venv(str(value))
    save_config(config)
    apply_env(config)


def reset_defaults() -> None:
    defaults = {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
    # Preserve the active arch key so safety nets survive a defaults reset.
    arch = load_config().get(_ARCH_KEY, "")
    if arch:
        defaults[_ARCH_KEY] = arch
    save_config(defaults)
    apply_env(defaults)
    log.info(f'ROCm reset_defaults: config reset to defaults arch={arch or "(none)"}')


def clear_env() -> None:
    """Remove all managed ROCm vars and known noise vars from os.environ without writing to disk."""
    cleared = 0
    for var in ROCM_ENV_VARS:
        if var in os.environ:
            del os.environ[var]
            cleared += 1
    for var in _UNSET_VARS | _EXTRA_CLEAR_VARS:
        if var in os.environ:
            del os.environ[var]
            cleared += 1
    log.info(f'ROCm clear_env: cleared={cleared}')


def delete_config() -> None:
    """Delete the saved config file, clear all vars, and wipe the MIOpen user DB cache."""
    import shutil  # pylint: disable=import-outside-toplevel
    global _cache  # pylint: disable=global-statement
    clear_env()
    if CONFIG.exists():
        CONFIG.unlink()
        log.info(f'ROCm delete_config: deleted {CONFIG}')
    _cache = None
    # Delete the MIOpen user DB (~/.miopen/db) - stale entries can cause solver mismatches
    miopen_db = Path(os.path.expanduser('~')) / '.miopen' / 'db'
    if miopen_db.exists():
        shutil.rmtree(miopen_db, ignore_errors=True)
        log.info(f'ROCm delete_config: wiped MIOpen user DB at {miopen_db}')
    else:
        log.debug(f'ROCm delete_config: MIOpen user DB not found at {miopen_db} — nothing to wipe')


def apply_profile(name: str) -> None:
    """Merge an architecture profile on top of the current config, then save and apply."""
    profile = rocm_profiles.PROFILES.get(name)
    if profile is None:
        log.warning(f'ROCm apply_profile: unknown profile={name}')
        return
    config = load_config().copy()
    config.update(profile)
    config[_ARCH_KEY] = name  # stamp the active arch so safety nets survive restarts
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


def _extract_db_hash(db_path: Path) -> str:
    """Derive the cache subfolder name from udb.txt filenames.
    e.g. gfx1030_30.HIP.3_5_1_5454e9e2da.udb.txt  →  '3.5.1.5454e9e2da'"""
    for f in db_path.glob("*.HIP.*.udb.txt"):
        m = re.search(r'\.HIP\.([^.]+)\.udb\.txt$', f.name)
        if m:
            return m.group(1).replace("_", ".")
    return ""


def _user_cache_summary(path: Path) -> dict:
    """Return {filename: 'N KB'} for binary cache blobs in the resolved cache path."""
    out = {}
    if not path.exists():
        return out
    for f in sorted(path.iterdir()):
        if f.is_file():
            kb = f.stat().st_size // 1024
            out[f.name] = f"{kb} KB"
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

    # User cache (~/.miopen/cache/<version-hash>)
    cache_base = Path.home() / ".miopen" / "cache"
    db_hash = _extract_db_hash(user_db_path) if user_db_path.exists() else ""
    cache_path = cache_base / db_hash if db_hash else cache_base
    ucache = {"path": str(cache_path), "exists": cache_path.exists()}
    if cache_path.exists():
        cfiles = _user_cache_summary(cache_path)
        if cfiles:
            ucache["files"] = cfiles

    return {
        "rocm":       rocm_section,
        "torch":      torch_section,
        "gpu":        gpu_section,
        "system_db":  sdb,
        "user_db":    udb,
        "user_cache": ucache,
    }


# Apply saved config to os.environ at import time (only when ROCm is present)
if installer.torch_info.get('type', None) == 'rocm':
    if sys.platform == 'win32':
        try:
            apply_env()
        except Exception as _e:
            log.debug(f"[rocm_mgr] Warning: failed to apply env at import: {_e}")
    else:
        log.debug('Skipping ROCm Environment Manager: Currently only Windows is supported.')
