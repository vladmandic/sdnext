import os
import sys
import sysconfig
from typing import Any, Dict


_BIN_DIR = "bin" if sys.platform == "win32" else "lib"


def _sitepackages_subpath(*parts: str) -> str:
    """Return a {VIRTUAL_ENV}-prefixed path into site-packages using OS-native separators.

    Works on both Windows (Lib/site-packages) and Linux (lib/pythonX.Y/site-packages).
    """
    site_pkgs = sysconfig.get_path('purelib')  # absolute path to site-packages inside the active venv
    rel = os.path.relpath(site_pkgs, sys.prefix)  # platform-correct relative sub-path under venv root
    return os.path.join("{VIRTUAL_ENV}", rel, *parts)


GENERAL_VARS: Dict[str, Dict[str, Any]] = {
    "MIOPEN_FIND_MODE": {
        "default": "2",
        "desc": "MIOpen Find Mode",
        "widget": "dropdown",
        "options": [("1 - NORMAL", "1"), ("2 - FAST", "2"), ("3 - HYBRID", "3"), ("5 - DYNAMIC_HYBRID", "5"), ("6 - TRUST_VERIFY", "6"), ("7 - TRUST_VERIFY_FULL", "7")],
        "restart_required": True,
    },
    "MIOPEN_FIND_ENFORCE": {
        "default": "1",
        "desc": "MIOpen Find Enforce",
        "widget": "dropdown",
        "options": [("1 - NONE", "1"), ("2 - DB_UPDATE", "2"), ("3 - SEARCH", "3"), ("4 - SEARCH_DB_UPDATE", "4"), ("5 - DB_CLEAN", "5")],
        "restart_required": True,
    },
    "MIOPEN_SYSTEM_DB_PATH": {
        # "default": _sitepackages_subpath("{LIBS_PKG}", _BIN_DIR) + os.sep,  # auto-path disabled; kept for troubleshooting.
        "default": "",
        "desc": "MIOpen system path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "ROCBLAS_TENSILE_LIBPATH": {
        # "default": _sitepackages_subpath("{LIBS_PKG}", _BIN_DIR, "rocblas", "library"),  # auto-path disabled; kept for troubleshooting.
        "default": "",
        "desc": "rocBLAS Tensile Library Path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "HIPBLASLT_TENSILE_LIBPATH": {
        # "default": _sitepackages_subpath("{LIBS_PKG}", _BIN_DIR, "hipblaslt", "library"),  # auto-path disabled; kept for troubleshooting.
        "default": "",
        "desc": "hipBLASLt Tensile Library Path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": {
        "default": "0",
        "desc": "Enable rocBLAS kernels in PyTorch TunableOp",
        "widget": "dropdown",
        "options": [("0 - Disabled", "0"), ("1 - Enabled", "1")],
        "restart_required": True,
    },
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": {
        "default": "0",
        "desc": "Enable hipBLASLt kernels in PyTorch TunableOp",
        "widget": "dropdown",
        "options": [("0 - Disabled", "0"), ("1 - Enabled", "1")],
        "restart_required": True,
    },
    "MIOPEN_LOG_LEVEL": {
        "default": "0",
        "desc": "MIOpen log verbosity level",
        "widget": "dropdown",
        "options": [("0 - Default", "0"), ("1 - Quiet", "1"), ("3 - Error", "3"), ("4 - Warning", "4"), ("5 - Info", "5"), ("6 - Detail", "6"), ("7 - Trace", "7")],
        "restart_required": False,
    },
    "MIOPEN_DEBUG_ENABLE": {
        "default": "0",
        "desc": "Enable MIOpen logging",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": False,
    },
    "MIOPEN_SEARCH_CUTOFF": {
        "default": "0",
        "desc": "Enable early termination of suboptimal searches",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": True,
    },
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": {
        "default": "0",
        "desc": "Deterministic convolutions",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": False,
    },
    "MIOPEN_CONVOLUTION_MAX_WORKSPACE": {
        "default": "1073741824",
        "desc": "MIOpen convolutions: max workspace (bytes; 1 GB)",
        "widget": "textbox",
        "options": None,
        "restart_required": False,
    },
}

# Solver controls are currently hidden. Keep their metadata above for reference, but do not
# expose or apply it through the active ROCm configuration registry.
ROCM_ENV_VARS: Dict[str, Dict[str, Any]] = {}
ROCM_ENV_VARS.update(GENERAL_VARS)

MIOPEN_LOGGING_VARS = {"MIOPEN_LOG_LEVEL", "MIOPEN_DEBUG_ENABLE"}


def apply_rocm_path_toggles(config_dict: Dict[str, str]) -> Dict[str, str]:
    """Automatically adjusts TunableOp flags based on which library path is configured."""
    rocblas_path = config_dict.get("ROCBLAS_TENSILE_LIBPATH", "").strip()
    hipblaslt_path = config_dict.get("HIPBLASLT_TENSILE_LIBPATH", "").strip()

    if rocblas_path and not hipblaslt_path:
        # User is pointing to a rocblas path -> force-disable hipblaslt tuner
        config_dict["PYTORCH_TUNABLEOP_ROCBLAS_ENABLED"] = "1"
        config_dict["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    elif hipblaslt_path and not rocblas_path:
        # User is pointing to a hipblaslt path -> force-disable rocblas tuner
        config_dict["PYTORCH_TUNABLEOP_ROCBLAS_ENABLED"] = "0"
        config_dict["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "1"

    return config_dict
