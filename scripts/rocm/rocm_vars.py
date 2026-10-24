import os
from typing import Any, Dict

MIOPEN_LOGGING_VARS = {"MIOPEN_LOG_LEVEL", "MIOPEN_DEBUG_ENABLE"}

GENERAL_VARS: Dict[str, Dict[str, Any]] = {
    "MIOPEN_FIND_MODE": {
        "default": "2",
        "desc": "MIOpen Find Mode",
        "widget": "dropdown",
        "options": [
            ("1 - NORMAL", "1"),
            ("2 - FAST", "2"),
            ("3 - HYBRID", "3"),
            ("5 - DYNAMIC_HYBRID", "5"),
            ("6 - TRUST_VERIFY", "6"),
            ("7 - TRUST_VERIFY_FULL", "7"),
        ],
        "restart_required": True,
    },
    "MIOPEN_FIND_ENFORCE": {
        "default": "1",
        "desc": "MIOpen Find Enforce",
        "widget": "dropdown",
        "options": [
            ("1 - NONE", "1"),
            ("2 - DB_UPDATE", "2"),
            ("3 - SEARCH", "3"),
            ("4 - SEARCH_DB_UPDATE", "4"),
            ("5 - DB_CLEAN", "5"),
        ],
        "restart_required": True,
    },
    "MIOPEN_SYSTEM_DB_PATH": {
        "default": "",
        "desc": "MIOpen system path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "ROCBLAS_TENSILE_LIBPATH": {
        "default": "",
        "desc": "rocBLAS Tensile Library Path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "HIPBLASLT_TENSILE_LIBPATH": {
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
    "ROCBLAS_LAYER": {
            "default": "0",
            "desc": "rocBLAS Logging",
            "widget": "dropdown",
            "options": [
                ("0 - Disabled (Hidden)", "0"),
                ("1 - Trace", "1"),
                ("2 - Bench", "2"),
                ("4 - Profile", "4"),
                ("8 - Internal API", "8"),
            ],
            "restart_required": True,
        },
        "HIPBLASLT_LOG_LEVEL": {
            "default": "0",
            "desc": "hipBLASLt Logging",
            "widget": "dropdown",
            "options": [
                ("0 - Off (Hidden)", "0"),
                ("1 - Error", "1"),
                ("2 - Trace", "2"),
                ("3 - Hints", "3"),
                ("4 - Info", "4"),
                ("5 - API Trace", "5"),
            ],
            "restart_required": True,
        },
    "MIOPEN_LOG_LEVEL": {
        "default": "0",
        "desc": "MIOpen log verbosity level",
        "widget": "dropdown",
        "options": [
            ("0 - Default", "0"),
            ("1 - Quiet", "1"),
            ("3 - Error", "3"),
            ("4 - Warning", "4"),
            ("5 - Info", "5"),
            ("6 - Detail", "6"),
            ("7 - Trace", "7"),
        ],
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

ROCM_ENV_VARS: Dict[str, Dict[str, Any]] = {}
ROCM_ENV_VARS.update(GENERAL_VARS)


def sync_rocm_env_vars(config_dict: Dict[str, str]) -> None:
    """Synchronizes config dictionary values to os.environ, respecting explicit manual settings."""
    for key in ROCM_ENV_VARS.keys():
        val = str(config_dict.get(key, "")).strip()
        if not val:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
