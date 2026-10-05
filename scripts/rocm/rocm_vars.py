import os
import sys
import sysconfig
from typing import Any, Dict


_BIN_DIR = "bin" if sys.platform == "win32" else "lib"


def _sitepackages_subpath(*parts: str) -> str: # auto-path helper (currently not called, kept for troubleshooting.)
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
        "desc": "rocBLAS Tensile library path",
        "widget": "textbox",
        "options": None,
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
    "MIOPEN_GEMM_ENFORCE_BACKEND": {
        "default": "1",
        "desc": "GEMM backend",
        "widget": "dropdown",
        "options": [("1 - rocBLAS", "1"), ("5 - hipBLASLt", "5")],
        "restart_required": False,
    },
    "PYTORCH_ROCM_USE_ROCBLAS": {
        "default": "0",
        "desc": "PyTorch: Use rocBLAS",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": True,
    },
    "PYTORCH_HIPBLASLT_DISABLE": {
        "default": "1",
        "desc": "PyTorch: Use hipBLASLt",
        "widget": "dropdown",
        "options": [("0 - Allow hipBLASLt", "0"), ("1 - Disable hipBLASLt", "1")],
        "restart_required": True,
    },
    "ROCBLAS_USE_HIPBLASLT": {
        "default": "0",
        "desc": "rocBLAS: use hipBLASLt backend",
        "widget": "dropdown",
        "options": [("0 - Tensile (rocBLAS)", "0"), ("1 - hipBLASLt", "1")],
        "restart_required": True,
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
    
    # --> OUTCOMMENTED SECTION REMOVED FROM ACTIVE ROCm CONFIGURATION REGISTRY; KEPT FOR REFERENCE <--

    #"ROCBLAS_DEVICE_MEMORY_SIZE": {
    #    "default": "",
    #    "desc": "rocBLAS workspace size in bytes (empty = dynamic)",
    #    "widget": "textbox",
    #    "options": None,
    #    "restart_required": False,
    #},
    #"PYTORCH_TUNABLEOP_CACHE_DIR": {
    #    "default": os.path.join("{ROOT}", "models", "tunable"),
    #    "desc": "TunableOp cache directory",
    #    "widget": "textbox",
    #    "options": None,
    #    "restart_required": False,
    #},
    #
    #"ROCBLAS_STREAM_ORDER_ALLOC": {
    #    "default": "1",
    #    "desc": "rocBLAS stream-ordered memory allocation",
    #    "widget": "dropdown",
    #    "options": [("0 - Standard", "0"), ("1 - Stream-ordered", "1")],
    #    "restart_required": False,
    #},
    #"ROCBLAS_DEFAULT_ATOMICS_MODE": {
    #    "default": "1",
    #    "desc": "rocBLAS allow atomics",
    #    "widget": "dropdown",
    #    "options": [("0 - Off (deterministic)", "0"), ("1 - On (performance)", "1")],
    #    "restart_required": False,
    #},
    #"PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": {
    #    "default": "0",
    #    "desc": "TunableOp: Enable tuning",
    #    "widget": "dropdown",
    #    "options": [("0 - Off", "0"), ("1 - On", "1")],
    #    "restart_required": False,
    #},
    #"PYTORCH_TUNABLEOP_TUNING": {
    #    "default": "0",
    #    "desc": "TunableOp: Tuning mode",
    #    "widget": "dropdown",
    #    "options": [("0 - Use Cache", "0"), ("1 - Benchmark new shapes", "1")],
    #    "restart_required": False,
    #},
    #"PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": {
    #    "default": "0",
    #    "desc": "TunableOp: benchmark hipBLASLt kernels",
    #    "widget": "dropdown",
    #    "options": [("0 - Off", "0"), ("1 - On", "1")],
    #    "restart_required": False,
    #},
    #
    #"ROCBLAS_LAYER": {
    #    "default": "0",
    #    "desc": "rocBLAS logging",
    #    "widget": "dropdown",
    #    "options": [("0 - Off", "0"), ("1 - Trace", "1"), ("2 - Bench", "2"), ("3 - Trace+Bench", "3"), ("4 - Profile", "4"), ("5 - Trace+Profile", "5"), ("6 - Bench+Profile", "6"), ("7 - All", "7")],
    #    "restart_required": False,
    #},
    #"HIPBLASLT_LOG_LEVEL": {
    #    "default": "0",
    #    "desc": "hipBLASLt logging",
    #    "widget": "dropdown",
    #    "options": [("0 - Off", "0"), ("1 - Error", "1"), ("2 - Trace", "2"), ("3 - Hints", "3"), ("4 - Info", "4"), ("5 - API Trace", "5")],
    #    "restart_required": False,
    #},
}

# SOLVER_DTYPE_TAGS: Dict[str, str] = {
#     "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_3X3": "FP32",
#     "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD": "FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RXS": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2": "FP16/FP32",
#     "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "FP16/FP32",
#     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4": "FP16/FP32",
#     "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS": "FP16/BF16",
#     "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR": "FP16/BF16",
#     "MIOPEN_DEBUG_CK_DEFAULT_KERNELS": "FP16/BF16/FP32",
# }

# Solver controls are currently hidden. Keep their metadata above for reference, but do not
# expose or apply it through the active ROCm configuration registry.
ROCM_ENV_VARS: Dict[str, Dict[str, Any]] = {}
ROCM_ENV_VARS.update(GENERAL_VARS)

MIOPEN_LOGGING_VARS = {"MIOPEN_LOG_LEVEL", "MIOPEN_DEBUG_ENABLE"}

# Variables that are relevant only when hipBLASLt is the active GEMM backend.
# These are visually greyed-out in the UI when rocBLAS (MIOPEN_GEMM_ENFORCE_BACKEND="1") is selected.
HIPBLASLT_VARS: set = {
    "PYTORCH_HIPBLASLT_DISABLE",
    "ROCBLAS_USE_HIPBLASLT",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED",
    "HIPBLASLT_LOG_LEVEL",
}
