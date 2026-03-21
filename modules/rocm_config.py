# --- Compatibility stub for cross_attention_optimization ---
# This is only to prevent AttributeError if code expects this option in opts.
if 'cross_attention_optimization' not in globals():
    cross_attention_optimization = 'default'

# --- ROCm GPU/arch/hipblaslt info (from modules/rocm.py) ---
try:
    from modules import rocm
    # Minimal mapping for common AMD dGPUs (expand as needed)
    GFX_MARKETING = {
        'gfx1030': 'Radeon RX 6800',
        'gfx1031': 'Radeon RX 6900',
        'gfx1032': 'Radeon RX 6700',
        'gfx1034': 'Radeon RX 6600',
        'gfx1100': 'Radeon RX 7900 XTX',
        'gfx1101': 'Radeon RX 7900 XT',
        'gfx1102': 'Radeon RX 7800 XT',
        # Add more as needed
    }
    # Get all detected agents (GPUs)
    agents = rocm.get_agents() if hasattr(rocm, 'get_agents') else []
    current_gpu = agents[0] if agents else None
    if current_gpu:
        gfx_code = str(current_gpu)
        marketing = GFX_MARKETING.get(gfx_code, None)
        if marketing:
            current_gpu_display = f"{gfx_code} {marketing}"
        else:
            current_gpu_display = gfx_code
        current_gpu_arch = current_gpu.arch.value if hasattr(current_gpu, 'arch') else 'unavailable'
    else:
        current_gpu_display = 'unavailable'
        current_gpu_arch = 'unavailable'
    blaslt_path = getattr(rocm, 'blaslt_tensile_libpath', None)
except Exception as e:
    current_gpu_name = f'unavailable ({e})'
    current_gpu_arch = 'unavailable'
    blaslt_path = None

# These can be imported and used elsewhere as needed
ROCM_CURRENT_GPU = current_gpu_display
ROCM_CURRENT_ARCH = current_gpu_arch
ROCM_HIPBLASLT_PATH = blaslt_path

# --- Version reporting for ROCm, HIP, GPU, MIOpen, rocBLAS ---
def get_versions() -> dict:
    """Return torch/GPU/ROCm/MIOpen/rocBLAS info for display."""
    import os as _os
    import sys as _sys
    import glob as _glob
    import ctypes as _ctypes
    versions = {}

    # Torch version + HIP version + GPU info (all from torch)
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["hip"] = getattr(torch.version, 'hip', None) or 'unavailable'
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            versions["gpu"] = torch.cuda.get_device_name(device)
            versions["vram"] = f"{round(props.total_memory / 1024 / 1024)} MB"
            versions["arch"] = str(torch.cuda.get_device_capability(device))
            versions["cores"] = str(props.multi_processor_count)
    except Exception as e:
        versions["torch"] = f"unavailable ({e})"

    # ROCm version from modules.rocm (uses hip library via rocm-sdk package)
    try:
        from modules import rocm as _rocm
        versions["rocm"] = _rocm.version or 'unavailable'
    except Exception as e:
        versions["rocm"] = f"unavailable ({e})"

    # DLL search: _rocm_sdk_devel/bin first, then wildcard libraries, then core
    # Uses sys.prefix as fallback when VIRTUAL_ENV is not set in environment
    def find_rocm_dll(dll_name: str):
        venv = _os.environ.get('VIRTUAL_ENV') or _sys.prefix
        if venv:
            for pattern in [
                _os.path.join(venv, 'Lib', 'site-packages', '_rocm_sdk_devel', 'bin', dll_name),
                _os.path.join(venv, 'Lib', 'site-packages', '_rocm_sdk_libraries_*', 'bin', dll_name),
                _os.path.join(venv, 'Lib', 'site-packages', '_rocm_sdk_core', 'bin', dll_name),
            ]:
                candidates = _glob.glob(pattern)
                if candidates:
                    return candidates[0]
        return None

    # MIOpen version — uses miopenGetVersion(int* major, int* minor, int* patch)
    try:
        miopen_path = find_rocm_dll('MIOpen.dll')
        if miopen_path:
            _os.add_dll_directory(_os.path.dirname(miopen_path))
            miopen = _ctypes.CDLL(miopen_path)
            fn = getattr(miopen, 'miopenGetVersion', None)
            if fn:
                fn.restype = None
                fn.argtypes = [_ctypes.POINTER(_ctypes.c_int)] * 3
                major, minor, patch = _ctypes.c_int(), _ctypes.c_int(), _ctypes.c_int()
                fn(_ctypes.byref(major), _ctypes.byref(minor), _ctypes.byref(patch))
                versions['miopen'] = f'{major.value}.{minor.value}.{patch.value}'
            else:
                versions['miopen'] = 'unknown'
        else:
            versions['miopen'] = 'unavailable (DLL not found)'
    except Exception as e:
        versions['miopen'] = f'unavailable ({e})'

    # rocBLAS version — rocblas_get_version_string_size + rocblas_get_version_string
    try:
        rocblas_path = find_rocm_dll('rocblas.dll')
        if rocblas_path:
            _os.add_dll_directory(_os.path.dirname(rocblas_path))
            rocblas = _ctypes.CDLL(rocblas_path)
            sz_fn = getattr(rocblas, 'rocblas_get_version_string_size', None)
            vs_fn = getattr(rocblas, 'rocblas_get_version_string', None)
            if sz_fn and vs_fn:
                sz_fn.restype = _ctypes.c_int
                sz = _ctypes.c_size_t()
                sz_fn(_ctypes.byref(sz))
                buf = _ctypes.create_string_buffer(sz.value)
                vs_fn.restype = _ctypes.c_int
                vs_fn(buf, sz)
                versions['rocblas'] = buf.value.decode().strip('\x00')
            else:
                versions['rocblas'] = 'unknown'
        else:
            versions['rocblas'] = 'unavailable (DLL not found)'
    except Exception as e:
        versions['rocblas'] = f'unavailable ({e})'

    # hipBLASLt version — hipblasLtCreate + hipblasLtGetVersion(handle, int*) + hipblasLtDestroy
    try:
        hipblaslt_path = find_rocm_dll('libhipblaslt.dll')
        if hipblaslt_path:
            _os.add_dll_directory(_os.path.dirname(hipblaslt_path))
            hipblaslt = _ctypes.CDLL(hipblaslt_path)
            create_fn = getattr(hipblaslt, 'hipblasLtCreate', None)
            ver_fn    = getattr(hipblaslt, 'hipblasLtGetVersion', None)
            dest_fn   = getattr(hipblaslt, 'hipblasLtDestroy', None)
            git_fn = getattr(hipblaslt, 'hipblasLtGetGitRevision', None)
            if create_fn and ver_fn:
                create_fn.restype = _ctypes.c_int
                create_fn.argtypes = [_ctypes.POINTER(_ctypes.c_void_p)]
                handle = _ctypes.c_void_p()
                if create_fn(_ctypes.byref(handle)) == 0:
                    ver_fn.restype = _ctypes.c_int
                    ver_fn.argtypes = [_ctypes.c_void_p, _ctypes.POINTER(_ctypes.c_int)]
                    ver = _ctypes.c_int()
                    ver_str = 'unknown'
                    if ver_fn(handle, _ctypes.byref(ver)) == 0:
                        v = ver.value
                        ver_str = f'{v // 100000}.{(v % 100000) // 100}.{v % 100}'
                    git_hash = ''
                    if git_fn:
                        git_fn.restype = _ctypes.c_int
                        git_fn.argtypes = [_ctypes.c_void_p, _ctypes.c_char_p, _ctypes.c_size_t]
                        buf = _ctypes.create_string_buffer(256)
                        if git_fn(handle, buf, _ctypes.c_size_t(256)) == 0:
                            git_hash = buf.value.decode('ascii', errors='replace').strip()
                    versions['hipblaslt'] = f'{ver_str}.{git_hash}' if git_hash else ver_str
                    if dest_fn:
                        dest_fn.restype = _ctypes.c_int
                        dest_fn.argtypes = [_ctypes.c_void_p]
                        dest_fn(handle)
                else:
                    versions['hipblaslt'] = 'unknown'
            else:
                versions['hipblaslt'] = 'unknown'
        else:
            versions['hipblaslt'] = 'unavailable (DLL not found)'
    except Exception as e:
        versions['hipblaslt'] = f'unavailable ({e})'

    return versions

"""
ROCm/MIOpen/rocBLAS Environment Variable Configuration

- Loads/saves config from rocm-config.json in workspace root.
- Provides variable metadata for UI (type, options, help, restart requirement).
- Applies changes to os.environ for live effect where possible.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from modules.errors import log

CONFIG_PATH = Path("rocm-config.json")

# --- hipBLASLt availability (DLL probe at import time) ---
def _find_hipblaslt_dll() -> bool:
    import sys as _sys, glob as _glob
    venv = os.environ.get('VIRTUAL_ENV') or _sys.prefix
    for pattern in [
        os.path.join(venv, 'Lib', 'site-packages', '_rocm_sdk_devel', 'bin', 'libhipblaslt.dll'),
        os.path.join(venv, 'Lib', 'site-packages', '_rocm_sdk_libraries_*', 'bin', 'libhipblaslt.dll'),
    ]:
        if _glob.glob(pattern):
            return True
    return False

HIPBLASLT_AVAILABLE: bool = _find_hipblaslt_dll()

# --- General MIOpen/rocBLAS variables (dropdown/textbox) ---
GENERAL_VARS = {

    # --- TunableOp: rocBLAS / hipBLASLt kernel cache ---
    "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": {
        "default": "1",
        "desc": "Enable rocBLAS backend",
        "widget": "checkbox",
        "options": None,
        "restart_required": True,
    },
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": {
        "default": "0",
        "desc": "Enable hipBLASLt backend",
        "widget": "checkbox",
        "options": None,
        "restart_required": True,
        "available": HIPBLASLT_AVAILABLE,
    },
    "PYTORCH_TUNABLEOP_TUNING": {
        "default": "1",
        "desc": "Tuning mode (uncheck for cache-only)",
        "widget": "checkbox",
        "options": None,
        "restart_required": True,
    },
    "PYTORCH_TUNABLEOP_FILENAME": {
        "default": "models\\tunable\\tunable.csv",
        "desc": "Cache file path",
        "widget": "textbox",
        "options": None,
        "restart_required": True,
    },
    "PYTORCH_TUNABLEOP_VERBOSE": {
        "default": "0",
        "desc": "Verbose logging (print cache hits/misses)",
        "widget": "checkbox",
        "options": None,
        "restart_required": False,
    },
    "ROCBLAS_LAYER": {
        "default": "0",
        "desc": "rocBLAS logging layer",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - Trace", "1"), ("2 - Bench", "2"), ("3 - Trace+Bench", "3"), ("4 - Profile", "4"), ("5 - Trace+Profile", "5"), ("6 - Bench+Profile", "6"), ("7 - All", "7")],
        "restart_required": False,
    },

    # MIOPEN_FIND_MODE: single value 1-7
    "MIOPEN_FIND_MODE": {
        "default": "2",  # FAST mode default
        "desc": "MIOpen Find Mode",
        "widget": "dropdown",
        "options": [("1 - NORMAL", "1"), ("2 - FAST", "2"), ("3 - HYBRID", "3"), ("5 - DYNAMIC_HYBRID", "5"), ("6 - TRUST_VERIFY", "6"), ("7 - TRUST_VERIFY_FULL", "7")],
        "restart_required": True,
    },
    # MIOPEN_FIND_ENFORCE: single value 1-5
    "MIOPEN_FIND_ENFORCE": {
        "default": "1",  # NONE default
        "desc": "MIOpen Find Enforce",
        "widget": "dropdown",
        "options": [("1 - NONE", "1"), ("2 - DB_UPDATE", "2"), ("3 - SEARCH", "3"), ("4 - SEARCH_DB_UPDATE", "4"), ("5 - DB_CLEAN", "5")],
        "restart_required": True,
    },
    "MIOPEN_SYSTEM_DB_PATH": {
        "default": "{VIRTUAL_ENV}\\Lib\\site-packages\\_rocm_sdk_devel\\bin\\",
        "desc": "MIOpen system DB path (string)",
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
        "desc": "Enable MIOpen debug logging",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": False,
    },
}

SOLVER_VARS = {}

# --- Algorithm/Solver Group Enables ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_FFT": "Enable FFT solver",
    "MIOPEN_DEBUG_CONV_DIRECT": "Enable Direct solver",
    "MIOPEN_DEBUG_CONV_GEMM": "Enable GEMM solver",
    "MIOPEN_DEBUG_CONV_WINOGRAD": "Enable Winograd solver",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "Enable Implicit GEMM solver",
})

# --- Immediate Fallback Mode ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_IMMED_FALLBACK": "Enable Immediate Fallback",
    "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "Enable AI Immediate Mode Fallback",
    "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK": "Force Immediate Mode Fallback",
})

# --- Build Method Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_GCN_ASM_KERNELS": "Enable GCN ASM kernels",
    "MIOPEN_DEBUG_HIP_KERNELS": "Enable HIP kernels",
    "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS": "Enable OpenCL convolutions",
    "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP": "Enable OpenCL Wave64 NOWGP",
    "MIOPEN_DEBUG_ATTN_SOFTMAX": "Enable Attention Softmax",
})

# --- Direct ASM Solver Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U": "Enable Direct ASM 3x3U",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U": "Enable Direct ASM 1x1U",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2": "Enable Direct ASM 1x1UV2",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2": "Enable Direct ASM 5x10U2V2",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224": "Enable Direct ASM 7x7C3H224W224",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3": "Enable Direct ASM WRW3X3",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1": "Enable Direct ASM WRW1X1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS": "Enable Direct ASM 1x1U Perf Vals",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED": "Enable Direct ASM 1x1U Search Optimized",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR": "Enable Direct ASM 1x1U AI Heuristic",
})

# --- Direct OpenCL Solver Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD": "Enable Direct OCL FWD",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1": "Enable Direct OCL FWD1X1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11": "Enable Direct OCL FWD11X11",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN": "Enable Direct OCL FWDGEN",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2": "Enable Direct OCL WRW2",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53": "Enable Direct OCL WRW53",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1": "Enable Direct OCL WRW1X1",
})

# --- Winograd Solver Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3": "Enable AMD Winograd 3x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS": "Enable AMD Winograd RxS",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD": "Enable AMD Winograd RxS FWD/BWD",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW": "Enable AMD Winograd RxS WRW",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2": "Enable AMD Winograd RxS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3": "Enable AMD Winograd RxS F2x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1": "Enable AMD Winograd RxS F2x3 G1",
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD": "Enable AMD Fused Winograd",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3": "Enable AMD Winograd Fury RxS F2x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2": "Enable AMD Winograd Fury RxS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3": "Enable AMD Winograd Rage RxS F2x3",
})

# --- Multi-pass Winograd Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2": "Enable AMD Winograd MPASS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3": "Enable AMD Winograd MPASS F3x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4": "Enable AMD Winograd MPASS F3x4",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5": "Enable AMD Winograd MPASS F3x5",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6": "Enable AMD Winograd MPASS F3x6",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3": "Enable AMD Winograd MPASS F5x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4": "Enable AMD Winograd MPASS F5x4",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2": "Enable AMD Winograd MPASS F7x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3": "Enable AMD Winograd MPASS F7x3",
})

# --- MP BD Winograd Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3": "Enable AMD MP BD Winograd F2x3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3": "Enable AMD MP BD Winograd F3x3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3": "Enable AMD MP BD Winograd F4x3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3": "Enable AMD MP BD Winograd F5x3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3": "Enable AMD MP BD Winograd F6x3",
})

# --- MP BD XDLOPS Winograd Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3": "Enable AMD MP BD XDLOPS Winograd F2x3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3": "Enable AMD MP BD XDLOPS Winograd F3x3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3": "Enable AMD MP BD XDLOPS Winograd F4x3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3": "Enable AMD MP BD XDLOPS Winograd F5x3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3": "Enable AMD MP BD XDLOPS Winograd F6x3",
})

# --- ASM Implicit GEMM Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1": "Enable ASM Implicit GEMM FWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1": "Enable ASM Implicit GEMM FWD V4R1 1x1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1": "Enable ASM Implicit GEMM BWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1": "Enable ASM Implicit GEMM WRW V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS": "Enable ASM Implicit GEMM FWD GTC XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS": "Enable ASM Implicit GEMM BWD GTC XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS": "Enable ASM Implicit GEMM WRW GTC XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC": "Enable ASM Implicit GEMM FWD GTC XDLOPS NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC": "Enable ASM Implicit GEMM BWD GTC XDLOPS NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS_NHWC": "Enable ASM Implicit GEMM WRW GTC XDLOPS NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC": "Enable ASM Implicit GEMM FWD GTC DLOPS NCHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16": "Enable ASM PK Atomic Add FP16",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS": "Enable HIP Group BWD XDLOPS",
    "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR": "Enable Group HIP BWD XDLOPS AI Heuristic",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_FWD_V4R4_XDLOPS_ADD_VECTOR_LOAD_GEMMN_TUNE_PARAM": "Enable FWD V4R4 XDLOPS Add Vector Load GEMMN Tune Param",
})

# --- HIP Implicit GEMM Toggles ---
SOLVER_VARS.update({
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "Enable HIP Implicit GEMM FWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4": "Enable HIP Implicit GEMM FWD V4R4",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1": "Enable HIP Implicit GEMM BWD V1R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1": "Enable HIP Implicit GEMM BWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1": "Enable HIP Implicit GEMM WRW V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4": "Enable HIP Implicit GEMM WRW V4R4",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS": "Enable HIP Implicit GEMM FWD V4R4 XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS": "Enable HIP Implicit GEMM FWD V4R5 XDLOPS",
})
    # (removed invalid duplicate dictionary block)

# Merge all variable definitions
ROCM_ENV_VARS: Dict[str, Dict[str, Any]] = {}
ROCM_ENV_VARS.update(GENERAL_VARS)

# Vars that are disabled ("0") by default — XDLOPS (CDNA only), Fury/Rage (RDNA3/4),
# most multi-pass WrW Winograd, forced-fallback flag, and non-RDNA2 HIP variants.
_SOLVER_DISABLED_BY_DEFAULT = {
    "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK",
    # Winograd — RDNA3/4 only
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3",
    # Multi-pass Winograd WrW — not used in SD/WAN inference
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3",
    # MP BD Winograd — not used in SD/WAN
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3",
    # MP BD XDLOPS — CDNA only
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3",
    # ASM iGEMM XDLOPS — CDNA only
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS_NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS",
    "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_FWD_V4R4_XDLOPS_ADD_VECTOR_LOAD_GEMMN_TUNE_PARAM",
    # HIP iGEMM XDLOPS — CDNA only
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS",
}

for var, desc in SOLVER_VARS.items():
    ROCM_ENV_VARS[var] = {
        "default": "0" if var in _SOLVER_DISABLED_BY_DEFAULT else "1",
        "desc": desc,
        "widget": "checkbox",
        "options": None,
        "restart_required": False,
    }


# No cross_attention or unrelated version reporting. Only config logic remains.

def load_config() -> Dict[str, str]:
    """Load config from file or return defaults."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                log.debug(f'Read: file="{CONFIG_PATH}"')
                return config
    except Exception as e:
        log.warning(f"Failed to load ROCm config: {e}")
    return {k: v["default"] for k, v in ROCM_ENV_VARS.items()}

def save_config(config: Dict[str, str]) -> None:
    """Save config to file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        log.debug(f'Saved: file="{CONFIG_PATH}"')
    except Exception as e:
        log.error(f"Failed to save ROCm config: {e}")

def apply_env(config: Optional[Dict[str, str]] = None) -> None:
    """Apply config to os.environ, expanding {VIRTUAL_ENV} in path values; skip empty values."""
    if config is None:
        config = load_config()
    applied = []
    import sys as _sys
    _venv = os.environ.get("VIRTUAL_ENV", "") or _sys.prefix
    for var, value in config.items():
        value = str(value).replace("{VIRTUAL_ENV}", _venv)
        if value == "":
            continue  # don't override env with empty string
        os.environ[var] = value
        applied.append(f"{var}={value}")
    log.debug(f"ROCm config: Loaded from {CONFIG_PATH}")

def _get_venv() -> str:
    """Return the venv root (VIRTUAL_ENV env var or sys.prefix)."""
    import sys as _sys
    return os.environ.get("VIRTUAL_ENV", "") or _sys.prefix

def _collapse_venv(value: str) -> str:
    """Replace the real venv path with {VIRTUAL_ENV} placeholder so JSON stays portable."""
    venv = _get_venv()
    if venv and value.startswith(venv):
        return "{VIRTUAL_ENV}" + value[len(venv):]
    return value

def _dropdown_choices(options):
    """Return plain label strings for Gradio choices (tuple options → label only)."""
    if options and isinstance(options[0], tuple):
        return [label for label, _ in options]
    return options

def _dropdown_display(stored_val, options):
    """Map stored value (e.g. '2') to Gradio display label (e.g. '2 - FAST')."""
    if options and isinstance(options[0], tuple):
        return next((label for label, val in options if val == str(stored_val)), str(stored_val))
    return str(stored_val)

def _dropdown_stored(display_val, options):
    """Map Gradio display label (e.g. '2 - FAST') back to stored value (e.g. '2')."""
    if options and isinstance(options[0], tuple):
        return next((val for label, val in options if label == str(display_val)), str(display_val))
    return str(display_val)

def get_var_info() -> List[Dict[str, Any]]:
    """Return variable info for UI: name, value, description, widget, options, restart_required, with visible group headings."""
    config = load_config()
    info = []
    # Groupings: (heading, [vars])
    groups = [
        ("Algorithm/Solver Group Enables", [
            "MIOPEN_DEBUG_CONV_FFT", "MIOPEN_DEBUG_CONV_DIRECT", "MIOPEN_DEBUG_CONV_GEMM", "MIOPEN_DEBUG_CONV_WINOGRAD", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"
        ]),
        ("Immediate Fallback Mode", [
            "MIOPEN_DEBUG_CONV_IMMED_FALLBACK", "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK", "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK"
        ]),
        ("Build Method Toggles", [
            "MIOPEN_DEBUG_GCN_ASM_KERNELS", "MIOPEN_DEBUG_HIP_KERNELS", "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS", "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP", "MIOPEN_DEBUG_ATTN_SOFTMAX"
        ]),
        ("Direct ASM Solver Toggles", [
            "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2", "MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2", "MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224", "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3", "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR"
        ]),
        ("Direct OpenCL Solver Toggles", [
            "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD", "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1", "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11", "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN", "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2", "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53", "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1"
        ]),
        ("Winograd Solver Toggles", [
            "MIOPEN_DEBUG_AMD_WINOGRAD_3X3", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1", "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD", "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3", "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3"
        ]),
        ("Multi-pass Winograd Toggles", [
            "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3"
        ]),
        ("MP BD Winograd Toggles", [
            "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3", "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3", "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3", "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3", "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3"
        ]),
        ("MP BD XDLOPS Winograd Toggles", [
            "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3", "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3", "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3", "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3", "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3"
        ]),
        ("ASM Implicit GEMM Toggles", [
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS_NHWC", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS", "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_FWD_V4R4_XDLOPS_ADD_VECTOR_LOAD_GEMMN_TUNE_PARAM"
        ]),
        ("HIP Implicit GEMM Toggles", [
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS"
        ]),
    ]
    # Collect all vars already covered by a group so we don't render them twice
    grouped_vars = {v for _, varlist in groups for v in varlist}
    import sys as _sys
    _venv = os.environ.get("VIRTUAL_ENV", "") or _sys.prefix

    # Add general vars first (skip any that are rendered under a group heading)
    _tunableop_heading_done = False
    _miopen_heading_done = False
    for var, meta in GENERAL_VARS.items():
        if var in grouped_vars:
            continue
        if not meta.get("available", True):
            continue
        # Emit TunableOp section heading before first PYTORCH_TUNABLEOP_ var
        if not _tunableop_heading_done and var.startswith("PYTORCH_TUNABLEOP_"):
            info.append({"name": "tunableop_group_sep", "value": "<div><h2>Tunable Ops Settings</h2><hr style='margin:4px 0'></div>",
                          "desc": "", "widget": "html", "options": None, "restart_required": False})
            _tunableop_heading_done = True
        # Emit MIOpen section heading before first MIOPEN_ var
        if not _miopen_heading_done and var.startswith("MIOPEN_"):
            info.append({"name": "miopen_group_sep", "value": "<div><h2>MIOpen Settings</h2><hr style='margin:4px 0'></div>",
                          "desc": "", "widget": "html", "options": None, "restart_required": False})
            _miopen_heading_done = True
        val = config.get(var, meta["default"])
        if meta["widget"] == "checkbox":
            val = val == "1"
            disp, choices = val, meta["options"]
        elif meta["widget"] == "dropdown":
            disp = _dropdown_display(val, meta["options"])
            choices = _dropdown_choices(meta["options"])
        else:
            disp = str(val).replace("{VIRTUAL_ENV}", _venv)
            choices = meta["options"]
        info.append({
            "name": var,
            "value": disp,
            "desc": meta["desc"],
            "widget": meta["widget"],
            "options": choices,
            "restart_required": meta["restart_required"],
        })
    # Add solver groups with headings
    for group, varlist in groups:
        # Insert bordered group heading (closed div)
        info.append({
            "name": f"{group.lower().replace(' ', '_').replace('/', '_')}_group_sep",
            "value": f"<div><h2>{group}</h2><hr style='margin:4px 0'></div>",
            "desc": "",
            "widget": "html",
            "options": None,
            "restart_required": False,
        })
        for var in varlist:
            meta = ROCM_ENV_VARS[var]
            val = config.get(var, meta["default"])
            if meta["widget"] == "checkbox":
                val = val == "1"
                disp, choices = val, meta["options"]
            elif meta["widget"] == "dropdown":
                disp = _dropdown_display(val, meta["options"])
                choices = _dropdown_choices(meta["options"])
            else:
                disp, choices = val, meta["options"]
            info.append({
                "name": var,
                "value": disp,
                "desc": meta["desc"],
                "widget": meta["widget"],
                "options": choices,
                "restart_required": meta["restart_required"],
            })
    return info

def sync_from_opts() -> None:
    """Collect all ROCm var values from shared.opts and persist to rocm-config.json."""
    from modules import shared  # lazy import — not safe at module level
    existing = load_config()  # use saved values as fallback for any key not yet in shared.opts
    config = {}
    for name, meta in ROCM_ENV_VARS.items():
        value = shared.opts.data.get(name, existing.get(name, meta["default"]))
        if meta["widget"] == "checkbox":
            # opts.data may store a bool or a string "0"/"1"; normalise before coercing
            if isinstance(value, str):
                value = value == "1"
            config[name] = "1" if value else "0"
        elif meta["widget"] == "dropdown":
            config[name] = _dropdown_stored(value, meta["options"])
        else:
            config[name] = _collapse_venv(str(value))
    save_config(config)
    apply_env(config)

def set_var(name: str, value: Any) -> None:
    """Set a variable, save, and apply."""
    if name not in ROCM_ENV_VARS:
        log.warning(f"Attempted to set unknown ROCm variable: {name}")
        return
    config = load_config()
    # For checkboxes, value is bool; store as "1"/"0"
    if ROCM_ENV_VARS[name]["widget"] == "checkbox":
        config[name] = "1" if value else "0"
    else:
        config[name] = str(value)
    save_config(config)
    apply_env(config)
    log.info(f"Set ROCm variable: {name}={config[name]}")

def reset_defaults() -> None:
    """Reset all variables to defaults."""
    defaults = {k: v["default"] for k, v in ROCM_ENV_VARS.items()}
    save_config(defaults)
    apply_env(defaults)
    log.info("Reset ROCm config to defaults.")

# Apply saved config to os.environ at import time so env vars are set before first inference
try:
    apply_env()
except Exception as _e:
    import sys as _sys
    print(f"[rocm_config] Warning: failed to apply env at import: {_e}", file=_sys.stderr)

if __name__ == "__main__":
    print("ROCm/MIOpen/rocBLAS versions:")
    for k, v in get_versions().items():
        print(f"  {k}: {v}")
    print("\nROCm config variables:")
    for var in get_var_info():
        print(f"{var['name']}: {var['value']} ({var['desc']}) [{var['widget']}]")
