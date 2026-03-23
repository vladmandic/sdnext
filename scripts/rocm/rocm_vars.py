from typing import Dict, Any, List, Tuple

# --- General MIOpen/rocBLAS variables (dropdown/textbox/checkbox) ---
GENERAL_VARS: Dict[str, Dict[str, Any]] = {

    "MIOPEN_GEMM_ENFORCE_BACKEND": {
        "default": "1",
        "desc": "Enforce GEMM backend",
        "widget": "dropdown",
        "options": [("1 - rocBLAS", "1"), ("5 - hipBLASLt", "5")],
        "restart_required": False,
    },
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
    "MIOPEN_SEARCH_CUTOFF": {
        "default": "0",
        "desc": "Enable early termination of suboptimal searches",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": True,
    },
    "MIOPEN_SYSTEM_DB_PATH": {
        "default": "{VIRTUAL_ENV}\\Lib\\site-packages\\_rocm_sdk_devel\\bin\\",
        "desc": "MIOpen system DB path",
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
    "ROCBLAS_LAYER": {
        "default": "0",
        "desc": "rocBLAS logging",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - Trace", "1"), ("2 - Bench", "2"), ("3 - Trace+Bench", "3"), ("4 - Profile", "4"), ("5 - Trace+Profile", "5"), ("6 - Bench+Profile", "6"), ("7 - All", "7")],
        "restart_required": False,
    },
    "HIPBLASLT_LOG_LEVEL": {
        "default": "0",
        "desc": "hipBLASLt logging",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - Error", "1"), ("2 - Trace", "2"), ("3 - Hints", "3"), ("4 - Info", "4"), ("5 - API Trace", "5")],
        "restart_required": False,
    },
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": {
        "default": "0",
        "desc": "Deterministic convolution (reproducible results, may be slower)",
        "widget": "dropdown",
        "options": [("0 - Off", "0"), ("1 - On", "1")],
        "restart_required": False,
    },
}

# --- Solver toggles (inference/FWD only, RDNA2/3/4 compatible) ---
# Removed entirely — not representable in the UI, cannot be set by users:
#   WRW (weight-gradient) and BWD (data-gradient) — training passes only, never run during inference
#   XDLOPS, CK (Composable Kernel), MLIR, MP BD — CDNA (MI100/MI200/MI300) matrix engine; not on RDNA
#   Fixed-geometry (5x10, 7x7-ImageNet, 11x11) — shapes never appear in SD/video inference
#   FP32-reference (NAIVE_CONV_FWD, FWDGEN) — IsApplicable() unreliable for FP16/BF16
#   Wide MPASS (F3x4..F7x3) — kernel sizes that cannot match any SD convolution shape
_SOLVER_DESCS: Dict[str, str] = {}

_SOLVER_DESCS.update({
    "MIOPEN_DEBUG_CONV_FFT":           "Enable FFT solver",
    "MIOPEN_DEBUG_CONV_DIRECT":        "Enable Direct solver",
    "MIOPEN_DEBUG_CONV_GEMM":          "Enable GEMM solver",
    "MIOPEN_DEBUG_CONV_WINOGRAD":      "Enable Winograd solver",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "Enable Implicit GEMM solver",
})
_SOLVER_DESCS.update({
    "MIOPEN_DEBUG_CONV_IMMED_FALLBACK":           "Enable Immediate Fallback",
    "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "Enable AI Immediate Mode Fallback",
    "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK":     "Force Immediate Mode Fallback",
})
_SOLVER_DESCS.update({
    "MIOPEN_DEBUG_GCN_ASM_KERNELS":      "Enable GCN ASM kernels",
    "MIOPEN_DEBUG_HIP_KERNELS":          "Enable HIP kernels",
    "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS":  "Enable OpenCL convolutions",
    "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP":  "Enable OpenCL Wave64 NOWGP",
    "MIOPEN_DEBUG_ATTN_SOFTMAX":         "Enable Attention Softmax",
})
_SOLVER_DESCS.update({
    # Direct ASM — FWD inference only (WRW, fixed-geometry, FP32-reference removed)
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U":                  "Enable Direct ASM 3x3U",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U":                  "Enable Direct ASM 1x1U",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2":                "Enable Direct ASM 1x1UV2",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED": "Enable Direct ASM 1x1U Search Optimized",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR":          "Enable Direct ASM 1x1U AI Heuristic",
})
_SOLVER_DESCS.update({
    # Direct OCL — FWD inference only (WRW, FWD11X11 fixed-geom, FWDGEN FP32-ref removed)
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD":    "Enable Direct OCL FWD",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1": "Enable Direct OCL FWD1X1",
})
_SOLVER_DESCS.update({
    # Winograd FWD — WRW removed; Fury/Rage kept as RDNA3/4 inference (off by default)
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3":           "Enable AMD Winograd 3x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS":           "Enable AMD Winograd RxS",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD":   "Enable AMD Winograd RxS FWD",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2":      "Enable AMD Winograd RxS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3":      "Enable AMD Winograd RxS F2x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1":   "Enable AMD Winograd RxS F2x3 G1",
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD":         "Enable AMD Fused Winograd",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3": "Enable AMD Winograd Fury RxS F2x3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2": "Enable AMD Winograd Fury RxS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3": "Enable AMD Winograd Rage RxS F2x3",
})
_SOLVER_DESCS.update({
    # Multi-pass Winograd — only F3x2/F3x3 match typical 3x3 SD shapes; wider kernels removed
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2": "Enable AMD Winograd MPASS F3x2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3": "Enable AMD Winograd MPASS F3x3",
})
_SOLVER_DESCS.update({
    # Implicit GEMM FWD — BWD/WRW (training), all XDLOPS/CDNA variants removed
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1":     "Enable ASM Implicit GEMM FWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1": "Enable ASM Implicit GEMM FWD V4R1 1x1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1":     "Enable HIP Implicit GEMM FWD V4R1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4":     "Enable HIP Implicit GEMM FWD V4R4",
})


# Solvers still in the registry but disabled by default.
#   FORCE_IMMED_MODE_FALLBACK — overrides FIND_MODE entirely, defeats tuning DB
#   Fury RxS F2x3/F3x2       — RDNA3/4-only; harmless on RDNA2 but won't select
#   Rage RxS F2x3            — RDNA4-only
SOLVER_DISABLED_BY_DEFAULT = {
    "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3",
}

SOLVER_DTYPE_TAGS: Dict[str, str] = {
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U":                  "FP16/FP32",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U":                  "FP16/FP32",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2":                "FP16/FP32",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD":                   "FP16/FP32",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1":                "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3":                      "FP32",
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD":                    "FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS":                      "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD":              "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2":                 "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3":                 "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1":              "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3":            "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2":            "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3":            "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2":               "FP16/FP32",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3":               "FP16/FP32",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1":       "FP16/FP32",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1":   "FP16/FP32",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1":       "FP16/FP32",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4":       "FP16/FP32",
}

# Build full merged var registry
ROCM_ENV_VARS: Dict[str, Dict[str, Any]] = {}
ROCM_ENV_VARS.update(GENERAL_VARS)
for _var, _desc in _SOLVER_DESCS.items():
    ROCM_ENV_VARS[_var] = {
        "default": "0" if _var in SOLVER_DISABLED_BY_DEFAULT else "1",
        "desc": _desc,
        "widget": "checkbox",
        "options": None,
        "dtype": SOLVER_DTYPE_TAGS.get(_var),
        "restart_required": False,
    }

# UI group ordering for solver sections
SOLVER_GROUPS: List[Tuple[str, List[str]]] = [
    ("Algorithm/Solver Group Enables", [
        "MIOPEN_DEBUG_CONV_FFT", "MIOPEN_DEBUG_CONV_DIRECT", "MIOPEN_DEBUG_CONV_GEMM",
        "MIOPEN_DEBUG_CONV_WINOGRAD", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM",
    ]),
    ("Immediate Fallback Mode", [
        "MIOPEN_DEBUG_CONV_IMMED_FALLBACK", "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK",
        "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK",
    ]),
    ("Build Method Toggles", [
        "MIOPEN_DEBUG_GCN_ASM_KERNELS", "MIOPEN_DEBUG_HIP_KERNELS",
        "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS", "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP",
        "MIOPEN_DEBUG_ATTN_SOFTMAX",
    ]),
    ("Direct ASM Solver Toggles", [
        "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U",
        "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2",
        "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR",
    ]),
    ("Direct OpenCL Solver Toggles", [
        "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD", "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1",
    ]),
    ("Winograd Solver Toggles", [
        "MIOPEN_DEBUG_AMD_WINOGRAD_3X3", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS",
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD",
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1", "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD",
        "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3",
    ]),
    ("Multi-pass Winograd Toggles", [
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2", "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3",
    ]),
    ("Implicit GEMM Toggles", [
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1",
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4",
    ]),
]

