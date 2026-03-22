"""
Architecture-specific MIOpen solver profiles for AMD GCN/RDNA GPUs.

Sources:
  https://rocm.docs.amd.com/projects/MIOpen/en/develop/reference/env_variables.html

Key axis: consumer RDNA GPUs have NO XDLOPS hardware (that's CDNA/Instinct only).
  RDNA2 (gfx1030): RX 6000 series
  RDNA3 (gfx1100): RX 7000 series — adds Fury Winograd, wider MPASS
  RDNA4 (gfx1200): RX 9000 series — adds Rage Winograd, wider MPASS

Each profile is a dict of {var: value} that will be MERGED on top of the
current config (general vars like DB path / log level are preserved).
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Shared: everything that must be OFF on ALL consumer RDNA (no XDLOPS hw)
# ---------------------------------------------------------------------------
_XDLOPS_OFF: Dict[str, str] = {
    # GTC XDLOPS (CDNA-only)
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS":                                "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS":                                "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS":                                "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC":                          "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC":                          "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS_NHWC":                          "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC":                          "0",
    # HIP XDLOPS variants (CDNA-only)
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS":                              "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS":                              "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS":                              "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS":                              "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_XDLOPS":                              "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_PADDED_GEMM_XDLOPS":                 "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_PADDED_GEMM_XDLOPS":                 "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS":                                   "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS":                                   "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS":                                   "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS":                                            "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE":                                   "0",
    "MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM":                                     "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS":                             "0",
    "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR":                     "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_FWD_V4R4_XDLOPS_ADD_VECTOR_LOAD_GEMMN_TUNE_PARAM": "0",
    # 3D XDLOPS (CDNA-only; no 3D conv XDLOPS on consumer RDNA)
    "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS":                                "0",
    "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS":                                "0",
    "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS":                                "0",
    # Composable Kernel (requires XDLOPS / CDNA)
    "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW":                                  "0",
    "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV":                                        "0",
    "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_RES_ADD_ACTIV":                                "0",
    # MLIR (CDNA-only in practice)
    "MIOPEN_DEBUG_CONV_MLIR_IGEMM_WRW_XDLOPS":                                          "0",
    "MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD_XDLOPS":                                          "0",
    # MP BD Winograd (Multi-pass Block-Decomposed — CDNA / high-end only)
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3":                                              "0",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3":                                              "0",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3":                                              "0",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3":                                              "0",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3":                                              "0",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3":                                      "0",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3":                                      "0",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3":                                      "0",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3":                                      "0",
    "MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3":                                      "0",
}

# ---------------------------------------------------------------------------
# RDNA2 — gfx1030 (RX 6000 series)
# No XDLOPS, no Fury/Rage Winograd, MPASS limited to F3x2/F3x3
# ASM IGEMM: V4R1 variants only; HIP IGEMM: non-XDLOPS V4R1/R4 only
# ---------------------------------------------------------------------------
RDNA2: Dict[str, str] = {
    **_XDLOPS_OFF,
    # General settings (architecture-independent; set here so all profiles cover them)
    "MIOPEN_SEARCH_CUTOFF": "0",
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": "0",
    # Core algo enables
    "MIOPEN_DEBUG_CONV_FFT":               "1",
    "MIOPEN_DEBUG_CONV_DIRECT":            "1",
    "MIOPEN_DEBUG_CONV_GEMM":              "1",
    "MIOPEN_DEBUG_CONV_WINOGRAD":          "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM":     "1",
    "MIOPEN_DEBUG_CONV_IMMED_FALLBACK":    "1",
    "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "1",
    "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK":     "0",
    # Kernel backends
    "MIOPEN_DEBUG_GCN_ASM_KERNELS":        "1",
    "MIOPEN_DEBUG_HIP_KERNELS":            "1",
    "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS":    "1",
    "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP":    "1",
    "MIOPEN_DEBUG_ATTN_SOFTMAX":           "1",
    # Direct ASM — all work on RDNA2
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U":                    "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U":                    "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2":                  "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2":                "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224":           "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3":                  "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1":                  "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS":          "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED":   "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR":            "1",
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD":               "1",
    # Direct OCL — all work on RDNA2
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD":      "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1":   "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11": "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN":   "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2":     "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53":    "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1":   "1",
    # Winograd RxS — all base variants work on RDNA2
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3":                "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS":                "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD":        "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW":            "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2":           "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3":           "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1":        "1",
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD":              "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS": "1",
    # Fury/Rage Winograd — NOT available on RDNA2
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3": "0",
    # MPASS — only F3x2 and F3x3 are safe on RDNA2
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2": "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3": "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2": "0",
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3": "0",
    # ASM Implicit GEMM — V4R1 only; no GTC/XDLOPS on RDNA2
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1":     "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1":     "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1":     "1",
    # HIP Implicit GEMM — non-XDLOPS V4R1/R4 only
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4": "1",
}

# ---------------------------------------------------------------------------
# RDNA3 — gfx1100 (RX 7000 series)
# Fury Winograd added; MPASS F3x4 enabled
# ---------------------------------------------------------------------------
RDNA3: Dict[str, str] = {
    **RDNA2,
    # Fury Winograd — introduced for gfx1100 (RDNA3)
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3": "1",
    "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2": "1",
    # Wider MPASS on RDNA3
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4": "1",
}

# ---------------------------------------------------------------------------
# RDNA4 — gfx1200 (RX 9000 series)
# Rage Winograd added; MPASS F3x5 enabled
# ---------------------------------------------------------------------------
RDNA4: Dict[str, str] = {
    **RDNA3,
    # Rage Winograd — introduced for gfx1200 (RDNA4)
    "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3": "1",
    # Wider MPASS on RDNA4
    "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5": "1",
}

PROFILES: Dict[str, Dict[str, str]] = {
    "RDNA2": RDNA2,
    "RDNA3": RDNA3,
    "RDNA4": RDNA4,
}

# Vars that are architecturally unavailable (no supporting hardware) per arch.
# These will be visually marked in the UI with strikethrough.
_UNAVAILABLE_ALL_RDNA = set(_XDLOPS_OFF.keys())

UNAVAILABLE: Dict[str, set] = {
    "RDNA2": _UNAVAILABLE_ALL_RDNA | {
        "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2",
        "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3",
    },
    "RDNA3": _UNAVAILABLE_ALL_RDNA | {
        "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3",
    },
    "RDNA4": _UNAVAILABLE_ALL_RDNA | {
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2",
        "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3",
    },
}
