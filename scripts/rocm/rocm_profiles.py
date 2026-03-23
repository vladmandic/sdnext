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
    # Core algo enables — FFT is FP32-only but harmless (IsApplicable rejects it for fp16 tensors)
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
    # Direct ASM — dtype notes
    # 3X3U / 1X1U / 1X1UV2: FP32/FP16 forward — enabled
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U":                    "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U":                    "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2":                  "1",
    # 5X10U2V2: fixed geometry (5×10 stride-2), no SD conv matches — disabled
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2":                "0",
    # 7X7C3H224W224: hard-coded ImageNet stem (C=3, H=W=224, K=64) — never matches SD — disabled
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224":           "0",
    # WRW3X3 / WRW1X1: FP32-only weight-gradient (training only) — disabled for inference
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3":                  "0",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1":                  "0",
    # PERF_VALS intentionally blank: MIOpen reads this as a config string not a boolean;
    # setting to "1" causes GetPerfConfFromEnv to use a degenerate config and return float32
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS":          "",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED":   "1",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR":            "1",
    # NAIVE_CONV_FWD: scalar FP32 reference solver — IsApplicable does NOT reliably filter for FP16;
    # can be selected for unusual shapes (e.g. VAE decoder 3-ch output) and returns dtype=float32
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD":               "0",
    # Direct OCL — dtype notes
    # FWD / FWD1X1: FP32/FP16 forward — enabled
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD":      "1",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1":   "1",
    # FWD11X11: requires 11×11 kernel — no SD match — disabled
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11": "0",
    # FWDGEN: FP32 generic OCL fallback — IsApplicable does NOT reliably reject for FP16;
    # can produce dtype=float32 output for FP16 inputs — disabled
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN":   "0",
    # WRW2 / WRW53 / WRW1X1: training-only weight-gradient — disabled
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2":     "0",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53":    "0",
    "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1":   "0",
    # Winograd RxS — dtype per MIOpen docs
    # WINOGRAD_3X3: FP32-only — harmless (IsApplicable rejects for fp16); enabled
    "MIOPEN_DEBUG_AMD_WINOGRAD_3X3":                "1",
    # RXS: covers FP32/FP16 F(3,3) Fwd/Bwd + FP32 F(3,2) WrW — keep enabled (fp16 fwd/bwd path exists)
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS":                "1",
    # RXS_FWD_BWD: FP32/FP16 — explicitly the fp16-capable subset
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD":        "1",
    # RXS_WRW: FP32 WrW only — training-only, disabled for inference fp16 profile
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW":            "0",
    # RXS_F3X2: FP32/FP16 Fwd/Bwd
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2":           "1",
    # RXS_F2X3: FP32/FP16 Fwd/Bwd (group convolutions)
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3":           "1",
    # RXS_F2X3_G1: FP32/FP16 Fwd/Bwd (non-group convolutions)
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1":        "1",
    # FUSED_WINOGRAD: FP32-only — harmless (IsApplicable rejects for fp16); enabled
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD":              "1",
    # PERF_VALS intentionally blank: same reason as ASM_1X1U — not a boolean, config string
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS": "",
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
    # ASM Implicit GEMM — forward V4R1 only; no GTC/XDLOPS on RDNA2
    # BWD (backward data-gradient) and WrW (weight-gradient) are training-only — disabled
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1":     "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1":     "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1":     "0",
    # HIP Implicit GEMM — non-XDLOPS V4R1/R4 forward only
    # BWD (backward data-gradient) and WrW (weight-gradient) are training-only — disabled
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4": "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1": "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1": "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1": "0",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4": "0",
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
