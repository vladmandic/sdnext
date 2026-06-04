"""Packed-sequence role indicators and Qwen3-VL tap layers for Ideogram 4.

Text and image latent tokens share one sequence; each token carries a role
indicator, and image grid positions are offset so they never collide with text
token positions in the shared MRoPE space.
"""

from __future__ import annotations

SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

IMAGE_POSITION_OFFSET = 65536  # keeps image grid positions clear of text token indices

# Qwen3-VL hidden-state layers whose outputs are concatenated and fed to the DiT.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
