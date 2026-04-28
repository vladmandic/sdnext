import torch
import torch.nn.functional as F


def _clamp_fp16(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/inf in FP16 tensors to prevent overflow propagation (matches ComfyUI behaviour)."""
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return x


_applied = False


def apply_patches():
    """Replace FP16-unsafe method bodies in the diffusers ZImage transformer.

    Patches:
      - FeedForward._forward_silu_gating   : wraps SiLU gate output in _clamp_fp16
      - ZImageTransformerBlock.forward     : wraps attn/FFN outputs in _clamp_fp16
      - FinalLayer.forward                 : upstream-equivalent (adaLN calls unchanged)

    _clamp_fp16 is a no-op for bf16/fp32 — zero behavioural change for non-fp16 users.

    All replacements are closures; select_per_token is captured from the diffusers
    module at call time. No names are injected into the diffusers module namespace.

    Uses modules.patches.patch() — idempotent, reversible via modules.patches.undo().
    """
    global _applied # pylint: disable=global-statement
    if _applied:
        return
    _applied = True
    from modules import patches as sdnext_patches
    import diffusers.models.transformers.transformer_z_image as m

    _select_per_token = m.select_per_token  # upstream helper, captured by closure

    # ------------------------------------------------------------------
    # FeedForward._forward_silu_gating
    # ------------------------------------------------------------------
    def _patched_forward_silu_gating(self, x1, x3): # pylint: disable=unused-argument
        return _clamp_fp16(F.silu(x1) * x3)

    # ------------------------------------------------------------------
    # ZImageTransformerBlock.forward
    # ------------------------------------------------------------------
    def _patched_zimage_block_forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]
            if noise_mask is not None:
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)
                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)
                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()
                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean
                scale_msa = _select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = _select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa  = _select_per_token(gate_msa_noisy,  gate_msa_clean,  noise_mask, seq_len)
                gate_mlp  = _select_per_token(gate_mlp_noisy,  gate_mlp_clean,  noise_mask, seq_len)
            else:
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(_clamp_fp16(attn_out))
            x = x + gate_mlp * self.ffn_norm2(_clamp_fp16(self.feed_forward(self.ffn_norm1(x) * scale_mlp)))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(_clamp_fp16(attn_out))
            x = x + self.ffn_norm2(_clamp_fp16(self.feed_forward(self.ffn_norm1(x))))
        return x

    # ------------------------------------------------------------------
    # FinalLayer.forward
    # ------------------------------------------------------------------
    def _patched_final_layer_forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]
        if noise_mask is not None:
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = _select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)
        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x

    sdnext_patches.patch(__name__, m.FeedForward,            '_forward_silu_gating', _patched_forward_silu_gating)
    sdnext_patches.patch(__name__, m.ZImageTransformerBlock, 'forward',              _patched_zimage_block_forward)
    sdnext_patches.patch(__name__, m.FinalLayer,             'forward',              _patched_final_layer_forward)
