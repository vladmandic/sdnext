"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""

import os
import torch
import gc
from typing import Tuple, Optional
from src.common.cache import Cache
from src.models.dit_v2.rope import RotaryEmbeddingBase


def preinitialize_rope_cache(runner) -> None:
    """
    🚀 Pre-initialize RoPE cache to avoid OOM at first launch
    
    Args:
        runner: The model runner containing DiT and VAE models
    """

    try:
        # Create dummy tensors to simulate common shapes
        # Format: [batch, channels, frames, height, width] for vid_shape
        # Format: [batch, seq_len] for txt_shape
        common_shapes = [
            # Common video resolutions
            (torch.tensor([[1, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 1 frame, 77 tokens
            (torch.tensor([[4, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 4 frames
            (torch.tensor([[5, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 5 frames (4n+1 format)
            (torch.tensor([[1, 4, 4]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # Higher resolution
        ]

        # Create mock cache for pre-initialization

        temp_cache = Cache()

        # Access RoPE modules in DiT (recursive search)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(child.get_freqs):
                    rope_modules.append((name, child))
            return rope_modules

        rope_modules = find_rope_modules(runner.dit)

        # Pre-calculate for each RoPE module found
        for name, rope_module in rope_modules:
            # Temporarily move module to CPU if necessary
            original_device = next(rope_module.parameters()).device if list(rope_module.parameters()) else torch.device('cpu')
            rope_module.to('cpu')

            try:
                for vid_shape, txt_shape in common_shapes:
                    cache_key = f"720pswin_by_size_bysize_{tuple(vid_shape[0].tolist())}_sd3.mmrope_freqs_3d"

                    def compute_freqs():
                        # Calculate with reduced dimensions to avoid OOM
                        with torch.no_grad():
                            # Detect RoPE module type
                            module_type = type(rope_module).__name__

                            if module_type == 'NaRotaryEmbedding3d':
                                # NaRotaryEmbedding3d: only takes shape (vid_shape)
                                return rope_module.get_freqs(vid_shape.cpu())
                            else:
                                # Standard RoPE: takes vid_shape and txt_shape
                                return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())

                    # Store in cache
                    temp_cache(cache_key, compute_freqs)

            except Exception as e:
                print(f"    ❌ Error in module {name}: {e}")
            finally:
                # Restore to original device
                rope_module.to(original_device)

        # Copy temporary cache to runner cache
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache

    except Exception as e:
        print(f"  ⚠️ Error during RoPE pre-init: {e}")
        print("  🔄 Model will work but could have OOM at first launch")


def clear_rope_cache(runner) -> None:
    """
    🧹 Clear RoPE cache to free VRAM
    
    Args:
        runner: The model runner containing the cache
    """
    print("🧹 Cleaning RoPE cache...")

    if hasattr(runner, 'cache') and hasattr(runner.cache, 'cache'):
        # Count entries before cleanup
        cache_size = len(runner.cache.cache)

        # Free all tensors from cache
        for key, value in runner.cache.cache.items():
            if isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, 'cpu'):
                        item.cpu()
                        del item
            elif hasattr(value, 'cpu'):
                value.cpu()
                del value

        # Clear the cache
        runner.cache.cache.clear()
        print(f"  ✅ RoPE cache cleared ({cache_size} entries removed)")

    if hasattr(runner, 'dit'):
        cleared_lru_count = 0
        for module in runner.dit.modules():
            if isinstance(module, RotaryEmbeddingBase):
                if hasattr(module.get_axial_freqs, 'cache_clear'):
                    module.get_axial_freqs.cache_clear()
                    cleared_lru_count += 1
        if cleared_lru_count > 0:
            print(f"  ✅ Cleared {cleared_lru_count} LRU caches from RoPE modules.")

    print("🎯 RoPE cache cleanup completed!")
