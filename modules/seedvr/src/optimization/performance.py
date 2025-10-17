"""
Performance optimization module for SeedVR2
Contains optimized tensor operations and video processing functions

Extracted from: seedvr2.py (lines 1633-1730)
"""

import torch
from typing import List, Union


def optimized_video_rearrange(video_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    🚀 OPTIMIZED version of video rearrangement
    Replaces slow loops with vectorized operations

    Transforms:
    - 3D: c h w -> t c h w (with t=1)
    - 4D: c t h w -> t c h w

    Expected gains: 5-10x faster than naive loops

    Args:
        video_tensors: List of video tensors to rearrange

    Returns:
        List of rearranged tensors in t c h w format
    """
    if not video_tensors:
        return []

    # 🔍 Analyze dimensions to optimize processing
    videos_3d = []
    videos_4d = []
    indices_3d = []
    indices_4d = []

    for i, video in enumerate(video_tensors):
        if video.ndim == 3:
            videos_3d.append(video)
            indices_3d.append(i)
        else:  # ndim == 4
            videos_4d.append(video)
            indices_4d.append(i)

    # 🎯 Prepare final result
    samples = [None] * len(video_tensors)

    # 🚀 BATCH PROCESSING for 3D videos (c h w -> 1 c h w)
    if videos_3d:
        # Method 1: Stack + permute (faster than rearrange)
        # c h w -> c 1 h w -> 1 c h w
        batch_3d = torch.stack([v.unsqueeze(1) for v in videos_3d])  # [batch, c, 1, h, w]
        batch_3d = batch_3d.permute(0, 2, 1, 3, 4)  # [batch, 1, c, h, w]

        for i, idx in enumerate(indices_3d):
            samples[idx] = batch_3d[i]  # [1, c, h, w]

    # 🚀 BATCH PROCESSING for 4D videos (c t h w -> t c h w)
    if videos_4d:
        # Check if all 4D videos have the same shape for maximum optimization
        shapes = [v.shape for v in videos_4d]
        if len(set(shapes)) == 1:
            # 🎯 MAXIMUM OPTIMIZATION: All shapes identical
            # Stack + permute in single operation
            batch_4d = torch.stack(videos_4d)  # [batch, c, t, h, w]
            batch_4d = batch_4d.permute(0, 2, 1, 3, 4)  # [batch, t, c, h, w]

            for i, idx in enumerate(indices_4d):
                samples[idx] = batch_4d[i]  # [t, c, h, w]
        else:
            # 🔄 FALLBACK: Different shapes, optimized individual processing
            for i, idx in enumerate(indices_4d):
                # Use permute instead of rearrange (faster)
                samples[idx] = videos_4d[i].permute(1, 0, 2, 3)  # c t h w -> t c h w

    return samples


def optimized_single_video_rearrange(video: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED version for single video tensor
    Replaces rearrange() with native PyTorch operations

    Transforms:
    - 3D: c h w -> 1 c h w (add temporal dimension)
    - 4D: c t h w -> t c h w (permute dimensions)

    Expected gains: 2-5x faster than rearrange()

    Args:
        video: Input video tensor

    Returns:
        Rearranged tensor with temporal dimension first
    """
    if video.ndim == 3:
        # c h w -> 1 c h w (add temporal dimension t=1)
        return video.unsqueeze(0)
    else:  # ndim == 4
        # c t h w -> t c h w (permute channels and temporal)
        return video.permute(1, 0, 2, 3)


def optimized_sample_to_image_format(sample: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED version to convert sample to image format
    Replaces rearrange() with native PyTorch operations

    Transforms:
    - 3D: c h w -> 1 h w c (add temporal dimension + permute to image format)
    - 4D: t c h w -> t h w c (permute to image format)

    Expected gains: 2-5x faster than rearrange()

    Args:
        sample: Input sample tensor

    Returns:
        Tensor in image format (channels last)
    """
    if sample.ndim == 3:
        # c h w -> 1 h w c (add temporal dimension then permute)
        return sample.unsqueeze(0).permute(0, 2, 3, 1)
    else:  # ndim == 4
        # t c h w -> t h w c (permute channels to last)
        return sample.permute(0, 2, 3, 1)


def temporal_latent_blending(latents1: torch.Tensor, latents2: torch.Tensor, blend_frames: int) -> torch.Tensor:
    """
    🎨 Temporal blending in latent space to avoid discontinuities

    Args:
        latents1: Latents from previous batch (end frames)
        latents2: Latents from current batch (start frames)
        blend_frames: Number of frames to blend

    Returns:
        Blended latents for smooth transition
    """
    if latents1.shape[0] != latents2.shape[0]:
        # Adjust dimensions if necessary
        min_frames = min(latents1.shape[0], latents2.shape[0])
        latents1 = latents1[:min_frames]
        latents2 = latents2[:min_frames]

    # Create linear blending weights
    # Frame 0: 100% latents1, 0% latents2
    # Frame n: 0% latents1, 100% latents2
    weights1 = torch.linspace(1.0, 0.0, blend_frames).view(-1, 1, 1, 1).to(latents1.device)
    weights2 = torch.linspace(0.0, 1.0, blend_frames).view(-1, 1, 1, 1).to(latents2.device)

    # Apply blending
    blended_latents = weights1 * latents1 + weights2 * latents2

    return blended_latents
