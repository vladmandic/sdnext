# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import torch
import torch.nn.functional as F
import numpy as np


## Attention Utils
def get_dynamic_threshold(tensor):
    from skimage import filters
    return filters.threshold_otsu(tensor.float().cpu().numpy())


def attn_map_to_binary(attention_map, scaler=1.):
    from skimage import filters
    attention_map_np = attention_map.float().cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask


## Features

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Function to apply Gaussian smoothing on each 2D slice of a 3D tensor.
    """
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                      np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.Tensor(kernel / kernel.sum()).to(input_tensor.dtype).to(input_tensor.device)
    # Add batch and channel dimensions to the kernel
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    # Iterate over each 2D slice and apply convolution
    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        slice_tensor = input_tensor[i, :, :]
        slice_tensor = F.conv2d(slice_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)[0, 0]
        smoothed_slices.append(slice_tensor)
    # Stack the smoothed slices to get the final tensor
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)
    return smoothed_tensor


## Dense correspondence utils

def cos_dist(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T
    return 1 - res


def gen_nn_map(src_features, src_mask,  tgt_features, tgt_mask, device, batch_size=100, tgt_size=768):
    resized_src_features = F.interpolate(src_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_src_features = resized_src_features.permute(1,2,0).view(tgt_size**2, -1)
    resized_tgt_features = F.interpolate(tgt_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_tgt_features = resized_tgt_features.permute(1,2,0).view(tgt_size**2, -1)
    nearest_neighbor_indices = torch.zeros(tgt_size**2, dtype=torch.long, device=device)
    nearest_neighbor_distances = torch.zeros(tgt_size**2, dtype=src_features.dtype, device=device)
    if not batch_size:
        batch_size = tgt_size**2
    for i in range(0, tgt_size**2, batch_size):
        distances = cos_dist(resized_src_features, resized_tgt_features[i:i+batch_size])
        distances[~src_mask] = 2.
        min_distances, min_indices = torch.min(distances, dim=0)
        nearest_neighbor_indices[i:i+batch_size] = min_indices
        nearest_neighbor_distances[i:i+batch_size] = min_distances
    return nearest_neighbor_indices, nearest_neighbor_distances


def cyclic_nn_map(features, masks, latent_resolutions, device):
    bsz = features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, bsz, tgt_size**2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, bsz, tgt_size**2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(bsz):
                if i != j:
                    nearest_neighbor_indices, nearest_neighbor_distances = gen_nn_map(features[j], masks[tgt_size][j], features[i], masks[tgt_size][i], device, batch_size=None, tgt_size=tgt_size)
                    nn_map[i,j] = nearest_neighbor_indices
                    nn_distances[i,j] = nearest_neighbor_distances

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict


def anchor_nn_map(features, anchor_features, masks, anchor_masks, latent_resolutions, device):
    bsz = features.shape[0]
    anchor_bsz = anchor_features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, anchor_bsz, tgt_size**2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, anchor_bsz, tgt_size**2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(anchor_bsz):
                nearest_neighbor_indices, nearest_neighbor_distances = gen_nn_map(anchor_features[j], anchor_masks[tgt_size][j], features[i], masks[tgt_size][i], device, batch_size=None, tgt_size=tgt_size)
                nn_map[i,j] = nearest_neighbor_indices
                nn_distances[i,j] = nearest_neighbor_distances
        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict
