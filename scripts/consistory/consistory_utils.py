# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

from typing import List
from collections import defaultdict
import numpy as np
import torch
from .utils.general_utils import get_dynamic_threshold


class FeatureInjector:
    def __init__(self, nn_map, nn_distances, attn_masks, inject_range_alpha=[(10,20,0.8)], swap_strategy='min', dist_thr='dynamic', inject_unet_parts=['up']):
        self.nn_map = nn_map
        self.nn_distances = nn_distances
        self.attn_masks = attn_masks
        self.inject_range_alpha = inject_range_alpha if isinstance(inject_range_alpha, list) else [inject_range_alpha]
        self.swap_strategy = swap_strategy # 'min / 'mean' / 'first'
        self.dist_thr = dist_thr
        self.inject_unet_parts = inject_unet_parts
        self.inject_res = [64]

    def inject_outputs(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache=None):
        curr_unet_part = place_in_unet.split('_')[0]

        # Inject only in the specified unet parts (up, mid, down)
        if (curr_unet_part not in self.inject_unet_parts) or output_res not in self.inject_res:
            return output

        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res**2

        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha if min_range <= curr_iter <= max_range), None)
        if alpha:
            old_output = output#.clone()
            for i in range(bsz):
                other_outputs = []

                if self.swap_strategy == 'min':
                    curr_mapping = extended_mapping[i]

                    # If the current image is not mapped to any other image, skip
                    if not torch.any(torch.cat([curr_mapping[:i], curr_mapping[i+1:]])):
                        continue

                    min_dists = nn_distances[i][curr_mapping].argmin(dim=0)
                    curr_nn_map = nn_map[i][curr_mapping][min_dists, torch.arange(vector_dim)]

                    curr_nn_distances = nn_distances[i][curr_mapping][min_dists, torch.arange(vector_dim)]
                    dist_thr = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < dist_thr
                    final_mask_tgt = attn_masks[i] & dist_mask

                    other_outputs = old_output[curr_mapping][min_dists, curr_nn_map][final_mask_tgt]

                    output[i][final_mask_tgt] = alpha * other_outputs + (1 - alpha)*old_output[i][final_mask_tgt]

            if anchors_cache and anchors_cache.is_cache_mode():
                if place_in_unet not in anchors_cache.h_out_cache:
                    anchors_cache.h_out_cache[place_in_unet] = {}

                anchors_cache.h_out_cache[place_in_unet][curr_iter] = output

        return output

    def inject_anchors(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache):
        curr_unet_part = place_in_unet.split('_')[0]

        # Inject only in the specified unet parts (up, mid, down)
        if (curr_unet_part not in self.inject_unet_parts) or output_res not in self.inject_res:
            return output

        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res**2

        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha if min_range <= curr_iter <= max_range), None)
        if alpha:

            anchor_outputs = anchors_cache.h_out_cache[place_in_unet][curr_iter]

            old_output = output#.clone()
            for i in range(bsz):
                other_outputs = []

                if self.swap_strategy == 'min':
                    min_dists = nn_distances[i].argmin(dim=0)
                    curr_nn_map = nn_map[i][min_dists, torch.arange(vector_dim)]

                    curr_nn_distances = nn_distances[i][min_dists, torch.arange(vector_dim)]
                    dist_thr = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < dist_thr
                    final_mask_tgt = attn_masks[i] & dist_mask

                    other_outputs = anchor_outputs[min_dists, curr_nn_map][final_mask_tgt]

                    output[i][final_mask_tgt] = alpha * other_outputs + (1 - alpha)*old_output[i][final_mask_tgt]

        return output


class AnchorCache:
    def __init__(self):
        self.input_h_cache = {} # place_in_unet, iter, h_in
        self.h_out_cache = {} # place_in_unet, iter, h_out
        self.anchors_last_mask = None
        self.dift_cache = None

        self.mode = 'cache' # mode can be 'cache' or 'inject'

    def set_mode(self, mode):
        self.mode = mode

    def set_mode_inject(self):
        self.mode = 'inject'

    def set_mode_cache(self):
        self.mode = 'cache'

    def is_inject_mode(self):
        return self.mode == 'inject'

    def is_cache_mode(self):
        return self.mode == 'cache'


    def to_device(self, device):
        for key, value in self.input_h_cache.items():
            self.input_h_cache[key] = {k: v.to(device) for k, v in value.items()}

        for key, value in self.h_out_cache.items():
            self.h_out_cache[key] = {k: v.to(device) for k, v in value.items()}

        if self.anchors_last_mask:
            self.anchors_last_mask = {k: v.to(device) for k, v in self.anchors_last_mask.items()}

        if self.dift_cache is not None:
            self.dift_cache = self.dift_cache.to(device)


class QueryStore:
    def __init__(self, mode='store', t_range=[0, 1000], strength_start=1, strength_end=1):
        """
        Initialize an empty ActivationsStore
        """
        self.query_store = defaultdict(list)
        self.mode = mode
        self.t_range = t_range
        self.strengthes = np.linspace(strength_start, strength_end, (t_range[1] - t_range[0])+1)

    def set_mode(self, mode): # mode can be 'cache' or 'inject'
        self.mode = mode

    def cache_query(self, query, place_in_unet: str):
        self.query_store[place_in_unet] = query

    def inject_query(self, query, place_in_unet, t):
        if t >= self.t_range[0] and t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            new_query = strength * self.query_store[place_in_unet] + (1 - strength) * query
        else:
            new_query = query

        return new_query

class DIFTLatentStore:
    def __init__(self, steps: List[int], up_ft_indices: List[int]):
        self.steps = steps
        self.up_ft_indices = up_ft_indices
        self.dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        if t in self.steps and layer_index in self.up_ft_indices:
            self.dift_features[f'{int(t)}_{layer_index}'] = features

    def copy(self):
        copy_dift = DIFTLatentStore(self.steps, self.up_ft_indices)

        for key, value in self.dift_features.items():
            copy_dift.dift_features[key] = value.clone()

        return copy_dift

    def reset(self):
        self.dift_features = {}
