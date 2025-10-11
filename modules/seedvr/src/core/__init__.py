"""
Core Module for SeedVR2

Contains the main business logic and model management functionality:
- Model configuration and loading
- Architecture detection and memory estimation
- Runner creation and management
- Generation pipeline and logic
"""
'''
from .model_manager import (
    configure_runner,
    load_quantized_state_dict,
    configure_dit_model_inference,
    configure_vae_model_inference,
)

from .generation import (
    generation_step,
    generation_loop,
    cut_videos,
    prepare_video_transforms,
    load_text_embeddings,
    calculate_optimal_batch_params
)

from .infer import VideoDiffusionInfer

__all__ = [
    # Model management
    'configure_runner',
    'load_quantized_state_dict',
    'configure_dit_model_inference',
    'configure_vae_model_inference',

    # Generation logic
    'generation_step',
    'generation_loop',
    'cut_videos',
    'prepare_video_transforms',
    'load_text_embeddings',
    'calculate_optimal_batch_params',

    # Infer
    'VideoDiffusionInfer'
]
'''
