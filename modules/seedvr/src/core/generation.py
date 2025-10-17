import torch
from torchvision.transforms import Compose, Lambda, Normalize
from ..optimization.performance import optimized_video_rearrange, optimized_single_video_rearrange, optimized_sample_to_image_format
from ..common.seed import set_seed
from ..data.image.transforms.divisible_crop import DivisibleCrop
from ..data.image.transforms.na_resize import NaResize
from ..utils.color_fix import wavelet_reconstruction



def generation_step(runner, text_embeds_dict, cond_latents, temporal_overlap, device):
    """
    Execute a single generation step with adaptive dtype handling

    Args:
        runner: VideoDiffusionInfer instance
        text_embeds_dict (dict): Text embeddings for positive and negative prompts
        cond_latents (list): Conditional latents for generation
        temporal_overlap (int): Number of frames for temporal overlap

    Returns:
        tuple: (samples, last_latents) for potential temporal continuation

    Features:
        - Adaptive dtype detection (FP8/FP16/BFloat16)
        - Optimal autocast configuration for each model type
        - Memory-efficient noise generation and reuse
        - Automatic device placement with dtype preservation
        - Advanced inference optimization
    """
    # Adaptive dtype detection for optimal performance
    model_dtype = next(runner.dit.parameters()).dtype

    # Configure dtypes according to model architecture
    if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 native: use BFloat16 for intermediate calculations (optimal compatibility)
        dtype = torch.bfloat16
    elif model_dtype == torch.float16:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    def _move_to_cuda(x):
        """Move tensors to CUDA with adaptive optimal dtype"""
        return [i.to(device, dtype=dtype) for i in x]

    # Memory optimization: Generate noise once and reuse to save VRAM
    with torch.cuda.device(device):
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]

    # Move tensors with adaptive dtype (optimized for FP8/FP16/BFloat16)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)

    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Use adaptive optimal dtype
        t = (
            torch.tensor([1000.0], device=device, dtype=dtype)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    condition = runner.get_condition(
        noises[0],
        task="sr",
        latent_blur=_add_noise(cond_latents[0], aug_noises[0]),
    )
    conditions = [condition]

    with torch.no_grad():
        # Use adaptive autocast for optimal performance
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            temporal_overlap=temporal_overlap,
            **text_embeds_dict,
        )

    # Process samples with advanced optimization
    samples = optimized_video_rearrange(video_tensors)
    noises = noises[0].to("cpu")
    aug_noises = aug_noises[0].to("cpu")
    cond_latents = cond_latents[0].to("cpu")
    conditions = conditions[0].to("cpu")
    condition = condition.to("cpu")
    del noises, aug_noises, cond_latents, conditions, condition

    return samples #, last_latents


def cut_videos(videos):
    t = videos.size(1)

    if t % 4 == 1:
        return videos

    padding_needed = (4 - (t % 4)) % 4 + 1
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    return result


def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, temporal_overlap=0, progress_callback=None, device:str='cpu'):
    """
    Main generation loop with context-aware temporal processing

    Args:
        runner: VideoDiffusionInfer instance
        images (torch.Tensor): Input images for upscaling
        cfg_scale (float): Classifier-free guidance scale
        seed (int): Random seed for reproducibility
        res_w (int): Target resolution width
        batch_size (int): Batch size for processing
        temporal_overlap (int): Frames for temporal continuity
        progress_callback (callable): Optional callback for progress reporting

    Returns:
        torch.Tensor: Generated video frames

    Features:
        - Context-aware generation with temporal overlap
        - Adaptive dtype pipeline (FP8/FP16/BFloat16)
        - Memory-optimized batch processing
        - Advanced video transformation pipeline
        - Intelligent VRAM management throughout process
        - Real-time progress reporting
    """
    model_dtype = None
    model_dtype = next(runner.dit.parameters()).dtype
    compute_dtype = model_dtype

    # Configure classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # Configure sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # Set random seed
    set_seed(seed)

    # Advanced video transformation pipeline
    video_transform = Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])

    # Initialize generation state
    batch_samples = []

    # Load text embeddings with adaptive dtype
    text_embeds = {"texts_pos": [runner.text_pos_embeds], "texts_neg": [runner.text_neg_embeds]}

    # Calculate processing parameters
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0

    # Calculate total batches for progress reporting
    total_batches = len(range(0, len(images), step))

    # Main processing loop with context awareness
    for batch_count, batch_idx in enumerate(range(0, len(images), step)):
        # Calculate batch indices with overlap
        if batch_idx == 0:
            # First batch: no overlap
            start_idx = 0
            end_idx = min(batch_size, len(images))
            effective_batch_size = end_idx - start_idx
        else:
            # Subsequent batches: temporal overlap
            start_idx = batch_idx
            end_idx = min(start_idx + batch_size, len(images))
            effective_batch_size = end_idx - start_idx
            if effective_batch_size <= temporal_overlap:
                break  # Not enough new frames, stop

        current_frames = end_idx - start_idx

        # Process current batch
        video = images[start_idx:end_idx]
        # Use adaptive computation dtype
        video = video.permute(0, 3, 1, 2).to(device, dtype=compute_dtype)

        # Apply video transformations with memory optimization
        transformed_video = video_transform(video)
        del video
        #video = video.to("cpu")
        #del video
        ori_lengths = [transformed_video.size(1)]

        # Handle correct format: frames % 4 == 1
        t = transformed_video.size(1)

        if len(images) >= 5 and t % 4 != 1:
            transformed_video = cut_videos(transformed_video)

        # Context-aware temporal strategy
        # First batch: standard complete diffusion
        cond_latents = runner.vae_encode([transformed_video])

        # Normal generation
        samples = generation_step(runner, text_embeds, cond_latents=cond_latents, temporal_overlap=temporal_overlap, device=device)
        #del cond_latents
        del cond_latents

        # Post-process samples
        sample = samples[0]
        del samples
        #del samples
        if ori_lengths[0] < sample.shape[0]:
            sample = sample[:ori_lengths[0]]

        # Apply color correction if available
        transformed_video = transformed_video.to(device)
        input_video = [optimized_single_video_rearrange(transformed_video)]
        del transformed_video
        sample = wavelet_reconstruction(sample, input_video[0][:sample.size(0)])
        del input_video

        # Convert to final image format
        sample = optimized_sample_to_image_format(sample)
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
        sample_cpu = sample.to(torch.float16).to("cpu")
        del sample
        batch_samples.append(sample_cpu)
        #del sample

        if progress_callback:
            progress_callback(batch_count+1, total_batches, current_frames, "Processing batch...")


    # 1. Calculer la taille totale finale
    total_frames = sum(batch.shape[0] for batch in batch_samples)
    if len(batch_samples) > 0:
        sample_shape = batch_samples[0].shape
        H, W, C = sample_shape[1], sample_shape[2], sample_shape[3]
        final_video_images = torch.empty((total_frames, H, W, C), dtype=torch.float16)
        block_size = 500
        current_idx = 0

        for block_start in range(0, len(batch_samples), block_size):
            block_end = min(block_start + block_size, len(batch_samples))
            current_block = []
            for i in range(block_start, block_end):
                current_block.append(batch_samples[i].to(device))
            block_result = torch.cat(current_block, dim=0)
            block_frames = block_result.shape[0]
            final_video_images[current_idx:current_idx + block_frames] = block_result.to("cpu")
            current_idx += block_frames
            del current_block, block_result
    else:
        print("SeedVR2: No batch_samples to process")
        final_video_images = torch.empty((0, 0, 0, 0), dtype=torch.float16)

    return final_video_images


def prepare_video_transforms(res_w):
    """
    Prepare optimized video transformation pipeline

    Args:
        res_w (int): Target resolution width

    Returns:
        Compose: Configured transformation pipeline

    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            downsample_only=False,  # Model trained for high resolution
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w
    ])


def calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap):
    """
    Calculate optimal batch processing parameters

    Args:
        total_frames (int): Total number of frames
        batch_size (int): Desired batch size
        temporal_overlap (int): Temporal overlap frames

    Returns:
        dict: Optimized parameters and recommendations

    Features:
        - 4n+1 constraint optimization
        - Padding waste calculation
        - Performance recommendations
    """
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0

    # Find optimal batch sizes (4n+1 constraint)
    optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
    best_batch = max(optimal_batches) if optimal_batches else 1

    # Calculate potential padding waste
    padding_waste = 0
    if batch_size not in optimal_batches:
        padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))

    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': batch_size in optimal_batches
    }
