import torch
import einops
from modules import shared, devices


latent_rgb_factors = [ # from comfyui
    [-0.0395, -0.0331, 0.0445],
    [0.0696, 0.0795, 0.0518],
    [0.0135, -0.0945, -0.0282],
    [0.0108, -0.0250, -0.0765],
    [-0.0209, 0.0032, 0.0224],
    [-0.0804, -0.0254, -0.0639],
    [-0.0991, 0.0271, -0.0669],
    [-0.0646, -0.0422, -0.0400],
    [-0.0696, -0.0595, -0.0894],
    [-0.0799, -0.0208, -0.0375],
    [0.1166, 0.1627, 0.0962],
    [0.1165, 0.0432, 0.0407],
    [-0.2315, -0.1920, -0.1355],
    [-0.0270, 0.0401, -0.0821],
    [-0.0616, -0.0997, -0.0727],
    [0.0249, -0.0469, -0.1703]
]
latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]
vae_weight = None
vae_bias = None
taesd = None


def vae_decode_simple(latents):
    global vae_weight, vae_bias # pylint: disable=global-statement
    with devices.inference_context():
        if vae_weight is None or vae_bias is None:
            vae_weight = torch.tensor(latent_rgb_factors, device=devices.device, dtype=devices.dtype).transpose(0, 1)[:, :, None, None, None]
            vae_bias = torch.tensor(latent_rgb_factors_bias, device=devices.device, dtype=devices.dtype)
        images = torch.nn.functional.conv3d(latents, weight=vae_weight, bias=vae_bias, stride=1, padding=0, dilation=1, groups=1)
        images = (images + 1.2) * 100 # sort-of normalized
        images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
        images = images.to(torch.uint8).detach().cpu().numpy().clip(0, 255)
    return images


def vae_decode_tiny(latents):
    global taesd # pylint: disable=global-statement
    if taesd is None:
        from modules import sd_vae_taesd
        taesd = sd_vae_taesd.get_model(variant='TAE HunyuanVideo')
        shared.log.debug(f'Video VAE: type=Tiny cls={taesd.__class__.__name__} latents={latents.shape}')
    with devices.inference_context():
        taesd = taesd.to(device=devices.device, dtype=devices.dtype)
        latents = latents.transpose(1, 2) # pipe produces NCTHW and tae wants NTCHW
        images = taesd.decode_video(latents, parallel=False, show_progress_bar=False)
        images = images.transpose(1, 2).mul_(2).sub_(1) # normalize
        taesd = taesd.to(device=devices.cpu, dtype=devices.dtype)
    return images


def vae_decode_remote(latents):
    # from modules.sd_vae_remote import remote_decode
    # images = remote_decode(latents, model_type='hunyuanvideo')
    from diffusers.utils.remote_utils import remote_decode
    images = remote_decode(
        tensor=latents.contiguous(),
        endpoint='https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud',
        output_type='pt',
        return_type='pt',
    )
    return images


def vae_decode_full(latents):
    with devices.inference_context():
        vae = shared.sd_model.vae
        latents = (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
        images = vae.decode(latents).sample
    return images


def vae_decode(latents, vae_type):
    # Track framepack VAE decode operation
    from modules import pipeline_viz
    pipeline_viz.safe_track_operation('vae_decode', {
        'latents_shape': str(latents.shape) if hasattr(latents, 'shape') else None,
        'vae_type': f'framepack_{vae_type.lower()}',
        'framepack_variant': vae_type
    })
    
    latents = latents.to(device=devices.device, dtype=devices.dtype)
    try:
        if vae_type == 'Tiny':
            result = vae_decode_tiny(latents)
        elif vae_type == 'Preview':
            result = vae_decode_simple(latents)
        elif vae_type == 'Remote':
            result = vae_decode_remote(latents)
        else: # vae_type == 'Full'
            result = vae_decode_full(latents)
            
        # Complete framepack VAE decode tracking
        pipeline_viz.safe_track_operation_complete('vae_decode', {
            'success': result is not None,
            'output_shape': str(result.shape) if hasattr(result, 'shape') else None,
            'vae_type': f'framepack_{vae_type.lower()}'
        })
        
        return result
    except Exception as e:
        shared.log.error(f'Framepack VAE decode: {e}')
        pipeline_viz.safe_track_operation_fail('vae_decode', str(e))
        return None


def vae_encode(image):
    # Track framepack VAE encode operation
    from modules import pipeline_viz
    pipeline_viz.safe_track_operation('vae_encode', {
        'image_shape': str(image.shape) if hasattr(image, 'shape') else None,
        'vae_type': 'framepack_full'
    })
    
    try:
        with devices.inference_context():
            vae = shared.sd_model.vae
            latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
        # Complete framepack VAE encode tracking
        pipeline_viz.safe_track_operation_complete('vae_encode', {
            'success': latents is not None,
            'output_shape': str(latents.shape) if hasattr(latents, 'shape') else None,
            'vae_type': 'framepack_full'
        })
        
        return latents
    except Exception as e:
        shared.log.error(f'Framepack VAE encode: {e}')
        pipeline_viz.safe_track_operation_fail('vae_encode', str(e))
        return None
