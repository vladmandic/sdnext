import os
from modules.logger import log


# logging
debug = os.environ.get('SD_MOVE_DEBUG', None) is not None
verbose = os.environ.get('SD_MOVE_VERBOSE', None) is not None
debug_move = log.trace if debug else lambda *args, **kwargs: None


offload_allow_none = ['sd', 'sdxl'] # used to warn if offloading=none


offload_post = ['h1']
offload_hook_instance = None # instance of sd_offload_balanced.OffloadHook
balanced_offload_exclude = ['CogView4Pipeline', 'MeissonicPipeline']
group_offload_main = [ # component names entered once per denoising step
    "unet", "transformer", "transformer_2", "transformer_ref", "unconditional_transformer",
    "prior", "prior_prior", "decoder", "dit_model", "model", "controlnet",
] # a denoiser registered under any other name takes the aux profile until listed here
offload_reapply_options = [ # settings that re-place loaded components when changed
    "group_offload_type", "group_offload_stream", "group_offload_record", "group_offload_pin", "group_offload_blocks",
    "diffusers_offload_nonblocking", "models_not_to_offload", "diffusers_offload_never", "diffusers_offload_always",
]
no_split_module_classes = [
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Embedding",
    "SDNQLinear", "SDNQConv1d", "SDNQConv2d", "SDNQConv3d", "SDNQConvTranspose1d", "SDNQConvTranspose2d", "SDNQConvTranspose3d", "SDNQEmbedding",
    "WanTransformerBlock",
    "MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock",
]

accelerate_dtype_byte_size = None # monkey-patch accelerate.utils.modeling.dtype_byte_size
group_stats_reported = set()
move_stream = None
