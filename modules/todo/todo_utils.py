import torch
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
from modules.todo.todo_merge import TokenMergeAttentionProcessor


xformers_is_available = is_xformers_available()
torch2_is_available = hasattr(F, "scaled_dot_product_attention")


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3]) # pylint: disable=protected-access
        module._tome_info["timestep"] = args[1].item() # pylint: disable=protected-access
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook)) # pylint: disable=protected-access

def remove_tome_patch(pipe: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    if hasattr(pipe.unet, "_tome_info"):
        del pipe.unet._tome_info

    for _n, m in pipe.unet.named_modules():
        if hasattr(m, "processor"):
            m.processor = AttnProcessor2_0()

def patch_attention_proc(unet, token_merge_args=None):
    if token_merge_args is None:
        token_merge_args = {}
    unet._tome_info = { # pylint: disable=protected-access
        "size": None,
        "timestep": None,
        "hooks": [],
        "args": {
            "ratio": token_merge_args.get("ratio", 0.5),  # ratio of tokens to merge
            "sx": token_merge_args.get("sx", 2),  # stride x for sim calculation
            "sy": token_merge_args.get("sy", 2),  # stride y for sim calculation
            "use_rand": token_merge_args.get("use_rand", True),
            "generator": None,
            "merge_tokens": token_merge_args.get("merge_tokens", "keys/values"),  # ["all", "keys/values"]
            "merge_method": token_merge_args.get("merge_method", "downsample"),  # ["none","similarity", "downsample"]
            "downsample_method": token_merge_args.get("downsample_method", "nearest-exact"), # native torch interpolation methods ["nearest", "linear", "bilinear", "bicubic", "nearest-exact"]
            "downsample_factor": token_merge_args.get("downsample_factor", 2),  # amount to downsample by
            "timestep_threshold_switch": token_merge_args.get("timestep_threshold_switch", 0.2), # timestep to switch to secondary method, 0.2 means 20% steps remaining
            "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.0), # timestep to stop merging, 0.0 means stop at 0 steps remaining
            "secondary_merge_method": token_merge_args.get("secondary_merge_method", "similarity"), # ["none", "similarity", "downsample"]
            "downsample_factor_level_2": token_merge_args.get("downsample_factor_level_2", 1), # amount to downsample by at the 2nd down block of unet
            "ratio_level_2": token_merge_args.get("ratio_level_2", 0.5), # ratio of tokens to merge at the 2nd down block of unet
        }
    }
    hook_tome_model(unet)
    attn_modules = [module for name, module in unet.named_modules() if module.__class__.__name__ == 'BasicTransformerBlock']

    for _i, module in enumerate(attn_modules):
        module.attn1.processor = TokenMergeAttentionProcessor()
        module.attn1.processor._tome_info = unet._tome_info # pylint: disable=protected-access


def remove_patch(pipe: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    # this will remove our custom class
    if torch2_is_available:
        for _n, m in pipe.unet.named_modules():
            if hasattr(m, "processor"):
                m.processor = AttnProcessor2_0()

    elif xformers_is_available:
        pipe.enable_xformers_memory_efficient_attention()

    else:
        for _n, m in pipe.unet.named_modules():
            if hasattr(m, "processor"):
                m.processor = AttnProcessor()
