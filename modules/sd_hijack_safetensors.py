import safetensors.torch
import transformers
from installer import install
from modules.logger import log
from modules import errors


orig_load_file = safetensors.torch.load_file
orig_load_state_dict = transformers.modeling_utils.load_state_dict


def hijacked_load_file(checkpoint_file, device="cpu"):
    if not checkpoint_file.endswith('.safetensors'):
        return orig_load_file(checkpoint_file, device=device)

    install('runai_model_streamer>=0.15.1')
    state_dict = {}
    from runai_model_streamer import SafetensorsStreamer
    try:
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(checkpoint_file)
            for key, tensor in streamer.get_tensors():
                state_dict[key] = tensor.to(device)
    except Exception as e:
        log.error(f'Loader: {e}')
        errors.display(e, 'runai')
    return state_dict


def hijacked_load_state_dict(checkpoint_file, is_quantized: bool = False, map_location: str = "cpu", weights_only: bool = True):
    if not checkpoint_file.endswith(".safetensors"):
        return orig_load_state_dict(checkpoint_file=checkpoint_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only)

    install('runai_model_streamer>=0.15.1')
    state_dict = {}
    from runai_model_streamer import SafetensorsStreamer
    try:
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(checkpoint_file)
            for key, tensor in streamer.get_tensors():
                state_dict[key] = tensor.to(map_location) if map_location != "meta" else tensor
    except Exception as e:
        log.error(f'Loader: {e}')
        errors.display(e, 'runai')
    return state_dict


def hijack_safetensors(_diffusers: bool = True, _transformers: bool = True):
    if _diffusers:
        safetensors.torch.load_file = hijacked_load_file
    if _transformers:
        transformers.modeling_utils.load_state_dict = hijacked_load_state_dict


def restore_safetensors():
    safetensors.torch.load_file = orig_load_file
    transformers.modeling_utils.load_state_dict = orig_load_state_dict
