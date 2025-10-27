import re
import concurrent.futures
import torch


def map_keys(key: str, key_mapping: dict) -> str:
    new_key = key
    if key_mapping:
        for pattern, replacement in key_mapping.items():
            new_key, n_replace = re.subn(pattern, replacement, new_key)
            if n_replace > 0:
                break
    return new_key


def load_safetensors(files: list[str], state_dict: dict, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    from safetensors.torch import safe_open
    for fn in files:
        with safe_open(fn, framework="pt", device=str(device)) as f:
            for key in f.keys():
                state_dict[map_keys(key, key_mapping)] = f.get_tensor(key)


def load_threaded(files: list[str], key_mapping: dict = None, device: torch.device = "cpu", state_dict: dict = {}) -> dict:
    future_items = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for fn in files:
            future_items[executor.submit(load_safetensors, [fn], key_mapping=key_mapping, device=device, state_dict=state_dict)] = fn
        for future in concurrent.futures.as_completed(future_items):
            future.result()


def load_streamer(files: list[str], state_dict: dict, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    # requires pip install runai_model_streamer
    from runai_model_streamer import SafetensorsStreamer
    with SafetensorsStreamer() as streamer:
        streamer.stream_files(files)
        for key, tensor in streamer.get_tensors():
            state_dict[map_keys(key, key_mapping)] = tensor.to(device)


def load_files(files: list[str], method: str = None, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    # note: files is list-of-files within a module for chunked loading, not accross model
    method = method or 'safetensors'
    state_dict = {}
    if method == 'safetensors':
        load_safetensors(files, key_mapping=key_mapping, device=device, state_dict=state_dict)
    elif method == 'threaded':
        load_threaded(files, key_mapping=key_mapping, device=device, state_dict=state_dict)
    elif method == 'streamer':
        load_streamer(files, key_mapping=key_mapping, device=device, state_dict=state_dict)
    else:
        raise ValueError(f"Unsupported loading method: {method}")
    return state_dict
