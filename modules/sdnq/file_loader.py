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


def load_safetensors(files: list[str], state_dict: dict = None, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    from safetensors.torch import safe_open
    if state_dict is None:
        state_dict = {}
    for fn in files:
        with safe_open(fn, framework="pt", device=str(device)) as f:
            for key in f.keys():
                state_dict[map_keys(key, key_mapping)] = f.get_tensor(key)


def load_threaded(files: list[str], state_dict: dict = None, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    future_items = {}
    if state_dict is None:
        state_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for fn in files:
            future_items[executor.submit(load_safetensors, [fn], key_mapping=key_mapping, device=device, state_dict=state_dict)] = fn
        for future in concurrent.futures.as_completed(future_items):
            future.result()


def load_streamer(files: list[str], state_dict: dict = None, key_mapping: dict = None, device: torch.device = "cpu") -> dict:
    # requires pip install runai_model_streamer
    from runai_model_streamer import SafetensorsStreamer
    if state_dict is None:
        state_dict = {}
    with SafetensorsStreamer() as streamer:
        streamer.stream_files(files)
        for key, tensor in streamer.get_tensors():
            state_dict[map_keys(key, key_mapping)] = tensor.to(device)


def load_files(files: list[str], state_dict: dict = None, key_mapping: dict = None, device: torch.device = "cpu", method: str = None) -> dict:
    # note: files is list-of-files within a module for chunked loading, not accross model
    if isinstance(files, str):
        files = [files]
    if method is None:
        method = 'safetensors'
    if state_dict is None:
        state_dict = {}
    if method == 'safetensors':
        load_safetensors(files, state_dict=state_dict, key_mapping=key_mapping, device=device)
    elif method == 'threaded':
        load_threaded(files, state_dict=state_dict, key_mapping=key_mapping, device=device)
    elif method == 'streamer':
        load_streamer(files, state_dict=state_dict, key_mapping=key_mapping, device=device)
    else:
        raise ValueError(f"Unsupported loading method: {method}")
    return state_dict
