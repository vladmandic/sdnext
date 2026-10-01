import os
import torch
from huggingface_hub import hf_hub_download
from modules import devices, paths
from modules.logger import log
from modules.dlss import DLSSNRPipeline, DLSSVSRPipeline, DLSSFGPipeline


REPO_ID = 'vladmandic/sdnext-upscalers'
MODEL_MAP = { # model mapped to its sha256 hash
    'DLSSNeuralRender.safetensors': '5fd8c5dd7282e6e35474b5f24ef3b4de5fa866402d9c06654fb511a90e4cf3f7',
    'DLSSSuperRes.safetensors': 'bcf244324e107ec202c5c6a2428509a39f231930201ca1a438ca61ab440cb337',
    'DLSSFrameGen.safetensors': '35b95bba4d9122ba557429fb19c1d2604585ff329996643d7e66a6ed0a4f920c',
}
_PIPELINE_CACHE = {}
debug = os.environ.get('SD_DLSS_DEBUG', None) is not None


def _resolve_model_path(filename: str) -> str:
    sha256 = MODEL_MAP.get(filename, None)[0:8]
    base_path = os.path.join(paths.models_path, 'DLSS')
    os.makedirs(base_path, exist_ok=True)
    file_name = f"{sha256}.safetensors"
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return file_path
    try:
        log.info(f'DLSS download: model="{filename}" sha256={sha256}')
        file_path = hf_hub_download(repo_id=REPO_ID, filename=file_name, local_dir=base_path, force_download=False)
        if os.path.exists(file_path):
            return file_path
    except Exception as e:
        log.error(f'DLSS download: model="{filename}" sha256={sha256} error={e}')
        return None
    file_path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=base_path, force_download=False)
    log.error(f'DLSS: model="{filename}" not found')
    return None


def _get_nr_pipeline(model_path: str, graphs: bool, chunk: int = 131072, dtype: torch.dtype = torch.float16) -> DLSSNRPipeline:
    key = ('nr', model_path, graphs)
    if key not in _PIPELINE_CACHE:
        log.debug(f'DLSS load: cls=DLSSNRPipeline graphs={graphs} device={devices.device} dtype={dtype} chunk={chunk}')
        if debug:
            log.debug(f'DLSS load: file="{model_path}"')
        _PIPELINE_CACHE[key] = DLSSNRPipeline.from_safetensors(model_path, device=devices.cpu, dtype=dtype, graphs=graphs, chunk=chunk)
    pipe = _PIPELINE_CACHE[key]
    pipe.device = devices.device
    pipe.dtype = dtype
    pipe.model = pipe.model.to(devices.device, dtype=dtype)
    return pipe


def _get_fg_pipeline(model_path: str, graphs: bool, dtype: torch.dtype) -> DLSSFGPipeline:
    key = ('fg', model_path, graphs)
    if key not in _PIPELINE_CACHE:
        log.debug(f'DLSS load: cls=DLSSFGPipeline graphs={graphs} device={devices.device} dtype={dtype}')
        if debug:
            log.debug(f'DLSS load: file="{model_path}"')
        _PIPELINE_CACHE[key] = DLSSFGPipeline.from_safetensors(model_path, device=devices.cpu, dtype=dtype, graphs=graphs)
    pipe = _PIPELINE_CACHE[key]
    pipe.device = devices.device
    pipe.dtype = dtype
    pipe.model = pipe.model.to(devices.device, dtype=dtype)
    return pipe


def _get_vsr_pipeline(model_path: str, graphs: bool, dtype: torch.dtype) -> DLSSVSRPipeline:
    key = ('vsr', model_path, graphs)
    if key not in _PIPELINE_CACHE:
        log.debug(f'DLSS load: cls=DLSSVSRPipeline graphs={graphs} device={devices.device} dtype={dtype}')
        if debug:
            log.debug(f'DLSS load: file="{model_path}"')
        _PIPELINE_CACHE[key] = DLSSVSRPipeline.from_safetensors(model_path, device=devices.cpu, dtype=dtype, graphs=graphs)
    pipe = _PIPELINE_CACHE[key]
    pipe.device = devices.device
    pipe.dtype = dtype
    pipe.model = pipe.model.to(devices.device, dtype=dtype)
    return pipe
