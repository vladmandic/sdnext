import os
import torch
from huggingface_hub import hf_hub_download
from modules import devices, paths
from modules.logger import log
from modules.dlss import DLSSNRPipeline, DLSSVSRPipeline, DLSSFGPipeline


REPO_ID = 'SD-Next/DLSS'
_PIPELINE_CACHE = {}
debug = os.environ.get('SD_DLSS_DEBUG', None) is not None


def _resolve_model_path(filename: str) -> str:
    base_path = os.path.join(paths.models_path, 'DLSS')
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, filename)
    if os.path.exists(file_path):
        return file_path
    try:
        log.info(f'DLSS download: model="{filename}"')
        file_path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=base_path, force_download=False)
        if os.path.exists(file_path):
            return file_path
    except Exception as e:
        log.error(f'DLSS download: model="{filename}" error={e}')
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
