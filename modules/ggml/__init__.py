from modules.logger import log
import os
import time
import torch
import diffusers
import transformers


def install_gguf():
    # pip install git+https://github.com/junejae/transformers@feature/t5-gguf
    # https://github.com/ggerganov/llama.cpp/issues/9566
    from installer import install
    install('gguf', quiet=True)
    import importlib.metadata
    import gguf
    from modules import shared
    scripts_dir = os.path.join(os.path.dirname(gguf.__file__), '..', 'scripts')
    if os.path.exists(scripts_dir):
        os.rename(scripts_dir, scripts_dir + str(time.time()))
    # monkey patch transformers/diffusers so they detect newly installed gguf pacakge correctly
    ver = importlib.metadata.version('gguf')
    transformers.utils.import_utils._is_gguf_available = True # pylint: disable=protected-access
    transformers.utils.import_utils._gguf_version = ver # pylint: disable=protected-access
    diffusers.utils.import_utils._is_gguf_available = True # pylint: disable=protected-access
    diffusers.utils.import_utils._gguf_version = ver # pylint: disable=protected-access
    log.debug(f'Load GGUF: version={ver}')
    return gguf


def load_gguf_state_dict(path: str, compute_dtype: torch.dtype) -> dict:
    gguf = install_gguf()
    from .gguf_utils import TORCH_COMPATIBLE_QTYPES
    from .gguf_tensor import GGMLTensor
    sd: dict[str, GGMLTensor] = {}
    stats = {}
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        torch_tensor = torch.from_numpy(tensor.data)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            torch_tensor = torch_tensor.view(*shape)
        sd[tensor.name] = GGMLTensor(torch_tensor, ggml_quantization_type=tensor.tensor_type, tensor_shape=shape, compute_dtype=compute_dtype)
        if tensor.tensor_type.name not in stats:
            stats[tensor.tensor_type.name] = 0
        stats[tensor.tensor_type.name] += 1
    return sd, stats


def load_gguf(path, cls, compute_dtype: torch.dtype):
    _gguf = install_gguf()
    loader = cls.from_single_file if hasattr(cls, 'from_single_file') else cls.from_pretrained
    module = loader(
        path,
        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype,
    )
    module.gguf = 'gguf'
    return module
