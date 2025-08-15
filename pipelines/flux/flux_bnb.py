import diffusers
import transformers
from modules import devices, model_quant


def load_flux_bnb(checkpoint_info, diffusers_load_config): # pylint: disable=unused-argument
    transformer = None
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    model_quant.load_bnb('Load model: type=FLUX')
    quant = model_quant.get_quant(repo_path)
    if quant == 'fp8':
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=devices.dtype)
        transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    elif quant == 'fp4':
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'fp4')
        transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    elif quant == 'nf4':
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'nf4')
        transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    else:
        transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config)
    return transformer
