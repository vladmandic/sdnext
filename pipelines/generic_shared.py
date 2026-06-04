import os
import transformers
from transformers.models.qwen3_vl import Qwen3VLModel


shared_te_map = {
    'T5-XXL SDNQ-UInt4': {
        'cls': transformers.T5EncoderModel,
        'identifier': 'sdnq-uint4',
        'target_repo': 'Disty0/FLUX.1-dev-SDNQ-uint4-svd-r32',
    },
    'T5-XXL Base': { # template
        'cls': transformers.T5EncoderModel, # desired model class, used as primary matching criteria
        'identifier': None, # additional identifier to match in repo_id or None to ignore
        'target_repo': 'Disty0/t5-xxl', # repo to load from instead of original repo_id
        'target_subfolder': None, # subfolder in repo to load from, None to ignore
        'config_class': transformers.T5Config, # config class to use for loading or None to ignore
        'config_path': os.path.join('configs', 'flux', 'text_encoder_2', 'config.json'), # path to config file to use for loading or None to ignore
    },

    'UMT5 SDNQ-UInt4': {
        'cls': transformers.UMT5EncoderModel,
        'identifier': 'sdnq-uint4',
        'target_repo': 'Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'UMT5 Base': {
        'cls': transformers.UMT5EncoderModel,
        'target_repo': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        'target_subfolder': 'text_encoder',
    },

    'Qwen-2.5 SDNQ-4Bit': {
        'cls': transformers.Qwen2_5_VLForConditionalGeneration,
        'identifier': 'sdnq-4bit',
        'target_repo': 'Disty0/Qwen-Image-2512-SDNQ-uint4-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-2.5 SDNQ-UInt4': {
        'cls': transformers.Qwen2_5_VLForConditionalGeneration,
        'identifier': 'sdnq-uint4',
        'target_repo': 'Disty0/Qwen-Image-2512-SDNQ-uint4-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-2.5 Base': {
        'cls': transformers.Qwen2_5_VLForConditionalGeneration,
        'target_repo': 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers',
        'target_subfolder': 'text_encoder',
    },

    'Qwen-3 9B SDNQ-4bit': {
        'cls': transformers.Qwen3ForCausalLM,
        'identifier': '9b-sdnq-4bit',
        'target_repo': 'Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-3 9B SDNQ-UInt4': {
        'cls': transformers.Qwen3ForCausalLM,
        'identifier': '9b-sdnq-uint4',
        'target_repo': 'Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-3 9B Base': {
        'cls': transformers.Qwen3ForCausalLM,
        'identifier': '9b',
        'target_repo': 'black-forest-labs/FLUX.2-klein-9B',
        'target_subfolder': 'text_encoder',
    },

    'Qwen-3 4B SDNQ-4Bit': { # match after 9b
        'cls': transformers.Qwen3ForCausalLM,
        'identifier': 'sdnq-4bit',
        'target_repo': 'Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-3 4B SDNQ-UInt4': {
        'cls': transformers.Qwen3ForCausalLM,
        'identifier': 'sdnq-uint4',
        'target_repo': 'Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-3 4B Base': {
        'cls': transformers.Qwen3ForCausalLM,
        'target_repo': 'Tongyi-MAI/Z-Image-Turbo',
        'target_subfolder': 'text_encoder',
    },

    'Qwen-3 0.5B SDNQ-UInt4': {
        'cls': transformers.Qwen3Model,
        'identifier': 'uint4',
        'target_repo': 'vladmandic/Anima-1.0-Base-sdnq-svd-dynamic-uint4',
        'target_subfolder': 'text_encoder',
    },
    'Qwen-3 0.5B Base': {
        'cls': transformers.Qwen3Model,
        'target_repo': 'vladmandic/Anima-1.0-Base',
        'target_subfolder': 'text_encoder',
    },

    'Qwen3-VL 8B Base': {
        'cls': Qwen3VLModel,
        'target_repo': 'Qwen/Qwen3-VL-8B-Instruct',
    },
}
