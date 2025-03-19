from dataclasses import dataclass
import diffusers
import transformers


@dataclass
class Model():
    name: str
    repo: str = None
    repo_cls: classmethod = None
    dit: str = None
    dit_cls: classmethod = None
    dit_folder: str = 'transformer'
    te: str = None
    te_cls: classmethod = None
    te_folder: str = 'text_encoder'
    te_hijack: bool = True
    vae_hijack: bool = True


models = {
    'None': [],
    'Hunyuan Video': [
        Model(name='None'),
        Model(name='Hunyuan Video T2V',
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='Hunyuan Video I2V', # https://github.com/huggingface/diffusers/pull/10983
              repo='hunyuanvideo-community/HunyuanVideo-I2V',
              repo_cls=diffusers.HunyuanVideoImageToVideoPipeline,
              te_cls=transformers.LlavaForConditionalGeneration,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='SkyReels Hunyuan T2V', # https://github.com/huggingface/diffusers/pull/10837
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit='Skywork/SkyReels-V1-Hunyuan-T2V',
              dit_folder=None,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='SkyReels Hunyuan I2V', # https://github.com/huggingface/diffusers/pull/10837
              repo='hunyuanvideo-community/HunyuanVideo',
              te_cls=transformers.LlamaModel,
              dit='Skywork/SkyReels-V1-Hunyuan-I2V',
              dit_folder=None,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='Fast Hunyuan T2V', # https://github.com/hao-ai-lab/FastVideo/blob/8a77cf22c9b9e7f931f42bc4b35d21fd91d24e45/fastvideo/models/hunyuan/inference.py#L213
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit='FastVideo/FastHunyuan-diffusers',
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
    ],
}

"""
'LTX Video': [
    Model(name='None'),
    Model(name='LTXVideo 0.9.0 T2V', repo='a-r-r-o-w/LTX-Video-diffusers', subfolder='transformer'),
    Model(name='LTXVideo 0.9.1 T2V', repo='a-r-r-o-w/LTX-Video-0.9.1-diffusers', subfolder='transformer'),
    Model(name='LTXVideo 0.9.5 T2V', repo='Lightricks/LTX-Video-0.9.5'), # https://github.com/huggingface/diffusers/pull/10968
    Model(name='LTXVideo 0.9.0 I2V', repo='a-r-r-o-w/LTX-Video-diffusers', subfolder='transformer'),
    Model(name='LTXVideo 0.9.1 I2V', repo='a-r-r-o-w/LTX-Video-0.9.1-diffusers', subfolder='transformer'),
    Model(name='LTXVideo 0.9.5 I2V', repo='Lightricks/LTX-Video-0.9.5', subfolder='transformer'),
],"
"""
