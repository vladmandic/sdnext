from dataclasses import dataclass
import diffusers
import transformers


@dataclass
class Model():
    name: str
    url: str = ''
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
              url='https://huggingface.co/tencent/HunyuanVideo',
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='Hunyuan Video I2V', # https://github.com/huggingface/diffusers/pull/10983
              url='https://huggingface.co/tencent/HunyuanVideo-I2V',
              repo='hunyuanvideo-community/HunyuanVideo-I2V',
              repo_cls=diffusers.HunyuanVideoImageToVideoPipeline,
              te_cls=transformers.LlavaForConditionalGeneration,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='SkyReels Hunyuan T2V', # https://github.com/huggingface/diffusers/pull/10837
              url='https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V',
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit='Skywork/SkyReels-V1-Hunyuan-T2V',
              dit_folder=None,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='SkyReels Hunyuan I2V', # https://github.com/huggingface/diffusers/pull/10837
              url='https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V',
              repo='hunyuanvideo-community/HunyuanVideo',
              te_cls=transformers.LlamaModel,
              dit='Skywork/SkyReels-V1-Hunyuan-I2V',
              dit_folder=None,
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
        Model(name='Fast Hunyuan T2V', # https://github.com/hao-ai-lab/FastVideo/blob/8a77cf22c9b9e7f931f42bc4b35d21fd91d24e45/fastvideo/models/hunyuan/inference.py#L213
              url='https://huggingface.co/FastVideo/FastHunyuan',
              repo='hunyuanvideo-community/HunyuanVideo',
              repo_cls=diffusers.HunyuanVideoPipeline,
              te_cls=transformers.LlamaModel,
              dit='FastVideo/FastHunyuan-diffusers',
              dit_cls=diffusers.HunyuanVideoTransformer3DModel),
    ],
    'LTX Video': [
        Model(name='None'),
        Model(name='LTXVideo 0.9.5 T2V', # https://github.com/huggingface/diffusers/pull/10968
              url='https://huggingface.co/Lightricks/LTX-Video-0.9.5',
              repo='YiYiXu/ltx-95',
              repo_cls=diffusers.LTXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
        Model(name='LTXVideo 0.9.5 I2V',
              url='https://huggingface.co/Lightricks/LTX-Video-0.9.5',
              repo='YiYiXu/ltx-95',
              repo_cls=diffusers.LTXConditionPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
        Model(name='LTXVideo 0.9.1 T2V',
              url='https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.1-diffusers',
              repo='a-r-r-o-w/LTX-Video-0.9.1-diffusers',
              repo_cls=diffusers.LTXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
        Model(name='LTXVideo 0.9.1 I2V',
              url='https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.1-diffusers',
              repo='a-r-r-o-w/LTX-Video-0.9.1-diffusers',
              repo_cls=diffusers.LTXImageToVideoPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
        Model(name='LTXVideo 0.9.0 T2V',
              url='https://huggingface.co/a-r-r-o-w/LTX-Video-diffusers',
              repo='a-r-r-o-w/LTX-Video-diffusers',
              repo_cls=diffusers.LTXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
        Model(name='LTXVideo 0.9.0 I2V',
              url='https://huggingface.co/a-r-r-o-w/LTX-Video-diffusers',
              repo='a-r-r-o-w/LTX-Video-diffusers',
              repo_cls=diffusers.LTXImageToVideoPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.LTXVideoTransformer3DModel),
    ],
    'Mochi Video': [
        Model(name='None'),
        Model(name='Mochi 1 T2V',
              url='https://huggingface.co/genmo/mochi-1-preview',
              repo='genmo/mochi-1-preview',
              repo_cls=diffusers.MochiPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.MochiTransformer3DModel),
    ],
    'Allegro Video': [
        Model(name='None'),
        Model(name='Allegro T2V',
              url='https://huggingface.co/rhymes-ai/Allegro',
              repo='rhymes-ai/Allegro',
              repo_cls=diffusers.AllegroPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.AllegroTransformer3DModel),
    ],
    'Cog Video': [
        Model(name='None'),
        Model(name='CogVideoX 1.0 2B T2V',
              url='https://huggingface.co/THUDM/CogVideoX-2b',
              repo='THUDM/CogVideoX-2b',
              repo_cls=diffusers.CogVideoXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.CogVideoXTransformer3DModel),
        Model(name='CogVideoX 1.0 5B T2V',
              url='https://huggingface.co/THUDM/CogVideoX-5b',
              repo='THUDM/THUDM/CogVideoX-5b',
              repo_cls=diffusers.CogVideoXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.CogVideoXTransformer3DModel),
        Model(name='CogVideoX 1.0 5B I2V',
              url='https://huggingface.co/THUDM/CogVideoX-5b-I2V',
              repo='THUDM/CogVideoX-5b-I2V',
              repo_cls=diffusers.CogVideoXImageToVideoPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.CogVideoXTransformer3DModel),
        Model(name='CogVideoX 1.5 5B T2V',
              url='https://huggingface.co/THUDM/THUDM/CogVideoX1.5-5B',
              repo='THUDM/CogVideoX1.5-5B',
              repo_cls=diffusers.CogVideoXPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.CogVideoXTransformer3DModel),
        Model(name='CogVideoX 1.5 5B I2V',
              url='https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V',
              repo='THUDM/CogVideoX1.5-5B-I2V',
              repo_cls=diffusers.CogVideoXImageToVideoPipeline,
              te_cls=transformers.T5EncoderModel,
              dit_cls=diffusers.CogVideoXTransformer3DModel),
    ],
}
