import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_vibe(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=VIBE repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    pipe_cls = getattr(diffusers, 'VIBESanaEditingPipeline', None)
    transformer_cls = getattr(diffusers, 'VIBESanaEditingModel', None)
    text_encoder_cls = getattr(transformers, 'Qwen3VLForConditionalGeneration', None)
    processor_cls = getattr(transformers, 'Qwen3VLProcessor', None)

    if pipe_cls is not None and transformer_cls is not None and text_encoder_cls is not None and processor_cls is not None:
        transformer = generic.load_transformer(
            repo_id,
            cls_name=transformer_cls,
            load_config=diffusers_load_config,
            allow_quant=False,
        )
        text_encoder = generic.load_text_encoder(
            repo_id,
            cls_name=text_encoder_cls,
            load_config=diffusers_load_config,
            allow_quant=False,
            allow_shared=False,
        )
        processor = processor_cls.from_pretrained(repo_id, subfolder='tokenizer', cache_dir=shared.opts.hfcache_dir)

        pipe = pipe_cls.from_pretrained(
            repo_id,
            cache_dir=shared.opts.diffusers_dir,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=processor,
            **load_args,
        )

        del transformer
        del text_encoder
        del processor
    else:
        try:
            from installer import install, installed
            if not installed('vibe', quiet=True):
                install('git+https://github.com/ai-forever/VIBE', 'vibe')
            import vibe # pylint: disable=unused-import
        except Exception as e:
            raise RuntimeError('VIBE requires either native diffusers VIBESana classes or `vibe` package') from e

        pipe = diffusers.DiffusionPipeline.from_pretrained(
            repo_id,
            cache_dir=shared.opts.diffusers_dir,
            trust_remote_code=True,
            **load_args,
        )

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['vibe-sana'] = pipe.__class__
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['vibe-sana'] = pipe.__class__
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING['vibe-sana'] = pipe.__class__

    pipe.task_args = {
        'output_type': 'np',
    }

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe

""" Reference
  "VIBE Image Edit": {
    "path": "iitolstykh/VIBE-Image-Edit",
    "preview": "iitolstykh--VIBE-Image-Edit.jpg",
    "desc": "VIBE is an open-source text-guided image editing model combining Sana1.5-1.6B diffusion backbone with Qwen3-VL multimodal conditioning for fast, instruction-based edits.",
    "skip": true,
    "extras": "sampler: Default, cfg_scale: 4.5, image_guidance_scale: 1.2, steps: 20",
    "size": 9.72,
    "date": "2025 December"
  },
"""
