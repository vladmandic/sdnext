import os
import sys
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_bria(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    repo_lc = repo_id.lower()

    # FIBO family is upstream diffusers-based, so use native classes directly.
    if 'fibo' in repo_lc:
        load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
        log.debug(f'Load model: type=BriaFibo repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

        transformer = generic.load_transformer(
            repo_id,
            cls_name=diffusers.BriaFiboTransformer2DModel,
            load_config=diffusers_load_config,
            allow_quant=False,
        )
        text_encoder = generic.load_text_encoder(
            repo_id,
            cls_name=transformers.SmolLM3ForCausalLM,
            load_config=diffusers_load_config,
            allow_quant=False,
            allow_shared=False,
        )

        if 'fibo-edit' in repo_lc:
            cls = diffusers.BriaFiboEditPipeline
            diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['bria-fibo'] = cls
            diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['bria-fibo'] = cls
            diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING['bria-fibo'] = cls
        else:
            cls = diffusers.BriaFiboPipeline
            diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['bria-fibo'] = cls

        pipe = cls.from_pretrained(
            repo_id,
            transformer=transformer,
            text_encoder=text_encoder,
            cache_dir=shared.opts.diffusers_dir,
            **load_args,
        )
        from pipelines.bria import prompt_to_json
        pipe.before_prompt_encode = prompt_to_json.before_prompt_encode
        pipe.task_args = {
            'output_type': 'np',
        }

    else:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'bria'))
        from pipelines.bria.bria_pipeline import BriaPipeline
        from pipelines.bria.transformer_bria import BriaTransformer2DModel
        diffusers.BriaPipeline = BriaPipeline
        diffusers.BriaTransformer2DModel = BriaTransformer2DModel

        load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
        log.debug(f'Load model: type=Bria repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

        transformer = generic.load_transformer(repo_id, cls_name=BriaTransformer2DModel, load_config=diffusers_load_config)
        text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config)

        pipe = BriaPipeline.from_pretrained(
            repo_id,
            transformer=transformer,
            text_encoder=text_encoder,
            cache_dir=shared.opts.diffusers_dir,
            trust_remote_code=True,
            **load_args,
        )

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
