import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_ultraflux(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=UltraFlux repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.ultraflux.pipeline_flux import UltraFluxPipeline
    from pipelines.ultraflux.transformer_flux import FluxTransformer2DModel
    from pipelines.ultraflux.autoencoder_kl import AutoencoderUltraFluxKL

    transformer = generic.load_transformer(repo_id, cls_name=FluxTransformer2DModel, load_config=diffusers_load_config)
    text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder='text_encoder_2')
    if repo_id is None or repo_id.lower() == 'none':
        return None

    vae = AutoencoderUltraFluxKL.from_pretrained(
        repo_id,
        subfolder='vae',
        cache_dir=shared.opts.diffusers_dir,
        torch_dtype=devices.dtype,
    )
    pipe = UltraFluxPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        vae=vae,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }
    if hasattr(pipe, 'scheduler') and hasattr(pipe.scheduler, 'config'):
        if hasattr(pipe.scheduler.config, 'use_dynamic_shifting'):
            pipe.scheduler.config.use_dynamic_shifting = False
        if hasattr(pipe.scheduler.config, 'time_shift'):
            pipe.scheduler.config.time_shift = 4

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['ultraflux'] = UltraFluxPipeline

    del text_encoder_2
    del transformer
    del vae
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
