import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_qwen21(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, module='Model')
    log.debug(f'Load model: type=Qwen21 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.qwen import QWEN21_SPEC
    cls = diffusers.QwenImage21Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['qwen-image-21'] = cls
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['qwen-image-21'] = cls
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING['qwen-image-21'] = cls

    if repo_id is None or repo_id.lower() == 'none':
        return None

    # img_in reads 64 latent channels, below the int8 GEMM minimum K; modulation feeds every block, so its error compounds; the rest are the small embedding and output projections
    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.QwenImage21Transformer2DModel,
        load_config=diffusers_load_config,
        native_spec=QWEN21_SPEC,
        modules_to_not_convert=['img_in', 'txt_in', 'time_text_embed', 'modulation', 'norm_out', 'proj_out'],
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Qwen3VLForConditionalGeneration,
        load_config=diffusers_load_config,
    )

    pipe = cls.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        **load_args,
    )

    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
