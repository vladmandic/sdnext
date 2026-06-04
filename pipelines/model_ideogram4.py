import diffusers
from transformers import AutoTokenizer
from transformers.models.qwen3_vl import Qwen3VLModel
from modules import shared, devices, sd_models
from modules.logger import log
from pipelines import generic
from pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from pipelines.ideogram4.scheduler_ideogram4 import Ideogram4Scheduler
from pipelines.ideogram4.transformer_ideogram4 import Ideogram4Transformer2DModel


def load_ideogram4(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    log.debug(f'Load model: type=Ideogram4 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    if repo_id is None or repo_id.lower() == 'none':
        return None

    # Each transformer loads independently from its subfolder.
    transformer = generic.load_transformer(repo_id, cls_name=Ideogram4Transformer2DModel, subfolder="transformer", load_config=diffusers_load_config)
    unconditional_transformer = generic.load_transformer(repo_id, cls_name=Ideogram4Transformer2DModel, subfolder="unconditional_transformer", load_config=diffusers_load_config)
    # shared_te_map redirects to the shared Qwen3-VL repo (deduped with VQA + prompt-enhance);
    # the bundled text_encoder is the fallback when sharing is off.
    text_encoder = generic.load_text_encoder(repo_id, cls_name=Qwen3VLModel, load_config=diffusers_load_config)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer", cache_dir=shared.opts.diffusers_dir)
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder="vae", cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype)
    scheduler = Ideogram4Scheduler()

    pipe = Ideogram4Pipeline(
        transformer=transformer,
        unconditional_transformer=unconditional_transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
    )
    # The pipeline decodes the packed latent itself, so keep SD.Next from re-decoding it.
    pipe.task_args = {'output_type': 'pil'}

    del transformer, unconditional_transformer, text_encoder, vae
    devices.torch_gc(force=True, reason='load')
    return pipe
