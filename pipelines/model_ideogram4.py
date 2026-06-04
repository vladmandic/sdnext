import diffusers
from transformers import AutoTokenizer
from transformers.models.qwen3_vl import Qwen3VLModel
from modules import shared, devices, sd_models
from modules.logger import log
from pipelines import generic


class Ideogram4Pipeline(diffusers.Ideogram4Pipeline):
    """SD.Next integration subclass for the diffusers-native Ideogram 4 pipeline.

    ``encode_prompt`` drives the Qwen3-VL tap by calling ``language_model`` submodules
    directly, which bypasses the balanced-offload pre-forward hook. Move the encoder
    on-device for the tap and release it afterward so it does not pin VRAM.
    """

    def encode_prompt(self, *args, **kwargs):
        self.text_encoder.to(self._execution_device)
        try:
            return super().encode_prompt(*args, **kwargs)
        finally:
            if shared.opts.diffusers_offload_mode != 'none':
                self.text_encoder.to(devices.cpu)


def load_ideogram4(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    log.debug(f'Load model: type=Ideogram4 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    if repo_id is None or repo_id.lower() == 'none':
        return None

    # Each transformer loads independently from its subfolder, so each gets its own SDNQ config.
    cls = diffusers.Ideogram4Transformer2DModel
    transformer = generic.load_transformer(repo_id, cls_name=cls, subfolder="transformer", load_config=diffusers_load_config)
    unconditional_transformer = generic.load_transformer(repo_id, cls_name=cls, subfolder="unconditional_transformer", load_config=diffusers_load_config)
    # shared_te_map redirects to the shared Qwen3-VL repo (deduped with VQA + prompt-enhance);
    # the bundled text_encoder is the fallback when sharing is off.
    text_encoder = generic.load_text_encoder(repo_id, cls_name=Qwen3VLModel, load_config=diffusers_load_config)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer", cache_dir=shared.opts.diffusers_dir)
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder="vae", cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler", cache_dir=shared.opts.diffusers_dir)

    pipe = Ideogram4Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        unconditional_transformer=unconditional_transformer,
    )
    # The pipeline decodes internally; the CFG scale slider drives guidance_scale, which is
    # mutually exclusive with the pipeline's default per-step guidance_schedule.
    pipe.task_args = {'output_type': 'pil', 'guidance_schedule': None} # pylint: disable=attribute-defined-outside-init

    del transformer, unconditional_transformer, text_encoder, vae
    devices.torch_gc(force=True, reason='load')
    return pipe
