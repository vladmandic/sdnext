# import transformers
from modules import shared, devices, sd_models, model_quant # pylint: disable=unused-import
from modules.logger import log
from pipelines import generic # pylint: disable=unused-import


def load_nextstep(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    log.error(f'Load model: type=NextStep model="{checkpoint_info.name}" repo="{repo_id}" not supported')

    """
    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    log.debug(f'Load model: type=NextStep model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.nextstep import NextStepPipeline, NextStep

    def __call__(self,
                 prompt = None,
                 image = None,
                 height = 1024,
                 width = 1024,
                 num_inference_steps: int = 20,
                 guidance_scale: float = 1.0,
                 generator = None,
                ):
        return self.generate_image(self,
                                   captions = prompt,
                                   images = [image] if image is not None else None,
                                   num_images_per_caption = 1,
                                   positive_prompt = None,
                                   negative_prompt = None,
                                   hw = (height, width),
                                   use_norm = False,
                                   cfg = guidance_scale,
                                   cfg_img = 1.0,
                                   cfg_schedule = "constant", # "linear", "constant"
                                   num_sampling_steps = num_inference_steps,
                                   timesteps_shift = 1.0,
                                   seed = generator.initial_seed(),
                                   progress = True,
                                  )

    NextStepPipeline.__call__ = __call__

    # tokenizer = transformers.AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
    model = generic.load_transformer(repo_id, cls_name=NextStep, load_config=diffusers_load_config)
    pipe = NextStepPipeline(
        repo_id,
        model=model,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    from modules.video_models import video_vae
    pipe.vae.orig_decode = pipe.vae.decode
    pipe.vae.decode = video_vae.hijack_vae_decode

    devices.torch_gc()
    return pipe
    """

    return None
