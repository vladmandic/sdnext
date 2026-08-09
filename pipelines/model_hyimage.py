from types import SimpleNamespace
import torch
import transformers
import diffusers
from modules import shared, sd_models, devices, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_hyimage(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=HunyuanImage21 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.hyimage import HUNYUANIMAGE_SPEC
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.HunyuanImageTransformer2DModel, load_config=diffusers_load_config, subfolder="transformer", native_spec=HUNYUANIMAGE_SPEC)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config, subfolder="text_encoder")
    text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_2", allow_shared=False)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    pipe = diffusers.HunyuanImagePipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }

    del transformer
    del text_encoder
    del text_encoder_2
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe


def load_hyimage3(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    log.debug(f'Load model: type=HunyuanImage3 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    allow_quant = True
    if 'sdnq-' in repo_id.lower():
        sd_models.allow_post_quant = False # we already handled it
        allow_quant = False

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True, allow_quant=allow_quant)
    generic.set_pipeline('HunyuanImage3', transformers.AutoModelForCausalLM)
    if repo_id is None or repo_id.lower() == 'none':
        return None
    pipe = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        attn_implementation="sdpa",
        moe_impl="eager",
        **load_args,
        **quant_args,
    )
    if not hasattr(pipe.config, 'model_version'):
        # HunyuanImage-3.0-Instruct and -Instruct-Distil read config.model_version in load_tokenizer but ship no such
        # key and no config default, so the documented entry point raises. The tokenizer discards the value.
        pipe.config.model_version = None
    pipe.load_tokenizer(repo_id)

    pipe.pipeline # noqa: B018 # call it to set up pipeline # pylint: disable=pointless-statement
    is_instruct = getattr(pipe.generation_config, 'sequence_template', 'pretrain') == 'instruct'
    log.debug(f'Load model: type=HunyuanImage3 variant={"instruct" if is_instruct else "base"}')
    pipe = HunyuanImage3InstructWrapper(pipe) if is_instruct else HunyuanImage3Wrapper(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe


def resolve_seeds(seed, batch_size):
    if seed is None or seed < 0:
        return None
    if batch_size <= 1:
        return seed
    return [seed + i for i in range(batch_size)] # int seed is replicated per batch entry upstream which makes identical images


class HunyuanImage3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_diffusion_config(self, num_inference_steps, guidance_scale):
        # instruct variant reads diffusion params from generation_config only, so both variants set them there
        gen_config = self.model.generation_config
        if num_inference_steps is not None and num_inference_steps > 0:
            gen_config.diff_infer_steps = num_inference_steps
        if guidance_scale is not None and guidance_scale > 0:
            gen_config.diff_guidance_scale = guidance_scale
        if hasattr(self.model._pipeline.model, "_hf_hook"): # pylint: disable=protected-access
            self.model._pipeline.model._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access

    @staticmethod
    def resolve_image_size(height, width):
        if height is None and width is None:
            return "auto"
        if height is None:
            return (width, width)
        if width is None:
            return (height, height)
        return (height, width)

    def __call__(
        self,
        prompt: str,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        **kwargs,
    ):
        self.set_diffusion_config(num_inference_steps, guidance_scale)

        if num_inference_steps > 1:
            if isinstance(prompt, str):
                prompt = [prompt]
            prompt = prompt * num_images_per_prompt

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        output = self.model.generate_image(
            prompt,
            image_size=self.resolve_image_size(height, width),
            seed=resolve_seeds(seed, batch_size),
            **kwargs,
        )

        if not isinstance(output, list):
            output = [output]
        return SimpleNamespace(images=output)


class HunyuanImage3InstructWrapper(HunyuanImage3Wrapper):
    # class name carries 'Instruct' so sd_models.get_diffusers_task routes the INSTRUCT task branch which supplies init images

    @staticmethod
    def resolve_instruct_image_size(images, height, width):
        # sizing policy for edits: explicit UI dims are snapped to the model resolution buckets;
        # image_size='auto' + infer_align_image_size=True is the upstream-recommended editing mode
        # where the model predicts the ratio and output is aligned back to the input image size
        if height is None and width is None:
            return "auto", images is not None
        return HunyuanImage3Wrapper.resolve_image_size(height, width), False

    def __call__(
        self,
        prompt: str,
        image = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        bot_task: str | None = None,
        use_system_prompt: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.set_diffusion_config(num_inference_steps, guidance_scale)

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if num_images_per_prompt > 1:
            prompts = prompts * num_images_per_prompt

        images = None
        if image is not None:
            images = [i for i in image if i is not None] if isinstance(image, list) else [image]
        image_size, align_size = self.resolve_instruct_image_size(images, height, width)

        call_args = {}
        if bot_task is not None:
            call_args['bot_task'] = bot_task
        if use_system_prompt is not None:
            call_args['use_system_prompt'] = use_system_prompt
        if system_prompt is not None:
            call_args['system_prompt'] = system_prompt

        cot_text, samples = self.model.generate_image(
            prompt=prompts,
            image=[images] * len(prompts) if images else None, # per-sample image lists must match batch size
            seed=resolve_seeds(seed, len(prompts)),
            image_size=image_size,
            infer_align_image_size=align_size,
            **call_args,
        )

        if cot_text:
            text = cot_text[0] if isinstance(cot_text, list) else cot_text
            log.debug(f'HunyuanImage3: cot="{text[:300]}"')
        return SimpleNamespace(images=samples, cot_text=cot_text)
