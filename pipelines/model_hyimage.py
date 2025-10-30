from types import SimpleNamespace
import torch
import transformers
import diffusers
from modules import shared, sd_models, devices, model_quant, sd_hijack_te, sd_hijack_vae
from pipelines import generic


def load_hyimage(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=HunyuanImage21 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.HunyuanImageTransformer2DModel, load_config=diffusers_load_config, subfolder="transformer")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config, subfolder="text_encoder")
    text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_2", allow_shared=False)

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


def load_hyimage3(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    shared.log.debug(f'Load model: type=HunyuanImage3 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    allow_quant = True
    if 'sdnq-' in repo_id.lower():
        from modules import sdnq # pylint: disable=unused-import # register to diffusers and transformers
        sd_models.allow_post_quant = False # we already handled it
        allow_quant = False

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True, allow_quant=allow_quant)
    pipe = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        attn_implementation="sdpa",
        moe_impl="eager",
        **load_args,
        **quant_args,
    )
    pipe.load_tokenizer(repo_id)

    pipe.pipeline # call it to set up pipeline
    pipe = HunyuanImage3Wrapper(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe


class HunyuanImage3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(
        self,
        prompt: str,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ["latents"],
        **kwargs,
    ):
        if hasattr(self.model._pipeline.model, "_hf_hook"):
            self.model._pipeline.model._hf_hook.execution_device = torch.device(devices.device)

        if num_inference_steps > 1:
            if isinstance(prompt, str):
                prompt = [prompt]
            prompt = prompt * num_images_per_prompt

        if height is None and width is None:
            image_size = "auto"
        if height is None:
            image_size = (width, width)
        if width is None:
            image_size = (height, height)
        else:
            image_size = (height, width)

        output = self.model.generate_image(
            prompt,
            image_size=(height, width),
            diff_infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )

        if not isinstance(output, list):
            output = [output]
        return SimpleNamespace(images=output)
