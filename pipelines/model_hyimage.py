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

    from sdnq import SDNQConfig # import sdnq to register it into transformers
    pipe = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=devices.dtype,
        device_map="auto",
        moe_impl="eager",
    )
    pipe.load_tokenizer(repo_id)
    pipe.__call__ = pipe.generate_image
    pipe.task_args = {
        'stream': True,
        'diff_infer_steps': 20,
    }

    devices.torch_gc(force=True, reason='load')
    return pipe
