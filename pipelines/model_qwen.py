import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from pipelines import generic


def load_qwen(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=Qwen model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.QwenImageTransformer2DModel, load_config=diffusers_load_config, modules_dtype_dict={"minimum_6bit": ["img_mod", "pos_embed", "time_text_embed", "img_in", "txt_in", "norm_out"]})
    repo_te = 'Qwen/Qwen-Image' if 'Qwen-Lightning' in repo_id else repo_id
    text_encoder = generic.load_text_encoder(repo_te, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config)

    pipe = diffusers.QwenImagePipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageInpaintPipeline

    del text_encoder
    del transformer

    sd_hijack_te.init_hijack(pipe)
    from modules.video_models import video_vae
    pipe.vae.orig_decode = pipe.vae.decode
    pipe.vae.decode = video_vae.hijack_vae_decode

    devices.torch_gc()
    return pipe
