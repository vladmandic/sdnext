import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae


def load_qwen(checkpoint_info, diffusers_load_config=None):
    from pipelines import generic, qwen
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    repo_subfolder = checkpoint_info.subfolder
    sd_models.hf_auth_check(checkpoint_info)
    transformer = None

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=Qwen model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    if '2509' in repo_id or '2511' in repo_id:
        cls_name = diffusers.QwenImageEditPlusPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPlusPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPlusPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPlusPipeline
    elif 'Edit' in repo_id:
        cls_name = diffusers.QwenImageEditPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageEditPipeline
    elif 'Layered' in repo_id:
        cls_name = diffusers.QwenImageLayeredPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["qwen-layered"] = diffusers.QwenImageLayeredPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["qwen-layered"] = diffusers.QwenImageLayeredPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["qwen-layered"] = diffusers.QwenImageLayeredPipeline
    else: # qwen-image, qwen-image-2512
        cls_name = diffusers.QwenImagePipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImagePipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageImg2ImgPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["qwen-image"] = diffusers.QwenImageInpaintPipeline

    if model_quant.check_nunchaku('Model'):
        transformer = qwen.load_qwen_nunchaku(repo_id, subfolder=repo_subfolder)

    if 'Qwen-Image-Distill-Full' in repo_id:
        repo_transformer = repo_id
        transformer_subfolder = None
        repo_id = 'Qwen/Qwen-Image'
    else:
        repo_transformer = repo_id
        if repo_subfolder is not None:
            transformer_subfolder = repo_subfolder + '/transformer'
        else:
            transformer_subfolder = "transformer"

    if transformer is None:
        transformer = generic.load_transformer(
            repo_transformer,
            subfolder=transformer_subfolder,
            cls_name=diffusers.QwenImageTransformer2DModel,
            load_config=diffusers_load_config,
            modules_to_not_convert=["transformer_blocks.0.img_mod.1.weight"],
        )

    repo_te = 'Qwen/Qwen-Image'
    text_encoder = generic.load_text_encoder(repo_te, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config)

    repo_id, repo_subfolder = qwen.check_qwen_pruning(repo_id, repo_subfolder)
    if repo_subfolder is not None and repo_subfolder.startswith('nunchaku'):
        repo_subfolder = None
    pipe = cls_name.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        subfolder=repo_subfolder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }
    if 'Layered' in repo_id:
        pipe.task_args['use_en_prompt'] = True
        pipe.task_args['cfg_normalize'] = False
        pipe.task_args['layers'] = shared.opts.model_qwen_layers
        pipe.task_args['resolution'] = 640

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
