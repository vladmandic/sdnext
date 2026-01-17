import os
import shutil
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from pipelines import generic


def ensure_tokenizer_files(checkpoint_info):
    """Ensure tokenizer files are compatible with Transformers v4."""
    local_path = checkpoint_info.path
    if not os.path.isdir(local_path):
        return
    tokenizer_path = os.path.join(local_path, 'tokenizer')
    if not os.path.isdir(tokenizer_path):
        return

    # Check all required files exist (v5 tokenizers may be missing these or have incompatible configs)
    required = ['vocab.json', 'merges.txt', 'tokenizer_config.json']
    missing = [f for f in required if not os.path.exists(os.path.join(tokenizer_path, f))]
    if not missing:
        return

    # Download v4-compatible tokenizer files from Z-Image-Turbo
    shared.log.debug(f'Load model: fetching v4-compatible tokenizer from Z-Image-Turbo (missing: {missing})')
    try:
        from huggingface_hub import hf_hub_download
        for f in required:
            src = hf_hub_download('Tongyi-MAI/Z-Image-Turbo', f'tokenizer/{f}', cache_dir=shared.opts.hfcache_dir)
            shutil.copy(src, os.path.join(tokenizer_path, f))
    except Exception as e:
        shared.log.warning(f'Load model: failed to fetch tokenizer files: {e}')


def load_flux2_klein(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    # Ensure tokenizer files exist for models created with Transformers v5
    ensure_tokenizer_files(checkpoint_info)

    # Detect SDNQ pre-quantized repo - disable shared text encoder to use pre-quantized weights from SDNQ repo
    is_sdnq = 'sdnq' in repo_id.lower()

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=Flux2Klein repo="{repo_id}" prequant={"sdnq" if is_sdnq else "none"} config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # Load transformer - Klein uses Flux2Transformer2DModel (same class as Flux2, different size)
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Flux2Transformer2DModel, load_config=diffusers_load_config)

    # Load text encoder - Klein uses Qwen3 (4B for Klein-4B, 8B for Klein-9B)
    # For SDNQ repos, disable shared text encoder to use pre-quantized weights from the repo
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3ForCausalLM, load_config=diffusers_load_config, allow_shared=not is_sdnq)

    pipe = diffusers.Flux2KleinPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
