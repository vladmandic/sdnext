import time
import diffusers
import transformers
from modules import shared, devices, errors, timer, sd_models, model_quant, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def hijack_encode_text(prompt: str | list[str]):
    jobid = shared.state.begin('TE Encode')
    t0 = time.time()
    try:
        prompt = shared.sd_model.refine_prompts(prompt)
    except Exception as e:
        log.error(f'Encode prompt: {e}')
        errors.display(e, 'Encode prompt')
    try:
        res = shared.sd_model.orig_encode_text(prompt)
    except Exception as e:
        log.error(f'Encode prompt: {e}')
        errors.display(e, 'Encode prompt')
        res = None
    t1 = time.time()
    timer.process.add('te', t1-t0)
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    shared.state.end(jobid)
    return res


def load_sdxs(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=SDXS repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3_5ForConditionalGeneration, load_config=diffusers_load_config, allow_shared=False)

    pipe = diffusers.DiffusionPipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
    )
    pipe.task_args = {
        'generator': None,
        'output_type': 'np',
    }

    pipe.orig_encode_text = pipe.encode_text
    pipe.encode_text = hijack_encode_text
    sd_hijack_vae.init_hijack(pipe)
    del text_encoder

    devices.torch_gc(force=True, reason='load')
    return pipe
