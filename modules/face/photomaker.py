import cv2
import numpy as np
import torch
import huggingface_hub as hf
from modules import shared, processing, sd_models, devices
from modules.logger import log


original_pipeline = None


def restore_pipeline():
    global original_pipeline # pylint: disable=global-statement
    if original_pipeline is not None:
        shared.sd_model = original_pipeline
        original_pipeline = None


def photo_maker(p: processing.StableDiffusionProcessing, app, model: str, input_images, trigger, strength, start): # pylint: disable=arguments-differ
    global original_pipeline # pylint: disable=global-statement
    from modules.face.photomaker_pipeline import PhotoMakerStableDiffusionXLPipeline

    # prepare pipeline
    if len(input_images) == 0:
        log.warning('PhotoMaker: no input images')
        return None

    if len(trigger) == 0:
        log.warning('PhotoMaker: no trigger word')
        return None

    c = shared.sd_model.__class__.__name__ if shared.sd_loaded else ''
    if c != 'StableDiffusionXLPipeline':
        log.warning(f'PhotoMaker invalid base model: current={c} required=StableDiffusionXLPipeline')
        return None

    # validate prompt
    if not p.all_prompts:
        processing.process_init(p)
        p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
    trigger_ids = shared.sd_model.tokenizer.encode(trigger) + shared.sd_model.tokenizer_2.encode(trigger)
    prompt_ids1 = shared.sd_model.tokenizer.encode(p.all_prompts[0])
    prompt_ids2 = shared.sd_model.tokenizer_2.encode(p.all_prompts[0])
    for t in trigger_ids:
        if prompt_ids1.count(t) != 1:
            log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.all_prompts[0]} ids={prompt_ids1}')
            return None
        if prompt_ids2.count(t) != 1:
            log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.all_prompts[0]} ids={prompt_ids1}')
            return None

    # create new pipeline
    original_pipeline = shared.sd_model # backup current pipeline definition
    # orig_pipeline = shared.sd_model # backup current pipeline definition
    shared.sd_model = sd_models.switch_pipe(PhotoMakerStableDiffusionXLPipeline, shared.sd_model)
    shared.sd_model.restore_pipeline = restore_pipeline
    # sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
    sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
    sd_models.apply_balanced_offload(shared.sd_model) # apply balanced offload

    orig_prompt_attention = shared.opts.prompt_attention
    shared.opts.data['prompt_attention'] = 'fixed' # otherwise need to deal with class_tokens_mask
    p.task_args['input_id_images'] = input_images
    p.task_args['start_merge_step'] = int(start * p.steps)
    p.task_args['prompt'] = p.all_prompts[0] if p.all_prompts else p.prompt

    is_v2 = 'v2' in model
    if is_v2:
        repo_id, fn = 'TencentARC/PhotoMaker-V2', 'photomaker-v2.bin'
    else:
        repo_id, fn = 'TencentARC/PhotoMaker', 'photomaker-v1.bin'

    photomaker_path = hf.hf_hub_download(repo_id=repo_id, filename=fn, repo_type="model", cache_dir=shared.opts.hfcache_dir)
    log.debug(f'PhotoMaker: model="{model}" uri="{repo_id}/{fn}" images={len(input_images)} trigger={trigger} args={p.task_args}')

    # load photomaker adapter
    shared.sd_model.load_photomaker_adapter(
        photomaker_path,
        trigger_word=trigger,
        weight_name='photomaker-v2.bin' if is_v2 else 'photomaker-v1.bin',
        pm_version='v2' if is_v2 else 'v1',
        device=devices.device,
        cache_dir=shared.opts.hfcache_dir,
    )
    shared.sd_model.set_adapters(["photomaker"], adapter_weights=[strength])

    # analyze faces
    if is_v2:
        id_embed_list = []
        for i, source_image in enumerate(input_images):
            faces = app.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
            face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
            id_embed_list.append(torch.from_numpy(face['embedding']))
            log.debug(f'PhotoMaker: face={i+1} score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')
        p.task_args['id_embeds'] = torch.stack(id_embed_list).to(device=devices.device, dtype=devices.dtype)

    # run processing
    # processed: processing.Processed = processing.process_images(p)
    p.extra_generation_params['PhotoMaker'] = f'{strength}'

    # unload photomaker adapter
    shared.sd_model.unload_lora_weights()

    # restore original pipeline
    shared.opts.data['prompt_attention'] = orig_prompt_attention
    # shared.sd_model = orig_pipeline
    return None
    # return processed
