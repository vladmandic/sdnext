import os
import cv2
import torch
import numpy as np
import huggingface_hub as hf
from modules import shared, processing, sd_models, devices
from modules.logger import log


REPO_ID = "InstantX/InstantID"
controlnet_model = None
debug = log.trace if os.environ.get('SD_FACE_DEBUG', None) is not None else lambda *args, **kwargs: None


def instant_id(p: processing.StableDiffusionProcessing, app, source_images, strength=1.0, conditioning=0.5, cache=True): # pylint: disable=arguments-differ
    from modules.face.instantid_model import StableDiffusionXLInstantIDPipeline, draw_kps
    from diffusers.models import ControlNetModel
    global controlnet_model # pylint: disable=global-statement

    # prepare pipeline
    if source_images is None or len(source_images) == 0:
        log.warning('InstantID: no input images')
        return None

    c = shared.sd_model.__class__.__name__ if shared.sd_loaded else ''
    if c not in ['StableDiffusionXLPipeline', 'StableDiffusionXLInstantIDPipeline']:
        log.warning(f'InstantID invalid base model: current={c} required=StableDiffusionXLPipeline')
        return None

    # prepare face emb
    face_embeds = []
    face_images = []
    for i, source_image in enumerate(source_images):
        faces = app.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
        face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
        face_embeds.append(torch.from_numpy(face['embedding']))
        face_images.append(draw_kps(source_image, face['kps']))
        p.extra_generation_params[f"InstantID {i+1}"] = f'{faces[0].det_score:.2f} {"female" if faces[0].gender==0 else "male"} {faces[0].age}y'
        log.debug(f'InstantID face: score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')

    log.debug(f'InstantID loading: model={REPO_ID}')
    face_adapter = hf.hf_hub_download(repo_id=REPO_ID, filename="ip-adapter.bin")
    if controlnet_model is None or not cache:
        controlnet_model = ControlNetModel.from_pretrained(REPO_ID, subfolder="ControlNetModel", torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir)
        sd_models.move_model(controlnet_model, devices.device)

    # create new pipeline
    orig_pipeline = shared.sd_model # backup current pipeline definition
    shared.sd_model = StableDiffusionXLInstantIDPipeline(
        vae = shared.sd_model.vae,
        text_encoder=shared.sd_model.text_encoder,
        text_encoder_2=shared.sd_model.text_encoder_2,
        tokenizer=shared.sd_model.tokenizer,
        tokenizer_2=shared.sd_model.tokenizer_2,
        unet=shared.sd_model.unet,
        scheduler=shared.sd_model.scheduler,
        controlnet=controlnet_model,
        force_zeros_for_empty_prompt=shared.opts.diffusers_force_zeros,
    )
    sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
    sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
    shared.sd_model.load_ip_adapter_instantid(face_adapter, scale=strength)
    shared.sd_model.set_ip_adapter_scale(strength)
    sd_models.move_model(shared.sd_model, devices.device) # move pipeline to device

    # pipeline specific args
    if not p.all_prompts:
        processing.process_init(p)
        p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
    orig_prompt_attention = shared.opts.prompt_attention
    shared.opts.data['prompt_attention'] = 'fixed' # otherwise need to deal with class_tokens_mask
    p.task_args['image_embeds'] = face_embeds[0].shape # placeholder
    p.task_args['image'] = face_images[0]
    p.task_args['controlnet_conditioning_scale'] = float(conditioning)
    p.task_args['ip_adapter_scale'] = float(strength)
    log.debug(f"InstantID args: {p.task_args}")
    p.task_args['prompt'] = p.all_prompts[0] if p.all_prompts else p.prompt
    p.task_args['negative_prompt'] = p.all_negative_prompts[0] if p.all_negative_prompts else p.negative_prompt
    p.task_args['image_embeds'] = face_embeds[0] # overwrite placeholder

    # run processing
    processed: processing.Processed = processing.process_images(p)
    shared.sd_model.set_ip_adapter_scale(0)
    p.extra_generation_params['InstantID'] = f'{strength}/{conditioning}'

    if not cache:
        controlnet_model = None
        devices.torch_gc()

    # restore original pipeline
    shared.opts.data['prompt_attention'] = orig_prompt_attention
    shared.sd_model = orig_pipeline
    return processed
