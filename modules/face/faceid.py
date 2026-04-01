import os
import cv2
import torch
import numpy as np
import diffusers
import huggingface_hub as hf
from PIL import Image
from modules import processing, shared, devices, extra_networks, sd_hijack_freeu, script_callbacks, ipadapter, token_merge
from modules.logger import log
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet


FACEID_MODELS = {
    "FaceID Base": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin",
    "FaceID Plus v1": "h94/IP-Adapter-FaceID/ip-adapter-faceid-plus_sd15.bin",
    "FaceID Plus v2": "h94/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin",
    "FaceID XL": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
    # "FaceID Portrait v10": "h94/IP-Adapter-FaceID/ip-adapter-faceid-portrait_sd15.bin",
    # "FaceID Portrait v11": "h94/IP-Adapter-FaceID/ip-adapter-faceid-portrait-v11_sd15.bin",
    # "FaceID XL Plus v2": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
}

faceid_model_weights = None
faceid_model_name = None
debug = log.trace if os.environ.get("SD_FACE_DEBUG", None) is not None else lambda *args, **kwargs: None


def hijack_load_ip_adapter(self):
    self.image_proj_model.load_state_dict(faceid_model_weights["image_proj"])
    ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
    ip_layers.load_state_dict(faceid_model_weights["ip_adapter"], strict=False)


def face_id(
    p: processing.StableDiffusionProcessing,
    app,
    source_images: list[Image.Image],
    model: str,
    override: bool,
    cache: bool,
    scale: float,
    structure: float,
):
    global faceid_model_weights, faceid_model_name  # pylint: disable=global-statement
    if source_images is None or len(source_images) == 0:
        log.warning('FaceID: no input images')
        return None

    from insightface.utils import face_align
    try:
        from ip_adapter.ip_adapter_faceid import (
            IPAdapterFaceID,
            IPAdapterFaceIDPlus,
            IPAdapterFaceIDXL,
            IPAdapterFaceIDPlusXL,
        )
        from ip_adapter.ip_adapter_faceid_separate import (
            IPAdapterFaceID as IPAdapterFaceIDPortrait,
        )
    except Exception as e:
        log.error(f"FaceID incorrect version of ip_adapter: {e}")
        return None

    processed_images = []

    faceid_model = None
    original_load_ip_adapter = None

    try:
        shared.prompt_styles.apply_styles_to_extra(p)

        if shared.opts.cuda_compile_backend == 'none':
            token_merge.apply_token_merging(shared.sd_model)
            sd_hijack_freeu.apply_freeu(p)

        script_callbacks.before_process_callback(p)

        with context_hypertile_vae(p), context_hypertile_unet(p), devices.inference_context():
            ip_ckpt = FACEID_MODELS[model]
            folder, filename = os.path.split(ip_ckpt)
            basename, _ext = os.path.splitext(filename)
            model_path = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.hfcache_dir)
            if model_path is None:
                log.error(f'FaceID download failed: model={model} file="{ip_ckpt}"')
                return None
            if faceid_model_weights is None or faceid_model_name != model or not cache:
                log.debug(f'FaceID load: model={model} file="{ip_ckpt}"')
                faceid_model_weights = torch.load(model_path, map_location="cpu")
            else:
                log.debug(f'FaceID cached: model={model} file="{ip_ckpt}"')

            if "XL Plus" in model and shared.sd_model_type == 'sd':
                image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                original_load_ip_adapter = IPAdapterFaceIDPlusXL.load_ip_adapter
                IPAdapterFaceIDPlusXL.load_ip_adapter = hijack_load_ip_adapter
                faceid_model = IPAdapterFaceIDPlusXL(
                    sd_pipe=shared.sd_model,
                    image_encoder_path=image_encoder_path,
                    ip_ckpt=model_path,
                    lora_rank=128,
                    num_tokens=4,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            elif "XL" in model and shared.sd_model_type == 'sdxl':
                original_load_ip_adapter = IPAdapterFaceIDXL.load_ip_adapter
                IPAdapterFaceIDXL.load_ip_adapter = hijack_load_ip_adapter
                faceid_model = IPAdapterFaceIDXL(
                    sd_pipe=shared.sd_model,
                    ip_ckpt=model_path,
                    lora_rank=128,
                    num_tokens=4,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            elif "Plus" in model and shared.sd_model_type == 'sd':
                original_load_ip_adapter = IPAdapterFaceIDPlus.load_ip_adapter
                IPAdapterFaceIDPlus.load_ip_adapter = hijack_load_ip_adapter
                image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                faceid_model = IPAdapterFaceIDPlus(
                    sd_pipe=shared.sd_model,
                    image_encoder_path=image_encoder_path,
                    ip_ckpt=model_path,
                    lora_rank=128,
                    num_tokens=4,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            elif "Portrait" in model and shared.sd_model_type == 'sd':
                original_load_ip_adapter = IPAdapterFaceIDPortrait.load_ip_adapter
                IPAdapterFaceIDPortrait.load_ip_adapter = hijack_load_ip_adapter
                faceid_model = IPAdapterFaceIDPortrait(
                    sd_pipe=shared.sd_model,
                    ip_ckpt=model_path,
                    num_tokens=16,
                    n_cond=5,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            elif "Base" in model and shared.sd_model_type == 'sd':
                original_load_ip_adapter = IPAdapterFaceID.load_ip_adapter
                IPAdapterFaceID.load_ip_adapter = hijack_load_ip_adapter
                faceid_model = IPAdapterFaceID(
                    sd_pipe=shared.sd_model,
                    ip_ckpt=model_path,
                    lora_rank=128,
                    num_tokens=4,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            else:
                log.error(f'FaceID model not supported: model="{model}" class={shared.sd_model.__class__.__name__}')
                return None

            if override:
                shared.sd_model.scheduler = diffusers.DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                    steps_offset=1,
                )

            shortcut = "v2" in model
            faceid_model_name = model
            face_embeds = []
            face_images = []

            for i, source_image in enumerate(source_images):
                np_image = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
                faces = app.get(np_image)
                if len(faces) == 0:
                    log.error("FaceID: no faces found")
                    break
                face_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0))
                face_images.append(face_align.norm_crop(np_image, landmark=faces[0].kps, image_size=224))
                log.debug(f'FaceID face: i={i+1} score={faces[0].det_score:.2f} gender={"female" if faces[0].gender==0 else "male"} age={faces[0].age} bbox={faces[0].bbox}')
                p.extra_generation_params[f"FaceID {i+1}"] = f'{faces[0].det_score:.2f} {"female" if faces[0].gender==0 else "male"} {faces[0].age}y'

            if len(face_embeds) == 0:
                log.error("FaceID: no faces found")
                return None

            face_embeds = torch.cat(face_embeds, dim=0)
            ip_model_dict = {  # main generate dict
                "num_samples": p.batch_size,
                "width": p.width,
                "height": p.height,
                "num_inference_steps": p.steps,
                "scale": scale,
                "guidance_scale": p.cfg_scale,
                "faceid_embeds": face_embeds.shape,  # placeholder
            }

            # optional generate dict
            if shortcut is not None:
                ip_model_dict["shortcut"] = shortcut
            if "Plus" in model:
                ip_model_dict["s_scale"] = structure
            log.debug(f"FaceID args: {ip_model_dict}")
            if "Plus" in model:
                ip_model_dict["face_image"] = face_images
            ip_model_dict["faceid_embeds"] = face_embeds # overwrite placeholder
            faceid_model.set_scale(scale)

            if not p.all_prompts:
                processing.process_init(p)
                p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
            for n in range(p.n_iter):
                p.iteration = n
                p.prompts = p.all_prompts[n * p.batch_size:(n+1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n+1) * p.batch_size]
                p.seeds = p.all_seeds[n * p.batch_size:(n+1) * p.batch_size]
                p.subseeds = p.all_subseeds[n * p.batch_size:(n+1) * p.batch_size]
                p.prompts, p.network_data = extra_networks.parse_prompts(p.prompts)

                extra_networks.activate(p, p.network_data)
                ip_model_dict.update({
                        "prompt": p.prompts[0],
                        "negative_prompt": p.negative_prompts[0],
                        "seed": p.seeds[0],
                    })
                debug(f"FaceID: {ip_model_dict}")
                res = faceid_model.generate(**ip_model_dict)
                if isinstance(res, list):
                    processed_images += res

            faceid_model.set_scale(0)
            faceid_model = None

            if not cache:
                faceid_model_weights = None
                faceid_model_name = None
            devices.torch_gc()

        ipadapter.unapply(shared.sd_model)
        extra_networks.deactivate(p, p.network_data)

        p.extra_generation_params["IP Adapter"] = f"{basename}:{scale}"
    finally:
        if faceid_model is not None and original_load_ip_adapter is not None:
            faceid_model.__class__.load_ip_adapter = original_load_ip_adapter
        if shared.opts.cuda_compile_backend == 'none':
            token_merge.remove_token_merging(shared.sd_model)
        script_callbacks.after_process_callback(p)

    return processed_images
