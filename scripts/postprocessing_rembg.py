import os
import gradio as gr
from PIL import Image
from modules import errors, processing, scripts_manager, scripts_postprocessing

models = [
    "none",
    "lucida",
    "ben2",
    "silueta",
    "u2net",
    "u2net_human_seg",
    "isnet-general-use",
    "isnet-anime",
    # "u2netp",
    # "u2net_cloth_seg",
    # "sam",
]


def dependencies():
    from installer import install
    for pkg in ["dctorch==0.1.2", "pymatting", "pooch", "rembg", "numba", "kornia", "kornia-rs"]:
        install(pkg, no_deps=True, ignore=False)


def create_ui(parent):
    def visible(model):
        return gr.update(visible=model not in ["none"]), gr.update(visible=model in ["ben2"]), gr.update(visible=model not in ["none", "ben2"])

    elem_prefix = "extras" if parent == "postprocess" else parent
    accordion_id = "postprocess_rembg_accordion" if parent == "postprocess" else f'{parent}_rembg_accordion'
    with gr.Accordion('Remove background', open=False, elem_id=accordion_id):
        with gr.Row():
            model = gr.Dropdown(label="Model", choices=models, value="none", elem_id=f"{elem_prefix}_rembg_model")
        with gr.Group(visible=False) as group_base:
            with gr.Row():
                merge_alpha = gr.Checkbox(label="Merge alpha", value=False)
                mask_only = gr.Checkbox(label="Mask only", value=False)
        with gr.Group(visible=False) as group_ben2:
            with gr.Row():
                refine = gr.Checkbox(label="Refine foreground", value=False)
        with gr.Group(visible=False) as group_rembg:
            with gr.Row():
                postprocess_mask = gr.Checkbox(label="Postprocess mask", value=False, elem_id=f"{elem_prefix}_rembg_process_mask")
                alpha_matting = gr.Checkbox(label="Alpha matting", value=False, elem_id=f"{elem_prefix}_rembg_alpha")
            with gr.Row(visible=True) as alpha_mask_row:
                alpha_matting_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=40, step=1, value=10, elem_id=f"{elem_prefix}_rembg_alpha_erode")
                alpha_matting_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240, elem_id=f"{elem_prefix}_rembg_alpha_foreground")
                alpha_matting_background_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value=10, elem_id=f"{elem_prefix}_rembg_alpha_background")
            alpha_matting.change(fn=lambda x: gr.update(visible=x), inputs=[alpha_matting], outputs=[alpha_mask_row])
        model.change(fn=lambda x: visible(x), inputs=[model], outputs=[group_base, group_ben2, group_rembg])
    return {
        "model": model,
        "merge_alpha": merge_alpha,
        "refine": refine,
        "mask_only": mask_only,
        "postprocess_mask": postprocess_mask,
        "alpha_matting": alpha_matting,
        "alpha_matting_foreground_threshold": alpha_matting_foreground_threshold,
        "alpha_matting_background_threshold": alpha_matting_background_threshold,
        "alpha_matting_erode_size": alpha_matting_erode_size,
    }


def remove_background(image, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size):
    from modules.logger import log
    if not model or model == "none" or image is None:
        return image, False
    log.info(f'RemoveBackground: model={model} merge_alpha={merge_alpha} refine={refine} mask_only={mask_only} postprocess_mask={postprocess_mask} alpha_matting={alpha_matting} alpha_matting_foreground_threshold={alpha_matting_foreground_threshold} alpha_matting_background_threshold={alpha_matting_background_threshold} alpha_matting_erode_size={alpha_matting_erode_size}')
    dependencies()
    if model == 'ben2':
        try:
            from modules.rembg import ben2
            image = ben2.remove(image, refine=refine)
        except Exception as e:
            log.error(f'RemoveBackground: model={model} {e}')
            errors.display(e, 'Rembg model=ben2')
            return image, False
    elif model == 'lucida':
        try:
            from modules.rembg import lucida
            image = lucida.remove(image)
        except Exception as e:
            log.error(f'RemoveBackground: model={model} {e}')
            errors.display(e, 'Rembg model=lucida')
            return image, False
    else:
        try:
            import rembg
            if "U2NET_HOME" not in os.environ:
                from modules.paths import models_path
                os.environ["U2NET_HOME"] = os.path.join(models_path, "Rembg")
            image = rembg.remove(image,
                                post_process_mask=postprocess_mask,
                                alpha_matting=alpha_matting,
                                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                                alpha_matting_background_threshold=alpha_matting_background_threshold,
                                alpha_matting_erode_size=alpha_matting_erode_size,
                                session=rembg.new_session(model))
        except Exception as e:
            log.error(f'RemoveBackground: model={model} {e}')
            errors.display(e, f'Rembg model={model}')
            return image, False

    if mask_only and image.mode == "RGBA":
        _r, _g, _b, alpha = image.split()
        image = alpha
    if merge_alpha and image.mode == "RGBA":
        flattened = Image.new("RGBA", image.size, "BLACK")
        flattened.paste(image, mask=image)
        image = flattened.convert("RGB")
    return image, True


class ScriptRembg(scripts_manager.Script):
    def title(self):
        return "Remove background"

    def show(self, *args, **kwargs): # pylint: disable=unused-argument
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return list(create_ui(self.parent).values())

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size): # pylint: disable=arguments-differ,unused-argument
        pp.image, _success = remove_background(pp.image, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size)


class ScriptPostprocessingRembg(scripts_postprocessing.ScriptPostprocessing):
    name = "Remove background"
    order = 20000

    def ui(self):
        return create_ui("postprocess")

    def process(self, pp: scripts_postprocessing.PostprocessedImage, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size): # pylint: disable=arguments-differ
        if not model or model == "none":
            return pp
        if isinstance(pp, Image.Image):
            image = pp
            info = {}
        else:
            image = pp.image
            info = pp.info
        image, success = remove_background(image, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size)
        if not success:
            return pp
        if isinstance(pp, Image.Image):
            pp = scripts_postprocessing.PostprocessedImage(image=image, info={**info, "Rembg": model})
        else:
            pp.image = image
            pp.info['Rembg'] = model
        return pp
