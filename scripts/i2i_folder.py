import os
import copy
import gradio as gr
from PIL import Image
from modules import shared, processing, scripts_manager, sd_samplers, images
from modules.logger import log
from modules.files_cache import list_files


class Script(scripts_manager.Script):
    def title(self):
        return "CeeTeeDees I2I folder batch inference"

    def show(self, is_img2img): # pylint: disable=unused-argument
        return True

    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Row():
            gr.HTML('<p><b>The purpose of this script is to assist in Img2Img of folders containing incrementally named images such as one would use when extracting frames from video.</p><br><p>It can use any model though is best done with high consistency models such as Flux.</p><br><p>Note the full path of your folder containing incrementally numbered images such as Frame000 to Frame900 and the script will run Inference on each image in order saving the images with identical names in an output subfolder.</p>')
        with gr.Row():
            folder = gr.Textbox(
                label="Input folder",
                placeholder="Path to folder containing png, jpg, jpeg, webp or jxl images",
                elem_id=self.elem_id("folder"),
            )
        with gr.Row():
            output_dir = gr.Textbox(
                label="Output folder",
                placeholder="<input folder>/output/",
                elem_id=self.elem_id("output_dir"),
            )
        with gr.Row():
            upscale_only = gr.Checkbox(
                label="Upscale only (skip inference, apply post-resize directly to inputs)",
                value=False,
                elem_id=self.elem_id("upscale_only"),
            )
        with gr.Row():
            prompt_override = gr.Textbox(
                label="Prompt",
                placeholder="Leave empty to use the panel prompt",
                elem_id=self.elem_id("prompt_override"),
            )
        with gr.Row():
            negative_override = gr.Textbox(
                label="Negative prompt",
                placeholder="Leave empty to use the panel value",
                elem_id=self.elem_id("negative_override"),
            )
        with gr.Row():
            seed_override = gr.Textbox(
                label="Seed (-1 = use panel)",
                value="-1",
                elem_id=self.elem_id("seed_override"),
            )
        with gr.Row():
            steps_override = gr.Slider(
                minimum=0, maximum=150, step=1, value=0,
                label="Steps (0 = use panel)",
                elem_id=self.elem_id("steps_override"),
            )
        with gr.Row():
            cfg_scale_override = gr.Slider(
                minimum=0.0, maximum=30.0, step=0.5, value=0.0,
                label="Guidance scale (0.0 = use panel)",
                elem_id=self.elem_id("cfg_scale_override"),
            )
        with gr.Row():
            sampler_override = gr.Dropdown(
                label="Sampler (empty = use panel)",
                choices=[""] + [s.name for s in sd_samplers.samplers_for_img2img],
                value="",
                elem_id=self.elem_id("sampler_override"),
            )
        with gr.Row():
            strength_override = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                label="Denoising strength (0.0 = use panel)",
                elem_id=self.elem_id("strength_override"),
            )
        with gr.Row():
            gr.HTML('<b>Pre-inference resize</b>')
        _upscaler_choices = [x.name for x in shared.sd_upscalers] or ["None"]
        with gr.Row():
            pre_resize_enabled = gr.Checkbox(
                label="Enable pre-inference resize",
                value=False,
                elem_id=self.elem_id("pre_resize_enabled"),
            )
        with gr.Row():
            pre_resize_mode = gr.Dropdown(
                label="Resize mode",
                choices=shared.resize_modes,
                type="index",
                value="None",
                elem_id=self.elem_id("pre_resize_mode"),
            )
            pre_resize_name = gr.Dropdown(
                label="Resize method",
                choices=_upscaler_choices,
                value=_upscaler_choices[0],
                elem_id=self.elem_id("pre_resize_name"),
            )
        with gr.Row():
            pre_resize_scale = gr.Slider(
                minimum=0.25, maximum=4.0, step=0.05, value=1.0,
                label="Scale factor (ignored if width/height set)",
                elem_id=self.elem_id("pre_resize_scale"),
            )
        with gr.Row():
            pre_resize_width = gr.Number(
                label="Width (0 = use scale factor)",
                value=0, precision=0,
                elem_id=self.elem_id("pre_resize_width"),
            )
            pre_resize_height = gr.Number(
                label="Height (0 = use scale factor)",
                value=0, precision=0,
                elem_id=self.elem_id("pre_resize_height"),
            )
        with gr.Row():
            gr.HTML('<b>Post-inference resize</b>')
        with gr.Row():
            resize_enabled = gr.Checkbox(
                label="Enable post-inference resize",
                value=False,
                elem_id=self.elem_id("resize_enabled"),
            )
        with gr.Row():
            resize_mode = gr.Dropdown(
                label="Resize mode",
                choices=shared.resize_modes,
                type="index",
                value="None",
                elem_id=self.elem_id("resize_mode"),
            )
            resize_name = gr.Dropdown(
                label="Resize method",
                choices=_upscaler_choices,
                value=_upscaler_choices[0],
                elem_id=self.elem_id("resize_name"),
            )
        with gr.Row():
            resize_scale = gr.Slider(
                minimum=1.0, maximum=8.0, step=0.05, value=2.0,
                label="Scale factor (ignored if width/height set)",
                elem_id=self.elem_id("resize_scale"),
            )
        with gr.Row():
            resize_width = gr.Number(
                label="Width (0 = use scale factor)",
                value=0, precision=0,
                elem_id=self.elem_id("resize_width"),
            )
            resize_height = gr.Number(
                label="Height (0 = use scale factor)",
                value=0, precision=0,
                elem_id=self.elem_id("resize_height"),
            )
        return [folder, output_dir, upscale_only, prompt_override, negative_override, seed_override, steps_override, cfg_scale_override, sampler_override, strength_override, pre_resize_enabled, pre_resize_mode, pre_resize_name, pre_resize_scale, pre_resize_width, pre_resize_height, resize_enabled, resize_mode, resize_name, resize_scale, resize_width, resize_height]

    def run(self, p, folder, output_dir, upscale_only, prompt_override, negative_override, seed_override, steps_override, cfg_scale_override, sampler_override, strength_override, pre_resize_enabled, pre_resize_mode, pre_resize_name, pre_resize_scale, pre_resize_width, pre_resize_height, resize_enabled, resize_mode, resize_name, resize_scale, resize_width, resize_height): # pylint: disable=arguments-differ
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            log.error(f"Image folder batch: invalid or missing folder: {folder!r}")
            return processing.Processed(p, [], p.seed, "Invalid or missing folder")

        files = sorted(list_files(folder, ext_filter=['.png', '.jpg', '.jpeg', '.webp', '.jxl'], recursive=False))
        if not files:
            log.error(f"Image folder batch: no image files found in: {folder!r}")
            return processing.Processed(p, [], p.seed, "No image files found")

        out_dir = (output_dir or "").strip() or os.path.join(folder, "output")
        os.makedirs(out_dir, exist_ok=True)
        resize_out_dir = os.path.join(os.path.dirname(out_dir), "output-resized") if resize_enabled else None
        if resize_out_dir:
            os.makedirs(resize_out_dir, exist_ok=True)

        log.info(f"Image folder batch: folder={folder!r} images={len(files)} output={out_dir!r}")

        processing.fix_seed(p)

        try:
            seed_val = int(str(seed_override).strip())
        except (ValueError, TypeError):
            seed_val = -1
        if seed_val >= 0:
            p.seed = seed_val
        if int(steps_override) > 0:
            p.steps = int(steps_override)
        if float(cfg_scale_override) > 0.0:
            p.cfg_scale = float(cfg_scale_override)
        if str(sampler_override).strip():
            p.sampler_name = str(sampler_override).strip()
        if float(strength_override) > 0.0:
            p.denoising_strength = float(strength_override)
        if prompt_override.strip():
            p.prompt = prompt_override.strip()
        if negative_override.strip():
            p.negative_prompt = negative_override.strip()

        shared.state.job_count = len(files)

        all_images = []
        all_prompts = []
        all_seeds = []
        all_negative = []
        infotexts = []

        for i, filepath in enumerate(files):
            if shared.state.interrupted:
                break
            shared.state.job = f"{i + 1}/{len(files)}"
            shared.state.job_no = i

            img = Image.open(filepath)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            cp = copy.copy(p)
            cp.init_images = [img]
            cp.width = img.width
            cp.height = img.height
            if pre_resize_enabled and pre_resize_mode != 0 and pre_resize_name not in ('None', '') and shared.sd_upscalers:
                pre_w = int(pre_resize_width) if int(pre_resize_width) > 0 else int(img.width * pre_resize_scale)
                pre_h = int(pre_resize_height) if int(pre_resize_height) > 0 else int(img.height * pre_resize_scale)
                img = images.resize_image(pre_resize_mode, img, pre_w, pre_h, pre_resize_name)
                cp.init_images = [img]
                cp.width = img.width
                cp.height = img.height
                log.info(f"Image folder batch: pre-resize to {img.size} mode={shared.resize_modes[pre_resize_mode]!r} method={pre_resize_name!r}")
            if upscale_only:
                log.info(f"Image folder batch: [{i + 1}/{len(files)}] upscale-only file={os.path.basename(filepath)} size={img.size}")
                out_img = img
            else:
                cp.batch_size = 1
                cp.n_iter = 1
                cp.do_not_save_samples = True
                cp.do_not_save_grid = True

                log.info(f"Image folder batch: [{i + 1}/{len(files)}] file={os.path.basename(filepath)} size={img.size} seed={cp.seed}")

                proc = processing.process_images(cp)
                img.close()

                if proc is None or not proc.images:
                    log.warning(f"Image folder batch: no output for {filepath!r}")
                    continue

                out_img = proc.images[0]
                all_prompts += proc.all_prompts
                all_seeds += proc.all_seeds
                all_negative += proc.all_negative_prompts
                infotexts += proc.infotexts

            if resize_enabled and resize_mode != 0 and resize_name not in ('None', '') and shared.sd_upscalers:
                target_w = int(resize_width) if int(resize_width) > 0 else int(out_img.width * resize_scale)
                target_h = int(resize_height) if int(resize_height) > 0 else int(out_img.height * resize_scale)
                resized_img = images.resize_image(resize_mode, out_img, target_w, target_h, resize_name)
                log.info(f"Image folder batch: resized to {resized_img.size} mode={shared.resize_modes[resize_mode]!r} method={resize_name!r}")
                res_name = os.path.basename(filepath)
                resized_img.save(os.path.join(resize_out_dir, res_name))
            out_name = os.path.basename(filepath)
            out_path = os.path.join(out_dir, out_name)
            out_img.save(out_path)
            log.info(f"Image folder batch: saved {out_path!r}")

            all_images.append(out_img)

        return processing.get_processed(p, all_images, p.seed, "", all_prompts=all_prompts, all_seeds=all_seeds, all_negative_prompts=all_negative, infotexts=infotexts)
