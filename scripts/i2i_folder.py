import os
import glob
import copy
import gradio as gr
from PIL import Image
from modules import processing, scripts_manager, sd_samplers, images
from modules import shared
from modules.processing import Processed, get_processed
from modules.shared import state, log


class Script(scripts_manager.Script):
    def title(self):
        return "CeeTeeDee's I2I folder batch inference"

    def show(self, is_img2img): # pylint: disable=unused-argument
        return True

    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Row():
            gr.HTML('<p><h2>Image folder batch inference by CeeTeeDee</h2><p><br>')
        with gr.Row():
            gr.HTML('<p><b>The purpose of this script is to assist in Img2Img of folders containing incrementally named images such as one would use when extracting frames from video.</p><br><p>It can use any model though is best done with high consistency models such as Flux.</p><br><p>Note the full path of your folder containing incrementally numbered images such as Frame000.png to Frame900.png and the script will run Inference on each image in order saving the images with identical names in an output subfolder.</p>')
        with gr.Row():

            folder = gr.Textbox(
                label="Input folder",
                placeholder="Path to folder containing PNG images",
                elem_id=self.elem_id("folder"),
            )
        with gr.Row():
            output_dir = gr.Textbox(
                label="Output folder",
                placeholder="Leave empty to save alongside inputs in <input folder>/output/",
                elem_id=self.elem_id("output_dir"),
            )
        with gr.Row():
            prompt_override = gr.Textbox(
                label="Prompt override",
                placeholder="Leave empty to use the prompt from the main panel",
                elem_id=self.elem_id("prompt_override"),
            )
        with gr.Row():
            negative_override = gr.Textbox(
                label="Negative prompt override",
                placeholder="Leave empty to use the negative prompt from the main panel",
                elem_id=self.elem_id("negative_override"),
            )
        with gr.Row():
            seed_override = gr.Textbox(
                label="Seed override (-1 = use panel seed)",
                value="-1",
                elem_id=self.elem_id("seed_override"),
            )
        with gr.Row():
            steps_override = gr.Slider(
                minimum=0, maximum=150, step=1, value=0,
                label="Steps override (0 = use panel steps)",
                elem_id=self.elem_id("steps_override"),
            )
        with gr.Row():
            cfg_scale_override = gr.Slider(
                minimum=0.0, maximum=30.0, step=0.5, value=0.0,
                label="Guidance scale override (0.0 = use panel value)",
                elem_id=self.elem_id("cfg_scale_override"),
            )
        with gr.Row():
            sampler_override = gr.Dropdown(
                label="Sampler override (empty = use panel sampler)",
                choices=[""] + [s.name for s in sd_samplers.samplers_for_img2img],
                value="",
                elem_id=self.elem_id("sampler_override"),
            )
        with gr.Row():
            strength_override = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                label="Denoising strength override (0.0 = use panel value)",
                elem_id=self.elem_id("strength_override"),
            )
        with gr.Row():
            gr.HTML('<b>Post-inference resize</b>')
        with gr.Row():
            _upscaler_choices = [x.name for x in shared.sd_upscalers] or ["None"]
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
                label="Scale factor",
                elem_id=self.elem_id("resize_scale"),
            )
        return [folder, output_dir, prompt_override, negative_override, seed_override, steps_override, cfg_scale_override, sampler_override, strength_override, resize_mode, resize_name, resize_scale]

    def run(self, p, folder, output_dir, prompt_override, negative_override, seed_override, steps_override, cfg_scale_override, sampler_override, strength_override, resize_mode, resize_name, resize_scale): # pylint: disable=arguments-differ
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            log.error(f"Image folder batch: invalid or missing folder: {folder!r}")
            return Processed(p, [], p.seed, "Invalid or missing folder")

        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not files:
            log.error(f"Image folder batch: no PNG files found in: {folder!r}")
            return Processed(p, [], p.seed, "No PNG files found")

        out_dir = (output_dir or "").strip() or os.path.join(folder, "output")
        os.makedirs(out_dir, exist_ok=True)

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

        state.job_count = len(files)

        all_images = []
        all_prompts = []
        all_seeds = []
        all_negative = []
        infotexts = []

        for i, filepath in enumerate(files):
            if state.interrupted:
                break
            state.job = f"{i + 1}/{len(files)}"
            state.job_no = i

            img = Image.open(filepath)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            cp = copy.copy(p)
            cp.init_images = [img]
            cp.width = img.width
            cp.height = img.height
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
            if resize_mode != 0 and resize_name != 'None':
                target_w = int(out_img.width * resize_scale)
                target_h = int(out_img.height * resize_scale)
                out_img = images.resize_image(resize_mode, out_img, target_w, target_h, resize_name)
                log.info(f"Image folder batch: resized to {out_img.size} mode={shared.resize_modes[resize_mode]!r} method={resize_name!r}")
            out_name = os.path.splitext(os.path.basename(filepath))[0] + ".png"
            out_path = os.path.join(out_dir, out_name)
            out_img.save(out_path)
            log.info(f"Image folder batch: saved {out_path!r}")

            all_images.append(out_img)
            all_prompts += proc.all_prompts
            all_seeds += proc.all_seeds
            all_negative += proc.all_negative_prompts
            infotexts += proc.infotexts

        return get_processed(p, all_images, p.seed, "", all_prompts=all_prompts, all_seeds=all_seeds, all_negative_prompts=all_negative, infotexts=infotexts)
