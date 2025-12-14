import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from sdnext_core import MODELDATA
from modules import images, processing, shared, scripts_manager
from modules.processing import get_processed
from modules.shared import opts, state


class Script(scripts_manager.Script):
    def title(self):
        return "HDR: High Dynamic Range"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML("<span>&nbsp HDR: High Dynamic Range</span><br>")
        with gr.Row():
            save_hdr = gr.Checkbox(label="Save HDR image", value=True)
            hdr_range = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.65, label='HDR range')
        with gr.Row():
            is_tonemap = gr.Checkbox(label="Enable tonemap", value=False)
            gamma = gr.Slider(minimum=0, maximum=2, step=0.05, value=1.0, label='Gamma', visible=False)
        with gr.Row():
            scale = gr.Slider(minimum=0, maximum=2, step=0.05, value=1.0, label='Scale', visible=False)
            saturation = gr.Slider(minimum=0, maximum=2, step=0.05, value=1.0, label='Saturation', visible=False)
        is_tonemap.change(fn=self.change_tonemap, inputs=[is_tonemap], outputs=[gamma, scale, saturation])
        return [hdr_range, save_hdr, is_tonemap, gamma, scale, saturation]

    def change_tonemap(self, is_tonemap):
        return [gr.update(visible=is_tonemap), gr.update(visible=is_tonemap), gr.update(visible=is_tonemap)]

    def merge(self, imgs: list, is_tonemap: bool, gamma, scale, saturation):
        shared.log.info(f'HDR: merge images={len(imgs)} tonemap={is_tonemap} sgamma={gamma} scale={scale} saturation={saturation}')
        imgs_np = [np.asarray(img).astype(np.uint8) for img in imgs]

        align = cv2.createAlignMTB()
        align.process(imgs_np, imgs_np)

        # cv2.createMergeRobertson()
        # cv2.createMergeDebevec()
        merge = cv2.createMergeMertens()
        hdr = merge.process(imgs_np)

        # cv2.createTonemapDrago()
        # cv2.createTonemapReinhard()
        if is_tonemap:
            tonemap = cv2.createTonemapMantiuk(gamma, scale, saturation)
            hdr = tonemap.process(hdr)

        ldr = np.clip(hdr * 255, 0, 255).astype(np.uint8)
        hdr = np.clip(hdr * 65535, 0, 65535).astype(np.uint16)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        return hdr, ldr

    def run(self, p, hdr_range, save_hdr, is_tonemap, gamma, scale, saturation): # pylint: disable=arguments-differ
        if MODELDATA.sd_model_type != 'sd' and MODELDATA.sd_model_type != 'sdxl':
            shared.log.error(f'HDR: incorrect base model: {MODELDATA.sd_model.__class__.__name__}')
            return None
        p.extra_generation_params = {
            "HDR range": hdr_range,
        }
        shared.log.info(f'HDR: range={hdr_range}')
        processing.fix_seed(p)
        imgs = []
        info = ''
        for i in range(3):
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.hdr_brightness = (i - 1) * (2.0 * hdr_range)
            p.hdr_mode = 0
            p.task_args['seed'] = p.seed
            processed: processing.Processed = processing.process_images(p)
            imgs += processed.images
            if i == 1:
                info = processed.info
            if state.interrupted:
                break

        if len(imgs) > 1:
            hdr, ldr = self.merge(imgs, is_tonemap, gamma, scale, saturation)
            img = Image.fromarray(ldr)
            if save_hdr:
                saved_fn, _txt, _exif = images.save_image(img, shared.opts.outdir_save, "", p.seed, p.prompt, opts.grid_format, info=processed.info, p=p)
                fn = os.path.splitext(saved_fn)[0] + '-hdr.png'
                # cv2.imwrite(fn, hdr, [cv2.IMWRITE_PNG_COMPRESSION, 6, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, cv2.IMWRITE_HDR_COMPRESSION, cv2.IMWRITE_HDR_COMPRESSION_RLE])
                cv2.imwrite(fn, hdr)
                shared.log.debug(f'Save: image="{fn}" type=PNG mode=HDR channels=16 size={os.path.getsize(fn)}')
            # if opts.grid_save:
            #    images.save_image(grid, p.outpath_grids, "grid", p.seed, p.prompt, opts.grid_format, info=processed.info, grid=True, p=p)
            grid = [images.image_grid(imgs, rows=1)] if opts.return_grid else []
            imgs = [img] + grid

        processed = get_processed(p, images_list=imgs, seed=p.seed, info=info)
        return processed
