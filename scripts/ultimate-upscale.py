from enum import Enum
import math
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from modules import processing, shared, images, scripts_manager
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.shared import opts, state
from modules.logger import log

elem_id = "ultimateupscale"

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class USDUpscaler():

    def __init__(self, p, image, upscaler_index:int, tile_width, tile_height) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image.Image = image
        self.scales = []
        self.result_images = []
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        log.info(f'USDUpscaler: canvas={self.p.width}x{self.p.height} image={self.image.width}x{self.image.height} scale={self.scale_factor} scales={list(self.scales)} upscaler={self.upscaler.name}')
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.Resampling.LANCZOS)
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for _index, value in self.scales:
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.Resampling.LANCZOS)

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        if type(self.p.prompt) != list:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.samples_format, info=self.initial_info, p=self.p)
        else:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt[0], opts.samples_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)
        state.job_count = redraw_job_count + seams_job_count

    def add_extra_info(self):
        log.info(f"USDUpscaler: tile={self.redraw.tile_width}x{self.redraw.tile_height} rows={self.rows} cols={self.cols} redraw={self.redraw.enabled} seams={self.seams_fix.mode.name}")
        self.p.extra_generation_params["USDUpscaler upscaler"] = self.upscaler.name
        self.p.extra_generation_params["USDUpscaler tile"] = f'{self.redraw.tile_width}x{self.redraw.tile_height}'
        self.p.extra_generation_params["USDUpscaler blur"] = self.p.mask_blur
        self.p.extra_generation_params["USDUpscaler padding"] = self.redraw.padding

    def process(self):
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
        state.end()

class USDURedraw():
    def __init__(self) -> None:
        self.enabled = False
        self.mode = USDUMode.NONE
        self.padding = 0
        self.tile_width = 0
        self.tile_height = 0
        self.initial_info = None

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height
        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        processed = None
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]
        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)
        return image

    def chess_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        processed = None
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)
        return image

class USDUSeamsFix():
    def __init__(self) -> None:
        self.enabled = False
        self.padding = 0
        self.denoise = 0
        self.mask_blur = 0
        self.width = 0
        self.mode = USDUSFMode.NONE
        self.tile_width = 0
        self.tile_height = 0
        self.initial_info = None

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        self.init_draw(p)
        processed = None
        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.Resampling.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.Resampling.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.Resampling.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.Resampling.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)
        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.Resampling.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2, yi*self.tile_height + self.tile_height//2))
                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if len(processed.images) > 0:
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, rows, cols):
        self.init_draw(p)
        processed = None
        p.denoising_strength = self.denoise
        p.mask_blur = 0
        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.Resampling.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.Resampling.BICUBIC), (0, 128))
        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.Resampling.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.Resampling.BICUBIC)

        for xi in range(1, cols):
            if state.interrupted:
                break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))
            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if len(processed.images) > 0:
                image = processed.images[0]
        for yi in range(1, rows):
            if state.interrupted:
                break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))
            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if len(processed.images) > 0:
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)
        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image

class UltimateSDUpscaleScript(scripts_manager.Script):
    def title(self):
        return "SD Ultimate Upscale"

    def show(self, is_img2img): # pylint: disable=unused-argument
        return True

    def ui(self, is_img2img): # pylint: disable=unused-argument
        elem_id_prefix = f"{self.parent}_{elem_id}"
        target_size_types = ["Default", "Custom", "Rescale"]
        seams_fix_types = ["None", "Band pass", "Half tile offset pass", "Half tile with intersections"]
        redraw_modes = ["Linear", "Chess", "None"]

        with gr.Row():
            gr.HTML('<a href="https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki">&nbsp SD Ultimate Upscale</a><br>')
        with gr.Row():
            target_size_type = gr.Dropdown(label="Target size", elem_id=f"{elem_id_prefix}_target_size_type", choices=target_size_types, type="index", value=next(iter(target_size_types)))
            custom_width = gr.Slider(label='Custom width', elem_id=f"{elem_id_prefix}_custom_width", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_height = gr.Slider(label='Custom height', elem_id=f"{elem_id_prefix}_custom_height", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_scale = gr.Slider(label='Target scale', elem_id=f"{elem_id_prefix}_custom_scale", minimum=1, maximum=16, step=0.2, value=2, visible=False, interactive=True)
        with gr.Row():
            upscaler_index = gr.Dropdown(label='Redraw upscaler', elem_id=f"{elem_id_prefix}_upscaler_index", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
            redraw_mode = gr.Dropdown(label="Redraw mode", elem_id=f"{elem_id_prefix}_redraw_mode", choices=redraw_modes, type="index", value=next(iter(redraw_modes)))
        with gr.Row():
            tile_width = gr.Slider(elem_id=f"{elem_id_prefix}_tile_width", minimum=0, maximum=2048, step=64, label='Tile width', value=1024)
            tile_height = gr.Slider(elem_id=f"{elem_id_prefix}_tile_height", minimum=0, maximum=2048, step=64, label='Tile height', value=0)
        with gr.Row():
            mask_blur = gr.Slider(elem_id=f"{elem_id_prefix}_mask_blur", label='Tile blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(elem_id=f"{elem_id_prefix}_padding", label='Tile padding', minimum=0, maximum=512, step=1, value=32)
        with gr.Row():
            seams_fix_type = gr.Dropdown(label="Seams fix", elem_id=f"{elem_id_prefix}_seams_fix_type", choices=seams_fix_types, type="index", value=next(iter(seams_fix_types)))
        with gr.Row():
            seams_fix_denoise = gr.Slider(label='Seams fix denoise', elem_id=f"{elem_id_prefix}_seams_fix_denoise", minimum=0, maximum=1, step=0.01, value=0.35, visible=False, interactive=True)
            seams_fix_width = gr.Slider(label='Seams fix width', elem_id=f"{elem_id_prefix}_seams_fix_width", minimum=0, maximum=128, step=1, value=64, visible=False, interactive=True)
            seams_fix_mask_blur = gr.Slider(label='Seams fix mask blur', elem_id=f"{elem_id_prefix}_seams_fix_mask_blur", minimum=0, maximum=64, step=1, value=4, visible=False, interactive=True)
            seams_fix_padding = gr.Slider(label='Seams fix padding', elem_id=f"{elem_id_prefix}_seams_fix_padding", minimum=0, maximum=128, step=1, value=16, visible=False, interactive=True)

        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2 or fix_index == 3
            width_visible = fix_index == 1
            return [gr.update(visible=all_visible), gr.update(visible=width_visible), gr.update(visible=mask_blur_visible), gr.update(visible=all_visible)]

        seams_fix_type.change(fn=select_fix_type, inputs=seams_fix_type, outputs=[seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding])

        def select_scale_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2
            return [gr.update(visible=is_custom_size), gr.update(visible=is_custom_size), gr.update(visible=is_custom_scale)]

        target_size_type.change(fn=select_scale_type, inputs=target_size_type, outputs=[custom_width, custom_height, custom_scale])

        def init_field(scale_name):
            try:
                scale_index = target_size_types.index(scale_name)
                custom_width.visible = custom_height.visible = scale_index == 1
                custom_scale.visible = scale_index == 2
            except Exception:
                pass

        target_size_type.init_field = init_field

        return [tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                upscaler_index, redraw_mode, seams_fix_mask_blur,
                seams_fix_type, target_size_type, custom_width, custom_height, custom_scale]

    def run(self, p: StableDiffusionProcessing, *args):
        tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding, upscaler_index, redraw_mode, seams_fix_mask_blur, seams_fix_type, target_size_type, custom_width, custom_height, custom_scale = args

        if p.init_images is None or len(p.init_images) == 0: # Dont run on t2i
            return None

        # Init
        processing.fix_seed(p)

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False
        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        # Init image
        init_img = p.init_images[0]
        if init_img is None:
            return Processed(p, [], p.seed, "Empty image")
        init_img = images.flatten(init_img, opts.img2img_background_color)

        #override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(p, init_img, upscaler_index, tile_width, tile_height)
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, p.seed, upscaler.initial_info if upscaler.initial_info is not None else "")
