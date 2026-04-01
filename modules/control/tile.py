import time
from PIL import Image
from modules import shared, processing, images, sd_models, sd_vae
from modules.logger import log


def get_tile(image: Image.Image, x: int, y: int, sx: int, sy: int) -> Image.Image:
    return image.crop((
        (x + 0) * image.width // sx,
        (y + 0) * image.height // sy,
        (x + 1) * image.width // sx,
        (y + 1) * image.height // sy
    ))


def set_tile(image: Image.Image, x: int, y: int, tiled: Image.Image):
    image.paste(tiled, (x * tiled.width, y * tiled.height))
    return image


def run_tiling(p: processing.StableDiffusionProcessing, input_image: Image.Image) -> processing.Processed:
    t0 = time.time()
    # prepare images
    sx, sy = p.control_tile.split('x')
    sx = int(sx)
    sy = int(sy)
    vae_scale_factor = sd_vae.get_vae_scale_factor()
    if sx <= 0 or sy <= 0:
        raise ValueError('Control Tile: invalid tile size')
    control_image = p.task_args.get('control_image', None) or p.task_args.get('image', None)
    control_upscaled = None
    if isinstance(control_image, list) and len(control_image) > 0:
        w, h = vae_scale_factor * int(sx * control_image[0].width) // vae_scale_factor, vae_scale_factor * int(sy * control_image[0].height) // vae_scale_factor
        control_upscaled = images.resize_image(resize_mode=1 if sx==sy else 5, im=control_image[0], width=w, height=h, context='add with forward')
    init_image = p.override or input_image
    init_upscaled = None
    if init_image is not None:
        w, h = vae_scale_factor * int(sx * init_image.width) // vae_scale_factor, vae_scale_factor * int(sy * init_image.height) // vae_scale_factor
        init_upscaled = images.resize_image(resize_mode=1 if sx==sy else 5, im=init_image, width=w, height=h, context='add with forward')
    t1 = time.time()
    log.debug(f'Control Tile: scale={sx}x{sy} resize={"fixed" if sx==sy else "context"} control={control_upscaled} init={init_upscaled} time={t1-t0:.3f}')

    # stop processing from restoring pipeline on each iteration
    orig_restore_pipeline = getattr(shared.sd_model, 'restore_pipeline', None)
    shared.sd_model.restore_pipeline = None

    # run tiling
    for x in range(sx):
        for y in range(sy):
            log.info(f'Control Tile: tile={x+1}-{sx}/{y+1}-{sy} target={control_upscaled}')
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            p.init_images = None
            p.task_args['control_mode'] = p.control_mode
            p.task_args['strength'] = p.denoising_strength
            if init_upscaled is not None:
                p.task_args['image'] = [get_tile(init_upscaled, x, y, sx, sy)]
            if control_upscaled is not None:
                p.task_args['control_image'] = [get_tile(control_upscaled, x, y, sx, sy)]
            processed: processing.Processed = processing.process_images(p) # run actual pipeline
            if processed is None or len(processed.images) == 0:
                continue
            control_upscaled = set_tile(control_upscaled, x, y, processed.images[0])

    # post-process
    p.width = control_upscaled.width
    p.height = control_upscaled.height
    processed.images = [control_upscaled]
    processed.info = processed.infotext(p, 0)
    processed.infotexts = [processed.info]
    shared.sd_model.restore_pipeline = orig_restore_pipeline
    if hasattr(shared.sd_model, 'restore_pipeline') and shared.sd_model.restore_pipeline is not None:
        shared.sd_model.restore_pipeline()
    t2 = time.time()
    log.debug(f'Control Tile: image={control_upscaled} time={t2-t0:.3f}')
    return processed
