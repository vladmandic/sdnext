import os
import sys
import queue
import datetime
import threading
import piexif.helper
from PIL import Image, PngImagePlugin
from modules import shared, script_callbacks, errors, paths
from modules.logger import log
from modules.json_helpers import writefile
from modules.image.grid import check_grid_size
from modules.image.namegen import FilenameGenerator
from modules.image.watermark import set_watermark


debug = log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_save = log.trace if os.environ.get('SD_SAVE_DEBUG', None) is not None else lambda *args, **kwargs: None


def sanitize_filename_part(text, replace_spaces=True):
    if text is None:
        return None
    if replace_spaces:
        text = text.replace(' ', '_')
    invalid_filename_chars = '#<>:"/\\|?*\n\r\t'
    invalid_filename_prefix = ' '
    invalid_filename_postfix = ' .'
    max_filename_part_length = 64
    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text


def atomically_save_image():
    Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes
    while True:
        image, filename, extension, params, exifinfo, filename_txt, is_grid = save_queue.get()
        jobid = shared.state.begin('Save image')
        shared.state.image_history += 1
        if len(exifinfo) > 2:
            with open(paths.params_path, "w", encoding="utf8") as file:
                file.write(exifinfo)
        fn = filename + extension
        filename = filename.strip()
        if extension[0] != '.': # add dot if missing
            extension = '.' + extension
        try:
            image_format = Image.registered_extensions()[extension]
        except Exception:
            log.warning(f'Save: unknown image format: {extension}')
            image_format = 'JPEG'
        exifinfo = (exifinfo or "") if shared.opts.image_metadata else ""
        # additional metadata saved in files
        if shared.opts.save_txt and len(exifinfo) > 0:
            try:
                with open(filename_txt, "w", encoding="utf8") as file:
                    file.write(f"{exifinfo}\n")
                log.info(f'Save: text="{filename_txt}" len={len(exifinfo)}')
            except Exception as e:
                log.warning(f'Save failed: description={filename_txt} {e}')

        # actual save
        if image_format == 'PNG':
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in params.pnginfo.items():
                pnginfo_data.add_text(k, str(v))
            debug_save(f'Save pnginfo: {params.pnginfo.items()}')
            save_args = { 'compress_level': 6, 'pnginfo': pnginfo_data if shared.opts.image_metadata else None }
        elif image_format == 'JPEG':
            if image.mode == 'RGBA':
                log.warning('Save: removing alpha channel')
                image = image.convert("RGB")
            elif image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("L")
            save_args = { 'optimize': True, 'quality': shared.opts.jpeg_quality }
            if shared.opts.image_metadata:
                debug_save(f'Save exif: {exifinfo}')
                save_args['exif'] = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        elif image_format == 'WEBP':
            if image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
            save_args = { 'optimize': True, 'quality': shared.opts.jpeg_quality, 'lossless': shared.opts.webp_lossless }
            if shared.opts.image_metadata:
                debug_save(f'Save exif: {exifinfo}')
                save_args['exif'] = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        elif image_format == 'JXL':
            if image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
            elif image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA")
            save_args = { 'optimize': True, 'quality': shared.opts.jpeg_quality, 'lossless': shared.opts.webp_lossless }
            if shared.opts.image_metadata:
                debug_save(f'Save exif: {exifinfo}')
                save_args['exif'] = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        else:
            save_args = { 'quality': shared.opts.jpeg_quality }
        try:
            debug_save(f'Save args: {save_args}')
            image.save(fn, format=image_format, **save_args)
        except Exception as e:
            log.error(f'Save failed: file="{fn}" format={image_format} args={save_args} {e}')
            errors.display(e, 'Image save')
        size = os.path.getsize(fn) if os.path.exists(fn) else 0
        what = 'grid' if is_grid else 'image'
        log.info(f'Save: {what}="{fn}" type={image_format} width={image.width} height={image.height} size={size}')

        if shared.opts.save_log_fn != '' and len(exifinfo) > 0:
            fn = os.path.join(paths.data_path, shared.opts.save_log_fn)
            if not fn.endswith('.json'):
                fn += '.json'
            entries = shared.readfile(fn, silent=True)
            if not isinstance(entries, list):
                entries = []
            idx = len(entries)
            entry = { 'id': idx, 'filename': filename, 'time': datetime.datetime.now().isoformat(), 'info': exifinfo }
            entries.append(entry)
            writefile(entries, fn, mode='w', silent=True)
            log.info(f'Save: json="{fn}" records={len(entries)}')
        shared.state.outputs(filename)
        shared.state.end(jobid)
        save_queue.task_done()


save_queue: queue.Queue[tuple[Image.Image, str, str, script_callbacks.ImageSaveParams, str, str | None, bool]] = queue.Queue()
save_thread = threading.Thread(target=atomically_save_image, daemon=True)
save_thread.start()


def save_image(image,
               path=None,
               basename='',
               seed=None,
               prompt=None,
               extension=shared.opts.samples_format,
               info=None,
               grid=False,
               pnginfo_section_name='parameters',
               p=None,
               existing_info=None,
               forced_filename=None,
               suffix='',
               save_to_dirs=None,
            ):
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    debug_save(f'Save: fn={fn}') # pylint: disable=protected-access
    if image is None:
        log.warning('Image is none')
        return None, None, None
    if isinstance(image, list):
        if len(image) > 1:
            log.warning(f'Save: images={image} multiple images provided only the first one will be saved')
        image = image[0]
    if not check_grid_size([image]):
        return None, None, None
    if path is None or path == '': # set default path to avoid errors when functions are triggered manually or via api and param is not set
        path = paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save)
    namegen = FilenameGenerator(p, seed, prompt, image, grid=grid)
    suffix = suffix if suffix is not None else ''
    basename = '' if basename is None else basename
    if save_to_dirs is not None and isinstance(save_to_dirs, str) and len(save_to_dirs) > 0:
        dirname = save_to_dirs
        path = os.path.join(path, dirname)
    elif shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        path = os.path.join(path, dirname)
    if forced_filename is None:
        if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0:
            file_decoration = shared.opts.samples_filename_pattern
        else:
            file_decoration = "[seq]-[prompt_words]"
        file_decoration = namegen.apply(file_decoration)
        file_decoration += suffix
        if file_decoration.startswith(basename):
            basename = ''
        filename = os.path.join(path, f"{file_decoration}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{file_decoration}.{extension}")
    else:
        forced_filename += suffix
        if forced_filename.startswith(basename):
            basename = ''
        filename = os.path.join(path, f"{forced_filename}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{forced_filename}.{extension}")
    pnginfo = existing_info or {}
    if info is None:
        info = image.info.get(pnginfo_section_name, '')
    if info is not None:
        pnginfo[pnginfo_section_name] = info

    wm_text = getattr(p, 'watermark_text', shared.opts.image_watermark)
    wm_image = getattr(p, 'watermark_image', shared.opts.image_watermark_image)
    image = set_watermark(image, wm_text, wm_image)

    params = script_callbacks.ImageSaveParams(image, p, filename, pnginfo)
    params.filename = namegen.sanitize(filename)
    dirname = os.path.dirname(params.filename)
    if dirname is not None and len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    params.filename = namegen.sequence(params.filename)
    params.filename = namegen.sanitize(params.filename)
    # callbacks
    script_callbacks.before_image_saved_callback(params)
    exifinfo = params.pnginfo.get('UserComment', '')
    exifinfo = exifinfo + ', ' if len(exifinfo) > 0 else ''
    exifinfo += params.pnginfo.get(pnginfo_section_name, '')
    filename, extension = os.path.splitext(params.filename)
    filename_txt = f"{filename}.txt" if shared.opts.save_txt and len(exifinfo) > 0 else None
    save_queue.put((params.image, filename, extension, params, exifinfo, filename_txt, grid)) # actual save is executed in a thread that polls data from queue
    save_queue.join()
    if not hasattr(params.image, 'already_saved_as'):
        debug(f'Image marked: "{params.filename}"')
        params.image.already_saved_as = params.filename
    script_callbacks.image_saved_callback(params)
    return params.filename, filename_txt, exifinfo
