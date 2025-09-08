import os
import json
import rich.progress as p
from PIL import Image
from modules import shared, errors, paths


pbar = None


def save_video_frame(filepath: str):
    from modules import video
    try:
        frames, fps, duration, w, h, codec, frame = video.get_video_params(filepath, capture=True)
    except Exception as e:
        shared.log.error(f'Video: file={filepath} {e}')
        return None
    if frame is not None:
        basename = os.path.splitext(filepath)
        thumb = f'{basename[0]}.thumb.jpg'
        shared.log.debug(f'Video: file={filepath} frames={frames} fps={fps} size={w}x{h} codec={codec} duration={duration} thumb={thumb}')
        frame.save(thumb)
    else:
        shared.log.error(f'Video: file={filepath} no frames found')
    return frame


def download_civit_meta(model_path: str, model_id):
    fn = os.path.splitext(model_path)[0] + '.json'
    url = f'https://civitai.com/api/v1/models/{model_id}'
    r = shared.req(url)
    if r.status_code == 200:
        try:
            data = r.json()
            shared.writefile(data, filename=fn, mode='w', silent=True)
            shared.log.info(f'CivitAI download: id={model_id} url={url} file="{fn}"')
            return r.status_code, len(data), '' # code/size/note
        except Exception as e:
            errors.display(e, 'civitai meta')
            shared.log.error(f'CivitAI meta: id={model_id} url={url} file="{fn}" {e}')
            return r.status_code, '', str(e)
    return r.status_code, '', ''


def download_civit_preview(model_path: str, preview_url: str):
    global pbar # pylint: disable=global-statement
    if model_path is None:
        pbar = None
        return 500, '', ''
    ext = os.path.splitext(preview_url)[1]
    preview_file = os.path.splitext(model_path)[0] + ext
    is_video = preview_file.lower().endswith('.mp4')
    is_json = preview_file.lower().endswith('.json')
    if is_json:
        shared.log.warning(f'CivitAI download: url="{preview_url}" skip json')
        return 500, '', 'exepected preview image got json'
    if os.path.exists(preview_file):
        return 304, '', 'already exists'
    # res = f'CivitAI download: url={preview_url} file="{preview_file}"'
    r = shared.req(preview_url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 16384 # 16KB blocks
    written = 0
    img = None
    shared.state.begin('CivitAI')
    if pbar is None:
        pbar = p.Progress(p.TextColumn('[cyan]Download'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn(), p.TextColumn('[yellow]{task.description}'), console=shared.console)
    try:
        with open(preview_file, 'wb') as f:
            with pbar:
                task = pbar.add_task(description=preview_file, total=total_size)
                for data in r.iter_content(block_size):
                    written = written + len(data)
                    f.write(data)
                    pbar.update(task, advance=block_size)
        if written < 1024: # min threshold
            os.remove(preview_file)
            return 400, '', 'removed invalid download'
        if is_video:
            img = save_video_frame(preview_file)
        else:
            img = Image.open(preview_file)
    except Exception as e:
        shared.log.error(f'CivitAI download error: url={preview_url} file="{preview_file}" written={written} {e}')
        return 500, '', str(e)
    shared.state.end()
    if img is None:
        return 500, '', 'image is none'
    shared.log.info(f'CivitAI download: url={preview_url} file="{preview_file}" size={total_size} image={img.size}')
    img.close()
    return 200, str(total_size), '' # code/size/note


def download_civit_model_thread(model_name: str, model_url: str, model_path: str = "", model_type: str = "Model", token: str = None):
    import hashlib
    sha256 = hashlib.sha256()
    sha256.update(model_url.encode('utf-8'))
    temp_file = sha256.hexdigest()[:8] + '.tmp'

    headers = {}
    starting_pos = 0
    if os.path.isfile(temp_file):
        starting_pos = os.path.getsize(temp_file)
        headers['Range'] = f'bytes={starting_pos}-'
    if token is None or len(token) == 0:
        token = shared.opts.civitai_token
    if token is not None and len(token) > 0:
        headers['Authorization'] = f'Bearer {token}'

    r = shared.req(model_url, headers=headers, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    if model_name is None or len(model_name) == 0:
        cn = r.headers.get('content-disposition', '')
        model_name = cn.split('filename=')[-1].strip('"')

    model_path = model_path.strip()
    if len(model_path) > 0:
        if os.path.isabs(model_path):
            pass
        else:
            model_path = os.path.join(paths.models_path, model_path)
    elif model_type.lower() == 'lora':
        model_path = shared.opts.lora_dir
    elif model_type.lower() == 'embedding':
        model_path = shared.opts.embeddings_dir
    elif model_type.lower() == 'vae':
        model_path = shared.opts.vae_dir
    else:
        model_path = shared.opts.ckpt_dir
    model_file = os.path.join(model_path, model_name)
    temp_file = os.path.join(model_path, temp_file)

    res = f'Model download: name="{model_name}" url="{model_url}" path="{model_path}" temp="{temp_file}"'
    if os.path.isfile(model_file):
        res += ' already exists'
        shared.log.warning(res)
        return res

    res += f' size={round((starting_pos + total_size)/1024/1024, 2)}Mb'
    shared.log.info(res)
    shared.state.begin('CivitAI')
    block_size = 16384 # 16KB blocks
    written = starting_pos
    global pbar # pylint: disable=global-statement
    if pbar is None:
        pbar = p.Progress(p.TextColumn('[cyan]{task.description}'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn(), p.TextColumn('[cyan]{task.fields[name]}'), console=shared.console)
    with pbar:
        task = pbar.add_task(description="Download starting", total=starting_pos+total_size, name=model_name)
        try:
            with open(temp_file, 'ab') as f:
                for data in r.iter_content(block_size):
                    if written == 0:
                        try: # check if response is JSON message instead of bytes
                            shared.log.error(f'Model download: response={json.loads(data.decode("utf-8"))}')
                            raise ValueError('response: type=json expected=bytes')
                        except Exception: # this is good
                            pass
                    written = written + len(data)
                    f.write(data)
                    pbar.update(task, description="Download", completed=written)
            if written < 1024: # min threshold
                os.remove(temp_file)
                raise ValueError(f'removed invalid download: bytes={written}')
        except Exception as e:
            shared.log.error(f'{res} {e}')
        finally:
            pbar.stop_task(task)
            pbar.remove_task(task)
    if starting_pos+total_size != written:
        shared.log.warning(f'{res} written={round(written/1024/1024)}Mb incomplete download')
    elif os.path.exists(temp_file):
        shared.log.debug(f'Model download complete: temp="{temp_file}" path="{model_file}"')
        os.rename(temp_file, model_file)
    shared.state.end()
    if os.path.exists(model_file):
        return model_file
    else:
        return None


def download_civit_model(model_url: str, model_name: str = '', model_path: str = '', model_type: str = '', token: str = None):
    import threading
    if model_url is None or len(model_url) == 0:
        shared.log.error('Model download: no url provided')
        return
    thread = threading.Thread(target=download_civit_model_thread, args=(model_name, model_url, model_path, model_type, token))
    thread.start()
    thread.join()
    from modules.sd_models import list_models  # pylint: disable=W0621
    list_models()
