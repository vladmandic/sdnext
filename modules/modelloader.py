import io
import os
import time
import shutil
import importlib
import contextlib
from typing import Dict
from urllib.parse import urlparse
import huggingface_hub as hf
from installer import install, log
from modules import shared, errors, files_cache
from modules.upscaler import Upscaler
from modules.paths import script_path, models_path


loggedin = None
diffuser_repos = []
debug = log.trace if os.environ.get('SD_DOWNLOAD_DEBUG', None) is not None else lambda *args, **kwargs: None
pbar = None


def hf_login(token=None):
    global loggedin # pylint: disable=global-statement
    token = token or shared.opts.huggingface_token
    install('hf_xet', quiet=True)
    if token is None or len(token) <= 4:
        log.debug('HF login: no token provided')
        return False
    if os.environ.get('HUGGING_FACE_HUB_TOKEN', None) is not None:
        os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
        os.unsetenv('HUGGING_FACE_HUB_TOKEN')
    if os.environ.get('HF_TOKEN', None) is not None:
        os.environ.pop('HF_TOKEN', None)
        os.unsetenv('HF_TOKEN')
    if loggedin != token:
        stdout = io.StringIO()
        try:
            hf.logout()
        except Exception:
            pass
        with contextlib.redirect_stdout(stdout):
            hf.login(token=token, add_to_git_credential=False, write_permission=False)
        os.environ['HF_TOKEN'] = token
        text = stdout.getvalue() or ''
        obfuscated_token = 'hf_...' + token[-4:]
        line = [l for l in text.split('\n') if 'Token' in l]
        log.info(f'HF login: token="{obfuscated_token}" fn="{hf.constants.HF_TOKEN_PATH}" {line[0] if len(line) > 0 else text}')
        loggedin = token
    return True


def download_diffusers_model(hub_id: str, cache_dir: str = None, download_config: Dict[str, str] = None, token = None, variant = None, revision = None, mirror = None, custom_pipeline = None):
    if hub_id is None or len(hub_id) == 0:
        return None
    from diffusers import DiffusionPipeline
    shared.state.begin('HuggingFace')
    if hub_id.startswith('huggingface/'):
        hub_id = hub_id.replace('huggingface/', '')
    if download_config is None:
        download_config = {
            "force_download": False,
            "resume_download": True,
            "cache_dir": shared.opts.diffusers_dir,
            "load_connected_pipeline": True,
        }
    if cache_dir is not None:
        download_config["cache_dir"] = cache_dir
    if variant is not None and len(variant) > 0:
        download_config["variant"] = variant
    if revision is not None and len(revision) > 0:
        download_config["revision"] = revision
    if mirror is not None and len(mirror) > 0:
        download_config["mirror"] = mirror
    if custom_pipeline is not None and len(custom_pipeline) > 0:
        download_config["custom_pipeline"] = custom_pipeline
    shared.log.debug(f'Diffusers downloading: id="{hub_id}" args={download_config}')
    token = token or shared.opts.huggingface_token
    if token is not None and len(token) > 2:
        hf_login(token)
    pipeline_dir = None

    ok = False
    err = None
    if not ok:
        try:
            pipeline_dir = DiffusionPipeline.download(hub_id, **download_config)
            ok = True
        except Exception as e:
            err = e
            ok = False
            debug(f'Diffusers download error: id="{hub_id}" {e}')
    if not ok and 'Repository Not Found' not in str(err):
        try:
            download_config.pop('load_connected_pipeline', None)
            download_config.pop('variant', None)
            pipeline_dir = hf.snapshot_download(hub_id, **download_config)
        except Exception as e:
            debug(f'Diffusers download error: id="{hub_id}" {e}')
            if 'gated' in str(e):
                shared.log.error(f'Diffusers download error: id="{hub_id}" model access requires login')
                return None
    if pipeline_dir is None:
        shared.log.error(f'Diffusers download error: id="{hub_id}" {err}')
        return None
    try:
        model_info_dict = hf.model_info(hub_id).cardData if pipeline_dir is not None else None
    except Exception:
        model_info_dict = None
    if model_info_dict is not None and "prior" in model_info_dict: # some checkpoints need to be downloaded as "hidden" as they just serve as pre- or post-pipelines of other pipelines
        download_dir = DiffusionPipeline.download(model_info_dict["prior"][0], **download_config)
        model_info_dict["prior"] = download_dir
        with open(os.path.join(download_dir, "hidden"), "w", encoding="utf-8") as f: # mark prior as hidden
            f.write("True")
    if pipeline_dir is not None:
        shared.writefile(model_info_dict, os.path.join(pipeline_dir, "model_info.json"))
    shared.state.end()
    return pipeline_dir


def load_diffusers_models(clear=True):
    # t0 = time.time()
    place = shared.opts.diffusers_dir
    if place is None or len(place) == 0 or not os.path.isdir(place):
        place = os.path.join(models_path, 'Diffusers')
    if clear:
        diffuser_repos.clear()
    already_found = []
    try:
        for folder in os.listdir(place):
            try:
                name = folder[8:] if folder.startswith('models--') else folder
                folder = os.path.join(place, folder)
                if name.endswith("-prior"):
                    continue
                if not os.path.isdir(folder):
                    continue
                name = name.replace("--", "/")
                friendly = os.path.join(place, name)
                has_index = os.path.exists(os.path.join(folder, 'model_index.json'))

                if has_index: # direct download of diffusers model
                    repo = { 'name': name, 'filename': name, 'friendly': friendly, 'folder': folder, 'path': folder, 'hash': None, 'mtime': os.path.getmtime(folder), 'model_info': os.path.join(folder, 'model_info.json'), 'model_index': os.path.join(folder, 'model_index.json') }
                    diffuser_repos.append(repo)
                    continue

                snapshots = os.listdir(os.path.join(folder, "snapshots"))
                if len(snapshots) == 0:
                    shared.log.warning(f'Diffusers folder has no snapshots: location="{place}" folder="{folder}" name="{name}"')
                    continue
                for snapshot in snapshots: # download using from_pretrained which uses huggingface_hub or huggingface_hub directly and creates snapshot-like structure
                    commit = os.path.join(folder, 'snapshots', snapshot)
                    mtime = os.path.getmtime(commit)
                    info = os.path.join(commit, "model_info.json")
                    index = os.path.join(commit, "model_index.json")
                    config = os.path.join(commit, "config.json")
                    if (not os.path.exists(index)) and (not os.path.exists(info)) and (not os.path.exists(config)):
                        debug(f'Diffusers skip model no info: {name}')
                        continue
                    if name in already_found:
                        debug(f'Diffusers skip model already found: {name}')
                        continue
                    repo = { 'name': name, 'filename': name, 'friendly': friendly, 'folder': folder, 'path': commit, 'hash': snapshot, 'mtime': mtime, 'model_info': info, 'model_index': index, 'model_config': config }
                    already_found.append(name)
                    diffuser_repos.append(repo)
                    if os.path.exists(os.path.join(folder, 'hidden')):
                        continue
            except Exception as e:
                debug(f'Error analyzing diffusers model: "{folder}" {e}')
    except Exception as e:
        shared.log.error(f"Error listing diffusers: {place} {e}")
    # shared.log.debug(f'Scanning diffusers cache: folder="{place}" items={len(list(diffuser_repos))} time={time.time()-t0:.2f}')
    return diffuser_repos


def find_diffuser(name: str, full=False):
    repo = [r for r in diffuser_repos if name == r['name'] or name == r['friendly'] or name == r['path']]
    if len(repo) > 0:
        return [repo[0]['name']]
    hf_api = hf.HfApi()
    models = list(hf_api.list_models(model_name=name, library=['diffusers'], full=True, limit=20, sort="downloads", direction=-1))
    if len(models) == 0:
        models = list(hf_api.list_models(model_name=name, full=True, limit=20, sort="downloads", direction=-1)) # widen search
    models = [m for m in models if m.id.startswith(name)] # filter exact
    shared.log.debug(f'Search model: repo="{name}" {len(models) > 0}')
    if len(models) > 0:
        if not full:
            return models[0].id
        else:
            return [m.id for m in models]
    return None


def get_reference_opts(name: str, quiet=False):
    model_opts = {}
    name = name.replace('Diffusers/', 'huggingface/')
    for k, v in shared.reference_models.items():
        model_name = v.get('path', '')
        if k == name or model_name == name:
            model_opts = v
            break
        model_name_split = os.path.splitext(model_name.split('@')[0])[0]
        if k == name or model_name_split == name:
            model_opts = v
            break
        model_name_replace = model_name.replace('huggingface/', '')
        if k == name or model_name_replace == name:
            model_opts = v
            break
    if not model_opts:
        # shared.log.error(f'Reference: model="{name}" not found')
        return {}
    if not quiet:
        shared.log.debug(f'Reference: model="{name}" {model_opts}')
    return model_opts


def load_reference(name: str, variant: str = None, revision: str = None, mirror: str = None, custom_pipeline: str = None):
    found = [r for r in diffuser_repos if name == r['name'] or name == r['friendly'] or name == r['path']]
    if len(found) > 0: # already downloaded
        model_opts = get_reference_opts(found[0]['name'])
        return True
    else:
        model_opts = get_reference_opts(name)
    if model_opts.get('skip', False):
        return True
    shared.log.debug(f'Reference: download="{name}"')
    model_dir = download_diffusers_model(
        hub_id=name,
        cache_dir=shared.opts.diffusers_dir,
        variant=variant or model_opts.get('variant', None),
        revision=revision or model_opts.get('revision', None),
        mirror=mirror or model_opts.get('mirror', None),
        custom_pipeline=custom_pipeline or model_opts.get('custom_pipeline', None)
    )
    if model_dir is None:
        shared.log.error(f'Reference download: model="{name}"')
        return False
    else:
        shared.log.debug(f'Reference download complete: model="{name}"')
        model_opts = get_reference_opts(name)
        from modules import sd_models
        sd_models.list_models()
        return True


def load_civitai(model: str, url: str):
    from modules import sd_models
    name, _ext = os.path.splitext(model)
    info = sd_models.get_closet_checkpoint_match(name)
    if info is not None:
        _model_opts = get_reference_opts(info.model_name)
        return name # already downloaded
    else:
        shared.log.debug(f'Reference download start: model="{name}"')
        from modules.civitai.download_civitai import download_civit_model_thread
        download_civit_model_thread(model_name=model, model_url=url, model_path='', model_type='safetensors', token=shared.opts.civitai_token)
        shared.log.debug(f'Reference download complete: model="{name}"')
        sd_models.list_models()
        info = sd_models.get_closet_checkpoint_match(name)
        if info is not None:
            shared.log.debug(f'Reference: model="{name}"')
            return name # already downloaded
        else:
            shared.log.error(f'Reference model="{name}" not found')
            return None


def download_url_to_file(url: str, dst: str):
    # based on torch.hub.download_url_to_file
    import uuid
    import tempfile
    from urllib.request import urlopen, Request
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    file_size = None
    req = Request(url, headers={"User-Agent": "sdnext"})
    u = urlopen(req) # pylint: disable=R1732
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length") # pylint: disable=R1732
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    dst = os.path.expanduser(dst)
    for _seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + '.' + uuid.uuid4().hex + '.partial'
        try:
            f = open(tmp_dst, 'w+b') # pylint: disable=R1732
        except FileExistsError:
            continue
        break
    else:
        shared.log.error('Error downloading: url={url} no usable temporary filename found')
        return
    try:
        with Progress(TextColumn('[cyan]{task.description}'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), TimeElapsedColumn(), console=shared.console) as progress:
            task = progress.add_task(description="Downloading", total=file_size)
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                progress.update(task, advance=len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_file_from_url(url: str, *, model_dir: str, progress: bool = True, file_name = None): # pylint: disable=unused-argument
    """Download a file from url into model_dir, using the file present if possible. Returns the path to the downloaded file."""
    if model_dir is None:
        shared.log.error('Download folder is none')
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        shared.log.info(f'Downloading: url="{url}" file="{cached_file}"')
        download_url_to_file(url, cached_file)
    if os.path.exists(cached_file):
        return cached_file
    else:
        return None


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.
    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    places = [x for x in list(set([model_path, command_path])) if x is not None] # noqa:C405
    output = []
    try:
        output:list = [*files_cache.list_files(*places, ext_filter=ext_filter, ext_blacklist=ext_blacklist)]
        if model_url is not None and len(output) == 0:
            if download_name is not None:
                dl = load_file_from_url(model_url, model_dir=places[0], progress=True, file_name=download_name)
                if dl is not None:
                    output.append(dl)
            else:
                output.append(model_url)
    except Exception as e:
        errors.display(e,f"Error listing models: {files_cache.unique_directories(places)}")
    return output


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path
    file = os.path.basename(file)
    model_name, _extension = os.path.splitext(file)
    return model_name


def friendly_fullname(file: str):
    if "http" in file:
        file = urlparse(file).path
    file = os.path.basename(file)
    return file


def cleanup_models():
    # This code could probably be more efficient if we used a tuple list or something to store the src/destinations
    # and then enumerate that, but this works for now. In the future, it'd be nice to just have every "model" scaler
    # somehow auto-register and just do these things...
    root_path = script_path
    src_path = models_path
    dest_path = os.path.join(models_path, "Stable-diffusion")
    # move_files(src_path, dest_path, ".ckpt")
    # move_files(src_path, dest_path, ".safetensors")
    src_path = os.path.join(root_path, "ESRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(models_path, "BSRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path, ".pth")
    src_path = os.path.join(root_path, "gfpgan")
    dest_path = os.path.join(models_path, "GFPGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SwinIR")
    dest_path = os.path.join(models_path, "SwinIR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "repositories/latent-diffusion/experiments/pretrained_models/")
    dest_path = os.path.join(models_path, "LDSR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SCUNet")
    dest_path = os.path.join(models_path, "SCUNet")
    move_files(src_path, dest_path)


def move_files(src_path: str, dest_path: str, ext_filter: str = None):
    try:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if os.path.exists(src_path):
            for file in os.listdir(src_path):
                fullpath = os.path.join(src_path, file)
                if os.path.isfile(fullpath):
                    if ext_filter is not None:
                        if ext_filter not in file:
                            continue
                    shared.log.warning(f"Moving {file} from {src_path} to {dest_path}.")
                    try:
                        shutil.move(fullpath, dest_path)
                    except Exception:
                        pass
            if len(os.listdir(src_path)) == 0:
                shared.log.info(f"Removing empty folder: {src_path}")
                shutil.rmtree(src_path, True)
    except Exception:
        pass


def load_upscalers():
    # We can only do this 'magic' method to dynamically load upscalers if they are referenced, so we'll try to import any _model.py files before looking in __subclasses__
    t0 = time.time()
    modules_dir = os.path.join(shared.script_path, "modules", "postprocess")
    for file in os.listdir(modules_dir):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.postprocess.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except Exception as e:
                shared.log.error(f'Error loading upscaler: {model_name} {e}')
    upscalers = []
    commandline_options = vars(shared.cmd_opts)
    # some of upscaler classes will not go away after reloading their modules, and we'll end up with two copies of those classes. The newest copy will always be the last in the list, so we go from end to beginning and ignore duplicates
    used_classes = {}
    for cls in reversed(Upscaler.__subclasses__()):
        classname = str(cls)
        if classname not in used_classes:
            used_classes[classname] = cls
    upscaler_types = []
    for cls in reversed(used_classes.values()):
        name = cls.__name__
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        commandline_model_path = commandline_options.get(cmd_name, None)
        scaler = cls(commandline_model_path)
        scaler.user_path = commandline_model_path
        scaler.model_download_path = commandline_model_path or scaler.model_path
        upscalers += scaler.scalers
        upscaler_types.append(name[8:])
    shared.sd_upscalers = upscalers
    t1 = time.time()
    shared.log.info(f"Available Upscalers: items={len(shared.sd_upscalers)} downloaded={len([x for x in shared.sd_upscalers if x.data_path is not None and os.path.isfile(x.data_path)])} user={len([x for x in shared.sd_upscalers if x.custom])} time={t1-t0:.2f} types={upscaler_types}")
    return [x.name for x in shared.sd_upscalers]
