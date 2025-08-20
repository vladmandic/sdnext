import io
import base64
import os
import re
import time
import json
import collections
from PIL import Image
from modules import shared, paths, modelloader, hashes, sd_hijack_accelerate


checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0


class CheckpointInfo:
    def __init__(self, filename, sha=None):
        self.name = None
        self.hash = sha
        self.filename = filename
        self.type = ''
        relname = filename
        app_path = os.path.abspath(paths.script_path)

        def rel(fn, path):
            try:
                return os.path.relpath(fn, path)
            except Exception:
                return fn

        if relname.startswith('..'):
            relname = os.path.abspath(relname)
        if relname.startswith(shared.opts.ckpt_dir):
            relname = rel(filename, shared.opts.ckpt_dir)
        elif relname.startswith(shared.opts.diffusers_dir):
            relname = rel(filename, shared.opts.diffusers_dir)
        elif relname.startswith(model_path):
            relname = rel(filename, model_path)
        elif relname.startswith(paths.script_path):
            relname = rel(filename, paths.script_path)
        elif relname.startswith(app_path):
            relname = rel(filename, app_path)
        else:
            relname = os.path.abspath(relname)
        relname, ext = os.path.splitext(relname)
        ext = ext.lower()[1:]

        if filename.lower() == 'none':
            self.name = 'none'
            self.relname = 'none'
            self.sha256 = None
            self.type = 'unknown'
        elif os.path.isfile(filename): # ckpt or safetensor
            self.name = relname
            self.filename = filename
            self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{relname}")
            self.type = ext
            if 'nf4' in filename:
                self.type = 'transformer'
        else: # maybe a diffuser
            if self.hash is None:
                repo = [r for r in modelloader.diffuser_repos if self.filename == r['name']]
            else:
                repo = [r for r in modelloader.diffuser_repos if self.hash == r['hash']]
            if len(repo) == 0:
                self.name = filename
                self.filename = filename
                self.sha256 = None
                self.type = 'unknown'
            else:
                self.name = os.path.join(os.path.basename(shared.opts.diffusers_dir), repo[0]['name'])
                self.filename = repo[0]['path']
                self.sha256 = repo[0]['hash']
                self.type = 'diffusers'

        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'
        self.path = self.filename
        self.model_name = os.path.basename(self.name)
        self.metadata = read_metadata_from_safetensors(filename)
        # shared.log.debug(f'Checkpoint: type={self.type} name={self.name} filename={self.filename} hash={self.shorthash} title={self.title}')

    def register(self):
        checkpoints_list[self.title] = self
        for i in [self.name, self.filename, self.shorthash, self.title]:
            if i is not None:
                checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None
        self.shorthash = self.sha256[0:10]
        if self.title in checkpoints_list:
            checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()
        return self.shorthash

    def __str__(self):
        return f'checkpoint: type={self.type} title="{self.title}" path="{self.path}"'


def setup_model():
    list_models()
    sd_hijack_accelerate.hijack_hfhub()
    # sd_hijack_accelerate.hijack_torch_conv()


def checkpoint_titles():
    def convert(name):
        return int(name) if name.isdigit() else name.lower()
    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    t0 = time.time()
    global checkpoints_list # pylint: disable=global-statement
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    ext_filter = [".safetensors"]
    model_list = list(modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=ext_filter, download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"]))
    safetensors_list = []
    for filename in sorted(model_list, key=str.lower):
        checkpoint_info = CheckpointInfo(filename)
        safetensors_list.append(checkpoint_info)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
    diffusers_list = []
    for repo in modelloader.load_diffusers_models(clear=True):
        checkpoint_info = CheckpointInfo(repo['name'], sha=repo['hash'])
        diffusers_list.append(checkpoint_info)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
    if shared.cmd_opts.ckpt is not None:
        checkpoint_info = CheckpointInfo(shared.cmd_opts.ckpt)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
            shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif shared.cmd_opts.ckpt != shared.default_sd_model_file and shared.cmd_opts.ckpt is not None:
        shared.log.warning(f'Load model: path="{shared.cmd_opts.ckpt}" not found')
    shared.log.info(f'Available Models: safetensors="{shared.opts.ckpt_dir}":{len(safetensors_list)} diffusers="{shared.opts.diffusers_dir}":{len(diffusers_list)} reference={len(list(shared.reference_models))} items={len(checkpoints_list)} time={time.time()-t0:.2f}')
    checkpoints_list = dict(sorted(checkpoints_list.items(), key=lambda cp: cp[1].filename))


def update_model_hashes():
    def update_model_hashes_table(rows):
        html = """
            <table class="simple-table">
                <thead>
                    <tr><th>Name</th><th>Type</th><th>Hash</th></tr>
                </thead>
                <tbody>
                    {tbody}
                </tbody>
            </table>
        """
        tbody = ''
        for row in rows:
            try:
                tbody += f"""
                    <tr>
                        <td>{row.name}</td>
                        <td>{row.type}</td>
                        <td>{row.shorthash}</td>
                    </tr>
                """
            except Exception as e:
                shared.log.error(f'Model list: row={row} {e}')
        return html.format(tbody=tbody)

    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.hash is None]
    for ckpt in lst:
        ckpt.hash = model_hash(ckpt.filename)
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.sha256 is None or ckpt.shorthash is None]
    shared.log.info(f'Models list: hash missing={len(lst)} total={len(checkpoints_list)}')
    updated = []
    for ckpt in lst:
        ckpt.sha256 = hashes.sha256(ckpt.filename, f"checkpoint/{ckpt.name}")
        ckpt.shorthash = ckpt.sha256[0:10] if ckpt.sha256 is not None else None
        updated.append(ckpt)
        yield update_model_hashes_table(updated)


def remove_hash(s):
    return re.sub(r'\s*\[.*?\]', '', s)


def get_closet_checkpoint_match(s: str) -> CheckpointInfo:
    if s.startswith('https://huggingface.co/'):
        model_name = s.replace('https://huggingface.co/', '')
        checkpoint_info = CheckpointInfo(model_name) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info

    if s.startswith('huggingface/'):
        model_name = s.replace('huggingface/', '')
        checkpoint_info = CheckpointInfo(model_name) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info

    # alias search
    checkpoint_info = checkpoint_aliases.get(s, None)
    if checkpoint_info is not None:
        return checkpoint_info

    # models search
    found = sorted([info for info in checkpoints_list.values() if os.path.basename(info.title).lower().startswith(s.lower())], key=lambda x: len(x.title))
    if found and len(found) == 1:
        return found[0]

    # nohash search
    nohash = remove_hash(s)
    found = sorted([info for info in checkpoints_list.values() if info.title.lower().startswith(nohash.lower())], key=lambda x: len(x.title))
    if found and len(found) == 1:
        return found[0]

    # absolute path
    if s.endswith('.safetensors') and os.path.isfile(s):
        checkpoint_info = CheckpointInfo(s)
        return checkpoint_info

    # huggingface search
    if shared.opts.sd_checkpoint_autodownload and s.count('/') == 1:
        modelloader.hf_login()
        found = modelloader.find_diffuser(s, full=True)
        if found is None:
            return None
        found = [f for f in found if f == s]
        shared.log.info(f'HF search: model="{s}" results={found}')
        if found is not None and len(found) == 1:
            checkpoint_info = CheckpointInfo(s)
            checkpoint_info.type = 'huggingface'
            return checkpoint_info

    # civitai search
    if shared.opts.sd_checkpoint_autodownload and s.startswith("https://civitai.com/api/download/models"):
        from modules.civitai.download_civitai import download_civit_model_thread
        fn = download_civit_model_thread(model_name=None, model_url=s, model_path='', model_type='Model', token=None)
        if fn is not None:
            checkpoint_info = CheckpointInfo(fn)
            return checkpoint_info

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            shorthash = m.hexdigest()[0:8]
            return shorthash
    except FileNotFoundError:
        return 'NOFILE'
    except Exception:
        return 'NOHASH'


def select_checkpoint(op='model', sd_model_checkpoint=None):
    model_checkpoint = sd_model_checkpoint or (shared.opts.data.get('sd_model_refiner', None) if op == 'refiner' else shared.opts.data.get('sd_model_checkpoint', None))
    if model_checkpoint is None or model_checkpoint == 'None' or len(model_checkpoint) < 3:
        return None
    checkpoint_info = get_closet_checkpoint_match(model_checkpoint)
    if checkpoint_info is not None:
        shared.log.info(f'Load {op}: select="{checkpoint_info.title if checkpoint_info is not None else None}"')
        return checkpoint_info
    if len(checkpoints_list) == 0:
        shared.log.error("No models found")
        shared.log.info("Set system paths to use existing folders")
        shared.log.info("  or use --models-dir <path-to-folder> to specify base folder with all models")
        shared.log.info("  or use --ckpt <path-to-checkpoint> to force using specific model")
        return None
    if model_checkpoint is not None:
        if model_checkpoint != 'model.safetensors' and model_checkpoint != 'stabilityai/stable-diffusion-xl-base-1.0':
            shared.log.info(f'Load {op}: search="{model_checkpoint}" not found')
        else:
            shared.log.info("Selecting first available checkpoint")
    else:
        shared.log.info(f'Load {op}: select="{checkpoint_info.title if checkpoint_info is not None else None}"')
    return checkpoint_info


def init_metadata():
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        sd_metadata = shared.readfile(sd_metadata_file, lock=True) if os.path.isfile(sd_metadata_file) else {}


def extract_thumbnail(filename, data):
    try:
        thumbnail = data.split(",")[1]
        thumbnail = base64.b64decode(thumbnail)
        thumbnail = io.BytesIO(thumbnail)
        thumbnail = Image.open(thumbnail)
        thumbnail = thumbnail.convert("RGB")
        thumbnail = thumbnail.resize((512, 512), Image.Resampling.HAMMING)
        fn = os.path.splitext(filename)[0]
        thumbnail = thumbnail.save(f"{fn}.thumb.jpg", quality=50)
    except Exception as e:
        shared.log.error(f"Error extracting thumbnail: {filename} {e}")


def read_metadata_from_safetensors(filename):
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        sd_metadata = shared.readfile(sd_metadata_file, lock=True) if os.path.isfile(sd_metadata_file) else {}
    res = sd_metadata.get(filename, None)
    if res is not None:
        return res
    if not filename.endswith(".safetensors"):
        return {}
    if shared.cmd_opts.no_metadata:
        return {}
    res = {}
    # try:
    t0 = time.time()
    with open(filename, mode="rb") as file:
        try:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)
            if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
                shared.log.error(f'Model metadata invalid: file="{filename}" len={metadata_len} start={json_start}')
                return res
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                if k == 'modelspec.thumbnail' and v.startswith("data:"):
                    extract_thumbnail(filename, v)
                if v.startswith("data:"):
                    v = 'data'
                if k == 'format' and v == 'pt':
                    continue
                large = True if len(v) > 2048 else False
                if large and k in ['ss_datasets', 'workflow', 'prompt', 'ss_bucket_info', 'sd_metadata_file']:
                    continue
                if v[0:1] == '{':
                    try:
                        v = json.loads(v)
                        if large and k == 'ss_tag_frequency':
                            v = { i: len(j) for i, j in v.items() }
                        if large and k == 'sd_merge_models':
                            scrub_dict(v, ['sd_merge_recipe'])
                    except Exception:
                        pass
                res[k] = v
        except Exception as e:
            shared.log.error(f'Model metadata: file="{filename}" {e}')
            from modules import errors
            errors.display(e, 'Model metadata')
    sd_metadata[filename] = res
    global sd_metadata_pending # pylint: disable=global-statement
    sd_metadata_pending += 1
    t1 = time.time()
    global sd_metadata_timer # pylint: disable=global-statement
    sd_metadata_timer += (t1 - t0)
    return res


def scrub_dict(dict_obj, keys):
    for key in list(dict_obj.keys()):
        if not isinstance(dict_obj, dict):
            continue
        if key in keys:
            dict_obj.pop(key, None)
        elif isinstance(dict_obj[key], dict):
            scrub_dict(dict_obj[key], keys)
        elif isinstance(dict_obj[key], list):
            for item in dict_obj[key]:
                scrub_dict(item, keys)


def write_metadata():
    global sd_metadata_pending # pylint: disable=global-statement
    if sd_metadata_pending == 0:
        shared.log.debug(f'Model metadata: file="{sd_metadata_file}" no changes')
        return
    shared.writefile(sd_metadata, sd_metadata_file)
    shared.log.info(f'Model metadata saved: file="{sd_metadata_file}" items={sd_metadata_pending} time={sd_metadata_timer:.2f}')
    sd_metadata_pending = 0
