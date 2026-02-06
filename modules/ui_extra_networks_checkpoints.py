import os
import html
import json
import concurrent
from datetime import datetime
from modules import shared, ui_extra_networks, sd_models, modelstats, paths, devices
from modules.json_helpers import readfile


version_map = {
    "QwenEdit": "Qwen",
    "QwenEditPlus": "Qwen",
    "Flux.1 D": "Flux",
    "Flux.1 S": "Flux",
    "FluxKontext": "Flux",
    "SDXL 1.0": "SD XL",
    "SDXL Hyper": "SD XL",
    "StableDiffusion3": "SD 3",
    "StableDiffusionXL": "SD XL",
    "WanToVideo": "Wan",
    "WanVACE": "Wan",
    "Z": "Z-Image",
    "Glm": "GLM-Image",
}

class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Model')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_reference(self): # pylint: disable=inconsistent-return-statements
        existing = [model.filename if model.type == 'safetensors' else model.name for model in sd_models.checkpoints_list.values()]

        def reference_downloaded(url):
            url = url.split('@')[0] if '@' in url else 'Diffusers/' + url
            url = url.split('+')[0] if '+' in url else url
            return any(model.endswith(url) for model in existing)

        if not shared.opts.sd_checkpoint_autodownload or not shared.opts.extra_network_reference_enable:
            shared.log.debug(f'Networks: type="reference" autodownload={shared.opts.sd_checkpoint_autodownload} enable={shared.opts.extra_network_reference_enable}')
            return []
        count = { 'total': 0, 'ready': 0, 'hidden': 0, 'experimental': 0, 'base': 0 }

        reference_base = readfile(os.path.join('data', 'reference.json'), as_type="dict")
        reference_quant = readfile(os.path.join('data', 'reference-quant.json'), as_type="dict")
        reference_distilled = readfile(os.path.join('data', 'reference-distilled.json'), as_type="dict")
        reference_community = readfile(os.path.join('data', 'reference-community.json'), as_type="dict")
        reference_cloud = readfile(os.path.join('data', 'reference-cloud.json'), as_type="dict")
        reference_nunchaku = readfile(os.path.join('data', 'reference-nunchaku.json'), as_type="dict")
        shared.reference_models = {}
        shared.reference_models.update(reference_base)
        shared.reference_models.update(reference_quant)
        shared.reference_models.update(reference_community)
        shared.reference_models.update(reference_distilled)
        shared.reference_models.update(reference_cloud)
        shared.reference_models.update(reference_nunchaku)

        for k, v in shared.reference_models.items():
            count['total'] += 1
            url = v['path']
            if v.get('hidden', False):
                count['hidden'] += 1
                continue
            experimental = v.get('experimental', False)
            if experimental:
                if shared.cmd_opts.experimental:
                    shared.log.debug(f'Networks: experimental model="{k}"')
                    count['experimental'] += 1
                else:
                    continue
            preview = v.get('preview', v['path'])
            preview_file = self.find_preview_file(os.path.join(paths.reference_path, preview))
            name = os.path.normpath(os.path.join(paths.reference_path, k)).replace('\\', '/')
            size = int(float(v.get('size', 0)) * 1024 * 1024 * 1024)
            mtime = v.get('date', None)
            if mtime is None:
                _size, mtime = modelstats.stat(preview_file)
            else:
                try:
                    mtime = datetime.strptime(mtime, '%Y %B') # 2025 January
                except Exception:
                    _size, mtime = modelstats.stat(preview_file)
            if len(v.get("subfolder", "")) > 0:
                path = f'{v.get("path", "")}+{v.get("subfolder", "")}'
            else:
                path = f'{v.get("path", "")}'

            tag = v.get('tags', '')
            if tag == 'nunchaku' and devices.backend != 'cuda':
                count['hidden'] += 1
                continue
            if tag in count:
                count[tag] += 1
            elif tag != '':
                count[tag] = 1
            else:
                count['base'] += 1

            ready = reference_downloaded(url)
            version = "ready" if ready else "download"
            if tag == 'cloud':
                version = 'Cloud'
            if not ready and shared.opts.offline_mode:
                count['hidden'] += 1
                continue
            if ready:
                count['ready'] += 1

            yield {
                "type": 'Model',
                "name": name,
                "title": name,
                "filename": url,
                "preview": self.find_preview(os.path.join(paths.reference_path, preview)),
                "local_preview": preview_file,
                "onclick": '"' + html.escape(f"selectReference({json.dumps(path)})") + '"',
                "hash": None,
                "mtime": mtime,
                "size": size,
                "info": {},
                "metadata": {},
                "description": v.get('desc', ''),
                "version": version,
                "tags": tag,
            }
        shared.log.debug(f'Networks: type="reference" {count}')

    def create_item(self, name):
        record = None
        try:
            checkpoint: sd_models.CheckpointInfo = sd_models.checkpoints_list.get(name)
            size, mtime = modelstats.stat(checkpoint.filename)
            record = {
                "type": 'Model',
                "name": checkpoint.name,
                "title": checkpoint.title,
                "filename": checkpoint.filename,
                "hash": checkpoint.shorthash,
                "metadata": checkpoint.metadata,
                "onclick": '"' + html.escape(f"selectCheckpoint({json.dumps(name)})") + '"',
                "mtime": mtime,
                "size": size,
            }
            record['info'] = self.find_info(checkpoint.filename)
            record['description'] = self.find_description(checkpoint.filename, record['info'])
            version = self.find_version(checkpoint, record['info'])
            if 'baseModel' in version:
                record['version'] = version.get("baseModel", "")
            elif '_class_name' in record['info']:
                record['version'] = record['info'].get('_class_name', '').replace('Pipeline', '').replace('Image', '')
            else:
                record['version'] = ''
            record['version'] = version_map.get(record['version'], record['version'])

        except Exception as e:
            shared.log.debug(f'Networks error: type=model file="{name}" {e}')
        return record

    def list_items(self):
        items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, cp): cp for cp in list(sd_models.checkpoints_list.copy())}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    items.append(item)
        for record in self.list_reference():
            items.append(record)
        self.update_all_previews(items)
        return items

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.ckpt_dir, paths.reference_path, sd_models.model_path] if v is not None]
