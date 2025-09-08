import os
import json
import concurrent
from modules import shared, ui_extra_networks, modelstats
from modules.lora import lora_load


debug = os.environ.get('SD_LORA_DEBUG', None) is not None


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')
        self.list_time = 0

    def refresh(self):
        lora_load.list_available_networks()

    @staticmethod
    def get_tags(l, info):
        tags = {}
        try:
            if l.metadata is not None:
                modelspec_tags = l.metadata.get('modelspec.tags', {})
                possible_tags = l.metadata.get('ss_tag_frequency', {}) # tags from model metedata
                if isinstance(possible_tags, str):
                    possible_tags = {}
                if isinstance(modelspec_tags, str):
                    modelspec_tags = {}
                if len(list(modelspec_tags)) > 0:
                    possible_tags.update(modelspec_tags)
                for k, v in possible_tags.items():
                    words = k.split('_', 1) if '_' in k else [v, k]
                    words = [str(w).replace('.json', '') for w in words]
                    if words[0] == '{}':
                        words[0] = 0
                    tag = ' '.join(words[1:]).lower()
                    tags[tag] = words[0]

            def find_version():
                found_versions = []
                current_hash = l.hash[:8].upper()
                all_versions = info.get('modelVersions', [])
                for v in info.get('modelVersions', []):
                    for f in v.get('files', []):
                        if any(h.startswith(current_hash) for h in f.get('hashes', {}).values()):
                            found_versions.append(v)
                if len(found_versions) == 0:
                    found_versions = all_versions
                return found_versions

            for v in find_version():  # trigger words from info json
                possible_tags = v.get('trainedWords', [])
                if isinstance(possible_tags, list):
                    for tag_str in possible_tags:
                        for tag in tag_str.split(','):
                            tag = tag.strip().lower()
                            if tag not in tags:
                                tags[tag] = 0

            possible_tags = info.get('tags', []) # tags from info json
            if not isinstance(possible_tags, list):
                possible_tags = list(possible_tags.values())
            for tag in possible_tags:
                tag = tag.strip().lower()
                if tag not in tags:
                    tags[tag] = 0
        except Exception:
            pass
        bad_chars = [';', ':', '<', ">", "*", '?', '\'', '\"', '(', ')', '[', ']', '{', '}', '\\', '/']
        clean_tags = {}
        for k, v in tags.items():
            tag = ''.join(i for i in k if i not in bad_chars).strip()
            clean_tags[tag] = v

        clean_tags.pop('img', None)
        clean_tags.pop('dataset', None)
        return clean_tags

    def create_item(self, name):
        l = lora_load.available_networks.get(name)
        if l is None:
            shared.log.warning(f'Networks: type=lora registered={len(list(lora_load.available_networks))} file="{name}" not registered')
            return None
        try:
            # path, _ext = os.path.splitext(l.filename)
            name = os.path.splitext(os.path.relpath(l.filename, shared.cmd_opts.lora_dir))[0]
            size, mtime = modelstats.stat(l.filename)
            info = self.find_info(l.filename)
            item = {
                "type": 'Lora',
                "name": name,
                "alias": os.path.splitext(os.path.basename(l.filename))[0],
                "filename": l.filename,
                "hash": l.shorthash,
                "prompt": json.dumps(f" <lora:{l.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"),
                "metadata": json.dumps(l.metadata, indent=4) if l.metadata else None,
                "mtime": mtime,
                "size": size,
                "version": l.sd_version,
                "info": info,
                "description": self.find_description(l.filename, info),
                "tags": self.get_tags(l, info),
            }
            return item
        except Exception as e:
            shared.log.error(f'Networks: type=lora file="{name}" {e}')
            if debug:
                from modules import errors
                errors.display(e, 'Lora')
            return None

    def list_items(self):
        items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, net): net for net in lora_load.available_networks}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    items.append(item)
        self.update_all_previews(items)
        return items

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]
