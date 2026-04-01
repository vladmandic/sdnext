import os
import json
from modules import shared, ui_extra_networks, modelstats, files_cache
from modules.logger import log


wildcards_list = []


class ExtraNetworksPageWildcards(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Wildcards')

    def parents(self, file):
        folder = os.path.dirname(file)
        if folder != os.path.abspath(shared.opts.wildcards_dir) and folder not in wildcards_list:
            wildcards_list.append(folder)
            self.parents(folder)

    def refresh(self):
        wildcards_list.clear()
        files = files_cache.list_files(shared.opts.wildcards_dir, ext_filter=[".txt"], recursive=True)
        for file in files:
            wildcards_list.append(file)
            self.parents(file)

    def list_items(self):
        self.refresh()
        for filename in wildcards_list:
            relname = os.path.relpath(filename, shared.opts.wildcards_dir)
            name = os.path.splitext(relname)[0]
            size, mtime = modelstats.stat(filename)
            try:
                record = {
                    "type": 'Wildcard',
                    "name": name,
                    "filename": filename,
                    "preview": self.find_preview(filename),
                    "local_preview": f"{os.path.splitext(filename)[0]}.{shared.opts.samples_format}",
                    "prompt": json.dumps(f" __{name}__"),
                    "mtime": mtime,
                    "size": size,
                    "description": '',
                    "info": {},
                }
                yield record
            except Exception as e:
                log.debug(f'Networks error: type=wildcard file="{filename}" {e}')

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.wildcards_dir] if v is not None]
