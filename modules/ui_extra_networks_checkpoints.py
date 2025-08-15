import os
import html
import json
import concurrent
from modules import shared, ui_extra_networks, sd_models, modelstats


reference_dir = os.path.join('models', 'Reference')

class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Model')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_reference(self): # pylint: disable=inconsistent-return-statements
        if not shared.opts.sd_checkpoint_autodownload or not shared.opts.extra_network_reference_enable:
            return []
        for k, v in shared.reference_models.items():
            url = v['path']
            experimental = v.get('experimental', False)
            if experimental:
                if shared.cmd_opts.experimental:
                    shared.log.debug(f'Networks: experimental model="{k}"')
                else:
                    continue
            preview = v.get('preview', v['path'])
            preview_file = self.find_preview_file(os.path.join(reference_dir, preview))
            _size, mtime = modelstats.stat(preview_file)
            yield {
                "type": 'Model',
                "name": os.path.join(reference_dir, k),
                "title": os.path.join(reference_dir, k),
                "filename": url,
                "preview": self.find_preview(os.path.join(reference_dir, preview)),
                "local_preview": preview_file,
                "onclick": '"' + html.escape(f"selectReference({json.dumps(url)})") + '"',
                "hash": None,
                "mtime": mtime,
                "size": 0,
                "info": {},
                "metadata": {},
                "description": v.get('desc', ''),
            }

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
            record["info"] = self.find_info(checkpoint.filename)
            record["description"] = self.find_description(checkpoint.filename, record["info"])
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
        return [v for v in [shared.opts.ckpt_dir, reference_dir, sd_models.model_path] if v is not None]
