import html
import json
import os
from modules import shared, ui_extra_networks, sd_vae, hashes, modelstats
from modules.logger import log


class ExtraNetworksPageVAEs(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('VAE')

    def refresh(self):
        shared.refresh_vaes()

    def list_items(self):
        for name, filename in sd_vae.vae_dict.items():
            try:
                size, mtime = modelstats.stat(filename)
                info = self.find_info(filename)
                version = self.find_version(None, info)
                record = {
                    "type": 'VAE',
                    "name": name,
                    "alias": os.path.splitext(os.path.basename(filename))[0],
                    "title": name,
                    "filename": filename,
                    "hash": hashes.sha256_from_cache(filename, f"vae/{filename}"),
                    "preview": self.find_preview(filename),
                    "local_preview": f"{os.path.splitext(filename)[0]}.{shared.opts.samples_format}",
                    "metadata": {},
                    "onclick": '"' + html.escape(f"""return selectVAE({json.dumps(name)})""") + '"',
                    "mtime": mtime,
                    "size": size,
                    "info": info,
                    "description": self.find_description(filename, info),
                    "version": version.get("baseModel", "N/A") if info else "N/A",
                }
                yield record
            except Exception as e:
                log.debug(f'Networks error: type=vae file="{filename}" {e}')

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.vae_dir] if v is not None]
