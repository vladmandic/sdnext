import html
import json
import os
from modules import shared, ui_extra_networks, sd_unet, hashes, modelstats
from modules.logger import log


class ExtraNetworksPageUNets(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('UNet/DiT')

    def refresh(self):
        return sd_unet.refresh_unet_list()

    def list_items(self):
        for name, filename in sd_unet.unet_dict.items():
            try:
                size, mtime = modelstats.stat(filename)
                info = self.find_info(filename)
                version = self.find_version(None, info)
                record = {
                    "type": 'UNet/DiT',
                    "name": name,
                    "alias": os.path.splitext(os.path.basename(filename))[0],
                    "title": name,
                    "filename": filename,
                    "hash": hashes.sha256_from_cache(filename, f"unet/{name}"),
                    "preview": self.find_preview(filename),
                    "local_preview": f"{os.path.splitext(filename)[0]}.{shared.opts.samples_format}",
                    "metadata": {},
                    "onclick": '"' + html.escape(f"""return selectUNet({json.dumps(name)})""") + '"',
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
        return [v for v in [shared.opts.unet_dir] if v is not None]
