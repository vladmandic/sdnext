import os
from modules.paths import data_path
from modules.logger import log


files = [
    'cache.json',
    'metadata.json',
    'html/extensions.json',
    'html/previews.json',
    'html/upscalers.json',
    'html/reference.json',
    'html/themes.json',
    'html/reference-quant.json',
    'html/reference-distilled.json',
    'html/reference-community.json',
    'html/reference-cloud.json',
]


def migrate_data():
    for f in files:
        old_filename = os.path.join(data_path, f)
        new_filename = os.path.join(data_path, "data", os.path.basename(f))
        if os.path.exists(old_filename):
            if not os.path.exists(new_filename):
                log.info(f'Migrating: file="{old_filename}" target="{new_filename}"')
                try:
                    os.rename(old_filename, new_filename)
                except Exception as e:
                    log.error(f'Migrating: file="{old_filename}" target="{new_filename}" {e}')
            else:
                log.warning(f'Migrating: file="{old_filename}" target="{new_filename}" skip existing')


migrate_data()
