import time
import json
import html
from modules import shared, ui_extra_networks


class ExtraNetworksPageHistory(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        # log.trace('History init')
        super().__init__('History')
        self.last_refresh = 0

    def refresh(self):
        # log.trace('History refresh')
        self.last_refresh = time.time()
        self.html = '<h1>buttons</h1>'
        for ts in shared.history.list:
            self.html += '<p>' + str(ts) + '</p>'

    def list_items(self):
        # log.trace('History list')
        for item in shared.history.latents:
            title = ', '.join(list(set(item.ops))) + '<br>' + item.name
            yield {
                "type": 'History',
                "name": title,
                "preview": item.preview,
                "mtime": item.ts,
                "size": item.size,
                # "info": item.info,
                # "description": item.info,
                "onclick": '"' + html.escape(f"""return selectHistory({json.dumps(item.name)})""") + '"',
            }

    def find_description(self, path, info=None):
        name = path.split('<br>')[-1]
        items = [l for l in shared.history.latents if l.name == name]
        if len(items) > 0:
            return items[0].info
        return ''
