import os
import html
import json
from datetime import datetime
from modules import shared, extra_networks, ui_extra_networks, styles


class ExtraNetworksPageStyles(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Style')

    def refresh(self):
        shared.prompt_styles.reload()

    def parse_desc(self, desc):
        lines = desc.strip().split("\n")
        params = { 'name': '', 'description': '', 'prompt': '', 'negative': '', 'extra': '', 'wildcards': ''}
        found = ''
        for line in lines:
            line = line.strip()
            if line.lower().startswith('name:'):
                found = 'name'
                params['name'] = line[5:].strip()
            elif line.lower().startswith('description:'):
                found = 'description'
                params['description'] = line[12:].strip()
            elif line.lower().startswith('prompt:'):
                found = 'prompt'
                params['prompt'] = line[7:].strip()
            elif line.lower().startswith('negative:'):
                found = 'negative'
                params['negative'] = line[9:].strip()
            elif line.lower().startswith('extra:'):
                found = 'extra'
                params['extra'] = line[6:].strip()
            elif line.lower().startswith('wildcards:'):
                found = 'wildcards'
                params['wildcards'] = line[10:].strip()
            elif found != '':
                params[found] += '\n' + line
        if params['name'] == '':
            return None
        if params['description'] == '':
            params['description'] = params['name']
        return params

    def create_style(self, params):
        from modules.images import FilenameGenerator
        from hashlib import sha256
        namegen = FilenameGenerator(p=None, seed=None, prompt=params.get('Prompt', ''), image=None, grid=False)
        name = namegen.prompt_words()
        sha = sha256(json.dumps(name).encode()).hexdigest()[0:8]
        fn = os.path.join(shared.opts.styles_dir, sha + '.json')
        item = {
            "type": 'Style',
            "name": name,
            "title": name,
            "filename": fn,
            "preview": self.find_preview(name),
            "description": params.get('Description', ''),
            "prompt": params.get('Prompt', ''),
            "negative": params.get('Negative prompt', ''),
            "extra": params.get('Extra', ''),
            "wildcards": params.get('Wildcards', ''),
            "local_preview": f"{name}.{shared.opts.samples_format}",
        }
        return item

    def create_item(self, k):
        item = None
        try:
            style = shared.prompt_styles.styles.get(k)
            fn = os.path.splitext(getattr(style, 'filename', ''))[0]
            name = getattr(style, 'name', '')
            if name == '':
                return item
            txt = f'Prompt: {getattr(style, "prompt", "")}'
            if len(getattr(style, 'negative_prompt', '')) > 0:
                txt += f'\nNegative: {style.negative_prompt}'
            item = {
                "type": 'Style',
                "name": name,
                "title": k,
                "alias": os.path.splitext(os.path.basename(style.filename))[0],
                "filename": style.filename,
                "preview": style.preview if getattr(style, 'preview', None) is not None and style.preview.startswith('data:') else None,
                "description": style.description if getattr(style, 'description', None) is not None and len(style.description) > 0 else txt,
                "prompt": getattr(style, 'prompt', ''),
                "negative": getattr(style, 'negative_prompt', ''),
                "extra": getattr(style, 'extra', ''),
                "wildcards": getattr(style, 'wildcards', ''),
                "local_preview": f"{fn}.{shared.opts.samples_format}",
                "onclick": '"' + html.escape(f"""return selectStyle({json.dumps(name)})""") + '"',
                "mtime": getattr(style, 'mtime', datetime.fromtimestamp(0)),
                "size": os.path.getsize(style.filename),
            }
        except Exception as e:
            shared.log.debug(f'Networks error: type=style file="{k}" {e}')
        return item

    def list_items(self):
        items = [self.create_item(k) for k in list(shared.prompt_styles.styles)]
        items = [item for item in items if item is not None]
        self.update_all_previews(items)
        return items

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.styles_dir] if v is not None] + ['html']


class ExtraNetworkStyles(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('style')
        self.indexes = {}

    def activate(self, p, params_list):
        for param in params_list:
            if len(param.items) > 0:
                style = None
                search = param.items[0]
                # style = shared.prompt_styles.find_style(param.items[0])
                match = [s for s in shared.prompt_styles.styles.values() if s.name == search]
                if len(match) > 0:
                    style = match[0]
                else:
                    match = [s for s in shared.prompt_styles.styles.values() if s.name.startswith(search)]
                    if len(match) > 0:
                        i = self.indexes.get(search, 0)
                        self.indexes[search] = (i + 1) % len(match)
                        style = match[self.indexes[search]]
                if style is not None:
                    p.styles.append(style.name)
                    p.prompts = [styles.merge_prompts(style.prompt, prompt) for prompt in p.prompts]
                    p.negative_prompts = [styles.merge_prompts(style.negative_prompt, prompt) for prompt in p.negative_prompts]
                    styles.apply_styles_to_extra(p, style)


    def deactivate(self, p):
        pass
