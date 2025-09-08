# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations
import re
import os
import csv
import json
import time
import random
from modules import files_cache, shared, infotext, sd_models, sd_vae


debug_enabled = os.environ.get('SD_STYLES_DEBUG', None) is not None


class Style():
    def __init__(self, name: str, desc: str = "", prompt: str = "", negative_prompt: str = "", extra: str = "", wildcards: str = "", filename: str = "", preview: str = "", mtime: float = 0):
        self.name = name
        self.description = desc
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.extra = extra
        self.wildcards = wildcards
        self.filename = filename
        self.preview = preview
        self.mtime = mtime


def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        original_prompt = prompt.strip()
        style_prompt = style_prompt.strip()
        parts = filter(None, (original_prompt, style_prompt))
        if original_prompt.endswith(","):
            res = " ".join(parts)
        else:
            res = ", ".join(parts)
    return res


def apply_styles_to_prompt(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)
    return prompt


def apply_file_wildcards(prompt, replaced = [], not_found = [], recursion=0, seed=-1):
    def check_wildcard_files(prompt, wildcard, files, file_only=True):
        trimmed = wildcard.replace('\\', os.path.sep).strip().lower()
        for file in files:
            if file_only:
                paths = [os.path.splitext(file)[0].lower(), os.path.splitext(os.path.basename(file).lower())[0]] # fullname and basename
            else:
                paths = [os.path.splitext(p.lower())[0] for p in os.path.normpath(file).split(os.path.sep)] # every path component
            if (trimmed in paths) or (os.path.sep in trimmed and trimmed in paths[0]):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            choice = random.choice(lines).strip(' \n')
                            if '|' in choice:
                                choice = random.choice(choice.split('|')).strip(' []{}\n')
                            prompt = prompt.replace(f"__{wildcard}__", choice, 1)
                            shared.log.debug(f'Apply wildcard: select="{wildcard}" choice="{choice}" file="{file}" choices={len(lines)}')
                            replaced.append(wildcard)
                            return prompt, True
                except Exception as e:
                    shared.log.error(f'Wildcards: wildcard={wildcard} file={file} {e}')
        if not file_only:
            return prompt, False
        return check_wildcard_files(prompt, wildcard, files, file_only=False)

    def get_wildcards(prompt):
        matches = re.findall(r'__(.*?)__', prompt, re.DOTALL)
        matches = [m for m in matches if m not in not_found]
        # matches = [m for m in matches if m not in replaced]
        return matches

    recursion += 1
    if not shared.opts.wildcards_enabled or recursion >= 10 or not isinstance(prompt, str) or len(prompt) == 0:
        return prompt, replaced, not_found
    wildcards = get_wildcards(prompt)
    if len(wildcards) == 0:
        return prompt, replaced, not_found
    files = list(files_cache.list_files(shared.opts.wildcards_dir, ext_filter=[".txt"], recursive=True))
    if len(files) == 0:
        return prompt, replaced, not_found
    for wildcard in wildcards:
        prompt, found = check_wildcard_files(prompt, wildcard, files)
        if found and wildcard in not_found:
            not_found.remove(wildcard)
        elif not found and wildcard not in not_found:
            not_found.append(wildcard)
    prompt, replaced, not_found = apply_file_wildcards(prompt, replaced, not_found, recursion, seed) # recursive until we get early return
    return prompt, replaced, not_found


def apply_wildcards_to_prompt(prompt, all_wildcards, seed=-1, silent=False):
    if prompt is None or len(prompt) == 0:
        return prompt
    old_state = None
    if seed > 0 and len(all_wildcards) > 0:
        random.seed(seed)
        old_state = random.getstate()
    replaced = {}
    t0 = time.time()
    for style_wildcards in all_wildcards:
        wildcards = [x.strip() for x in style_wildcards.replace('\n', ' ').split(";") if len(x.strip()) > 0]
        for wildcard in wildcards:
            try:
                what, words = wildcard.split("=", 1)
                if what in prompt:
                    words = [x.strip() for x in words.split(",") if len(x.strip()) > 0]
                    word = random.choice(words)
                    prompt = prompt.replace(what, word)
                    replaced[what] = word
            except Exception as e:
                shared.log.error(f'Wildcards: wildcard="{wildcard}" error={e}')
    t1 = time.time()
    prompt, replaced_file, not_found = apply_file_wildcards(prompt, [], [], recursion=0, seed=seed)
    t2 = time.time()
    if replaced and not silent:
        shared.log.debug(f'Apply wildcards: {replaced} path="{shared.opts.wildcards_dir}" type=style time={t1-t0:.2f}')
    if (len(replaced_file) > 0 or len(not_found) > 0) and not silent:
        shared.log.debug(f'Apply wildcards: found={replaced_file} missing={not_found} path="{shared.opts.wildcards_dir}" type=file time={t2-t2:.2f} ')
    if old_state is not None:
        random.setstate(old_state)
    return prompt


def get_reference_style():
    if getattr(shared.sd_model, 'sd_checkpoint_info', None) is None:
        return None
    name = shared.sd_model.sd_checkpoint_info.name
    name = name.replace('\\', '/').replace('Diffusers/', '')
    for k, v in shared.reference_models.items():
        model_file = os.path.splitext(v.get('path', '').split('@')[0])[0].replace('huggingface/', '')
        if k == name or model_file == name:
            return v.get('extras', None)
    return None


def apply_styles_to_extra(p, style: Style):
    if style is None:
        return
    name_map = {
        'sampler': 'sampler_name',
        'size-1': 'width',
        'size-2': 'height',
        'model': 'sd_model_checkpoint',
        'vae': 'sd_vae',
        'unet': 'sd_unet',
        'te': 'sd_text_encoder',
        'refine': 'enable_hr',
        'hires': 'hr_force',
    }
    name_exclude = [
        'size',
    ]
    reference_style = get_reference_style()
    extra = infotext.parse(reference_style) if shared.opts.extra_network_reference_values else {}
    style_extra = apply_wildcards_to_prompt(style.extra, [style.wildcards], silent=True)
    style_extra = ' ' + style_extra.lower()
    extra.update(infotext.parse(style_extra))
    extra.pop('Prompt', None)
    extra.pop('Negative prompt', None)
    params = []
    settings = []
    skipped = []
    for k, v in extra.items():
        k = k.lower().replace(' ', '_')
        if k in name_map: # rename some fields
            k = name_map[k]
        if k in name_exclude: # exclude some fields
            continue
        if hasattr(p, k):
            orig = getattr(p, k)
            if (type(orig) != type(v)) and (orig is not None):
                if not (type(orig) == int and type(v) == float): # dont convert float to int
                    v = type(orig)(v)
            setattr(p, k, v)
            if debug_enabled:
                shared.log.trace(f'Apply style param: {k}={v}')
            params.append(f'{k}={v}')
        elif shared.opts.data_labels.get(k, None) is not None:
            if debug_enabled:
                shared.log.trace(f'Apply style setting: {k}={v}')
            shared.opts.data[k] = v
            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()
            if k == 'sd_vae':
                sd_vae.reload_vae_weights()
            settings.append(f'{k}={v}')
        else:
            if debug_enabled:
                shared.log.trace(f'Apply style skip: {k}={v}')
            skipped.append(f'{k}={v}')
    shared.log.debug(f'Apply style: name="{style.name}" params={params} settings={settings} unknown={skipped} reference={True if reference_style else False}')


class StyleDatabase:
    def __init__(self, opts):
        from modules import paths

        self.no_style = Style("None")
        self.styles = {}
        self.path = opts.styles_dir
        self.built_in = opts.extra_networks_styles
        if os.path.isfile(opts.styles_dir) or opts.styles_dir.endswith(".csv"):
            legacy_file = opts.styles_dir
            self.load_csv(legacy_file)
            opts.styles_dir = os.path.join(paths.models_path, "styles")
            self.path = opts.styles_dir
            try:
                os.makedirs(opts.styles_dir, exist_ok=True)
                self.save_styles(opts.styles_dir, verbose=True)
                shared.log.debug(f'Migrated styles: file="{legacy_file}" folder="{opts.styles_dir}"')
                self.reload()
            except Exception as e:
                shared.log.error(f'styles failed to migrate: file="{legacy_file}" error={e}')
        if not os.path.isdir(opts.styles_dir):
            opts.styles_dir = os.path.join(paths.models_path, "styles")
            self.path = opts.styles_dir
            try:
                os.makedirs(opts.styles_dir, exist_ok=True)
            except Exception:
                pass

    def load_style(self, fn, prefix=None):
        with open(fn, 'r', encoding='utf-8') as f:
            new_style = None
            try:
                all_styles = json.load(f)
                if type(all_styles) is dict:
                    all_styles = [all_styles]
                for style in all_styles:
                    if type(style) is not dict or "name" not in style:
                        raise ValueError('cannot parse style')
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    name = re.sub(r'[\t\r\n]', '', style.get("name", basename)).strip()
                    if prefix is not None:
                        name = os.path.join(prefix, name)
                    else:
                        name = os.path.join(os.path.dirname(os.path.relpath(fn, self.path)), name)
                    new_style = Style(
                        name=name,
                        desc=style.get('description', name),
                        prompt=style.get("prompt", ""),
                        negative_prompt=style.get("negative", ""),
                        extra=style.get("extra", ""),
                        wildcards=style.get("wildcards", ""),
                        preview=style.get("preview", None),
                        filename=fn,
                        mtime=os.path.getmtime(fn),
                    )
                    self.styles[style["name"]] = new_style
            except Exception as e:
                shared.log.error(f'Failed to load style: file="{fn}" error={e}')
            return new_style


    def reload(self):
        t0 = time.time()
        self.styles.clear()

        def list_folder(folder):
            import concurrent
            future_items = {}
            candidates = list(files_cache.list_files(folder, ext_filter=['.json'], recursive=files_cache.not_hidden))
            with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
                for fn in candidates:
                    if os.path.isfile(fn) and fn.lower().endswith(".json"):
                        future_items[executor.submit(self.load_style, fn, None)] = fn
                        # self.load_style(fn)
                    elif os.path.isdir(fn) and not fn.startswith('.'):
                        list_folder(fn)
                self.styles = dict(sorted(self.styles.items(), key=lambda style: style[1].filename))
                if self.built_in:
                    fn = os.path.join('html', 'art-styles.json')
                    future_items[executor.submit(self.load_style, fn, 'Reference')] = fn
                for future in concurrent.futures.as_completed(future_items):
                    future.result()

        self.built_in = shared.opts.extra_networks_styles
        list_folder(self.path)
        t1 = time.time()
        shared.log.info(f'Available Styles: path="{self.path}" items={len(self.styles.keys())} time={t1-t0:.2f}')

    def find_style(self, name):
        found = [style for style in self.styles.values() if style.name == name]
        return found[0] if len(found) > 0 else self.no_style

    def get_style_prompts(self, styles):
        if styles is None:
            return []
        if not isinstance(styles, list):
            shared.log.error(f'Styles invalid: {styles}')
            return []
        return [self.find_style(x).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        if styles is None:
            return []
        if not isinstance(styles, list):
            shared.log.error(f'Styles invalid: {styles}')
            return []
        return [self.find_style(x).negative_prompt for x in styles]

    def apply_styles_to_prompts(self, prompts, negatives, styles, seeds):
        if styles is None:
            return prompts, negatives
        if not isinstance(styles, list):
            shared.log.error(f'Styles invalid styles: {styles}')
            return prompts, negatives
        if prompts is None or not isinstance(prompts, list):
            shared.log.error(f'Styles invalid prompts: {prompts}')
            return prompts, negatives
        if seeds is None or not isinstance(prompts, list):
            shared.log.error(f'Styles invalid seeds: {seeds}')
            return prompts, negatives
        parsed_positive = []
        parsed_negative = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            prompt = apply_styles_to_prompt(prompt, [self.find_style(x).prompt for x in styles])
            prompt = apply_wildcards_to_prompt(prompt, [self.find_style(x).wildcards for x in styles], seeds[i])
            parsed_positive.append(prompt)
        for i in range(len(negatives)):
            prompt = negatives[i]
            prompt = apply_styles_to_prompt(prompt, [self.find_style(x).negative_prompt for x in styles])
            prompt = apply_wildcards_to_prompt(prompt, [self.find_style(x).wildcards for x in styles], seeds[i])
            parsed_negative.append(prompt)
        return parsed_positive, parsed_negative

    def apply_styles_to_prompt(self, prompt, styles, wildcards:bool=True):
        if styles is None:
            return prompt
        if not isinstance(styles, list):
            shared.log.error(f'Styles invalid: {styles}')
            return prompt
        prompt = apply_styles_to_prompt(prompt, [self.find_style(x).prompt for x in styles])
        if wildcards:
            prompt = apply_wildcards_to_prompt(prompt, [self.find_style(x).wildcards for x in styles])
        return prompt

    def apply_negative_styles_to_prompt(self, prompt, styles, wildcards:bool=True):
        if styles is None:
            return prompt
        if not isinstance(styles, list):
            shared.log.error(f'Styles invalid: {styles}')
            return prompt
        prompt = apply_styles_to_prompt(prompt, [self.find_style(x).negative_prompt for x in styles])
        if wildcards:
            prompt = apply_wildcards_to_prompt(prompt, [self.find_style(x).wildcards for x in styles])
        return prompt

    def apply_styles_to_extra(self, p):
        if len(getattr(p, 'original_prompt', '')) == 0:
            p.original_prompt = p.prompt
        if len(getattr(p, 'original_negative', '')) == 0:
            p.original_negative = p.negative_prompt

        if p.styles is None:
            return
        if p.styles is None or not isinstance(p.styles, list):
            shared.log.error(f'Styles invalid: {p.styles}')
            return
        for style in p.styles:
            s = self.find_style(style)
            if s == self.no_style:
                shared.log.warning(f'Apply style: name="{style}" not found')
                continue
            apply_styles_to_extra(p, s)

    def extract_comments(self, p):
        if not isinstance(p.prompt, str):
            return
        match = re.search(r'/\*.*?\*/', p.prompt, flags=re.DOTALL)
        if match:
            comment = match.group()
            p.prompt = p.prompt.replace(comment, '')
            p.extra_generation_params['Comment'] = comment.replace('/*', '').replace('*/', '')

    def save_styles(self, path, verbose=False):
        for name in list(self.styles):
            style = {
                "name": name,
                "prompt": self.styles[name].prompt,
                "negative": self.styles[name].negative_prompt,
                "extra": "",
                "preview": "",
            }
            keepcharacters = (' ','.','_')
            fn = "".join(c for c in name if c.isalnum() or c in keepcharacters).strip()
            fn = os.path.join(path, fn + ".json")
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    json.dump(style, f, indent=2)
                    if verbose:
                        shared.log.debug(f'Saved style: name={name} file="{fn}"')
            except Exception as e:
                shared.log.error(f'Failed to save style: name={name} file="{path}" error={e}')
        count = len(list(self.styles))
        if count > 0:
            shared.log.debug(f'Saved styles: folder="{path}" items={count}')

    def load_csv(self, legacy_file):
        if not os.path.isfile(legacy_file):
            return
        with open(legacy_file, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            num = 0
            for row in reader:
                try:
                    name = row["name"]
                    prompt = row["prompt"] if "prompt" in row else row["text"]
                    negative = row.get("negative_prompt", "") if "negative_prompt" in row else row.get("negative", "")
                    self.styles[name] = Style(name, desc=name, prompt=prompt, negative_prompt=negative)
                    shared.log.debug(f'Migrated style: {self.styles[name].__dict__}')
                    num += 1
                except Exception:
                    shared.log.error(f'Styles error: file="{legacy_file}" row={row}')
            shared.log.info(f'Load legacy styles: file="{legacy_file}" loaded={num} created={len(list(self.styles))}')
