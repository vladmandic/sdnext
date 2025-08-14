import re
import os
import uuid
import string
import hashlib
import datetime
from pathlib import Path
from modules import shared, errors


debug = errors.log.trace if os.environ.get('SD_NAMEGEN_DEBUG', None) is not None else lambda *args, **kwargs: None
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
re_attention = re.compile(r'[\(*\[*](\w+)(:\d+(\.\d+))?[\)*\]*]|')
re_network = re.compile(r'\<\w+:(\w+)(:\d+(\.\d+))?\>|')
re_brackets = re.compile(r'[\([{})\]]')
seq = 0
NOTHING = object()


class FilenameGenerator:
    replacements = {
        'width': lambda self: self.image.width,
        'height': lambda self: self.image.height,
        'batch_number': lambda self: self.batch_number,
        'iter_number': lambda self: self.iter_number,
        'num': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'generation_number': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        'hasprompt': lambda self, *args: self.hasprompt(*args),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        'hash': lambda self: self.image_hash(),
        'image_hash': lambda self: self.image_hash(),
        'timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),

        'model': lambda self: shared.sd_model.sd_checkpoint_info.title if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_shortname': lambda self: shared.sd_model.sd_checkpoint_info.model_name if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_name': lambda self: shared.sd_model.sd_checkpoint_info.model_name if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_type': lambda self: shared.sd_model_type if shared.sd_loaded else '',
        'model_hash': lambda self: shared.sd_model.sd_checkpoint_info.shorthash if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',

        'prompt': lambda self: self.prompt_full(),
        'prompt_no_styles': lambda self: self.prompt_no_style(),
        'prompt_words': lambda self: self.prompt_words(),
        'prompt_hash': lambda self: hashlib.sha256(self.prompt.encode()).hexdigest()[0:8],

        'sampler': lambda self: self.p and self.p.sampler_name,
        'seed': lambda self: (self.seed and str(self.seed)) or '',
        'steps': lambda self: self.p and getattr(self.p, 'steps', 0),
        'cfg': lambda self: self.p and getattr(self.p, 'cfg_scale', 0),
        'pag': lambda self: self.p and getattr(self.p, 'pag_scale', 0),
        'clip_skip': lambda self: self.p and getattr(self.p, 'clip_skip', 0),
        'denoising': lambda self: self.p and getattr(self.p, 'denoising_strength', 0),
        'styles': lambda self: (self.p and ", ".join([style for style in self.p.styles if not style == "None"])) or "None",
        'uuid': lambda self: str(uuid.uuid4()),
    }
    default_time_format = '%Y%m%d%H%M%S'

    def __init__(self, p, seed, prompt, image, grid=False):
        if p is None:
            debug('Filename generator init skip')
        else:
            debug(f'Filename generator init: {seed} {prompt}')
        self.p = p
        if seed is not None and int(seed) > 0:
            self.seed = seed
        elif p is not None and hasattr(p, 'all_seeds'):
            self.seed = p.all_seeds[0]
        elif p is not None and hasattr(p, 'seeds'):
            self.seed = p.seeds[0]
        else:
            self.seed = p.seed if p is not None else 0
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = p.prompt if p is not None else ''
        if isinstance(self.prompt, list):
            self.prompt = ' '.join(self.prompt)
        self.image = image
        if not grid:
            self.batch_number = NOTHING if self.p is None or getattr(self.p, 'batch_size', 1) == 1 else (self.p.batch_index + 1 if hasattr(self.p, 'batch_index') else NOTHING)
            self.iter_number = NOTHING if self.p is None or getattr(self.p, 'n_iter', 1) == 1 else (self.p.iteration + 1 if hasattr(self.p, 'iteration') else NOTHING)
        else:
            self.batch_number = NOTHING
            self.iter_number = NOTHING

    def hasprompt(self, *args):
        lower = self.prompt.lower()
        if getattr(self, 'p', None) is None or getattr(self, 'prompt', None) is None:
            return None
        outres = ""
        for arg in args:
            if arg != "":
                division = arg.split("|")
                expected = division[0].lower()
                default = division[1] if len(division) > 1 else ""
                if lower.find(expected) >= 0:
                    outres = f'{outres}{expected}'
                else:
                    outres = outres if default == "" else f'{outres}{default}'
        return outres

    def image_hash(self):
        if getattr(self, 'image', None) is None:
            return None
        import base64
        from io import BytesIO
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        shorthash = hashlib.sha256(img_str).hexdigest()[0:8]
        return shorthash

    def prompt_full(self):
        return self.prompt_sanitize(self.prompt)

    def prompt_words(self):
        if getattr(self, 'prompt', None) is None:
            return ''
        no_attention = re_attention.sub(r'\1', self.prompt)
        no_network = re_network.sub(r'\1', no_attention)
        no_brackets = re_brackets.sub('', no_network)
        words = [x for x in re_nonletters.split(no_brackets or "") if len(x) > 0]
        prompt = " ".join(words[0:shared.opts.directories_max_prompt_words])
        return self.prompt_sanitize(prompt)

    def prompt_no_style(self):
        if getattr(self, 'p', None) is None or getattr(self, 'prompt', None) is None:
            return None
        prompt_no_style = self.prompt
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if len(style) > 0:
                for part in style.split("{prompt}"):
                    prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",")
                prompt_no_style = prompt_no_style.replace(style, "")
        return self.prompt_sanitize(prompt_no_style)

    def datetime(self, *args):
        import pytz
        time_datetime = datetime.datetime.now()
        time_format = args[0] if len(args) > 0 and args[0] != "" else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError:
            time_zone = None
        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError):
            formatted_time = time_zone_time.strftime(self.default_time_format)
        return formatted_time

    def prompt_sanitize(self, prompt):
        invalid_chars = '#<>:\'"\\|?*\n\t\r'
        sanitized = prompt.translate({ ord(x): '_' for x in invalid_chars }).strip()
        debug(f'Prompt sanitize: input="{prompt}" output={sanitized}')
        return sanitized

    def sanitize(self, filename):
        invalid_chars = '\'"|?*\n\t\r' # <https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file>
        invalid_folder = ':'
        invalid_files = ['CON', 'PRN', 'AUX', 'NUL', 'NULL', 'COM0', 'COM1', 'LPT0', 'LPT1']
        invalid_prefix = ', '
        invalid_suffix = '.,_ '
        fn, ext = os.path.splitext(filename)
        parts = Path(fn).parts
        newparts = []
        for i, part in enumerate(parts):
            part = part.translate({ ord(x): '_' for x in invalid_chars })
            if i > 0 or (len(part) >= 2 and part[1] != invalid_folder): # skip drive, otherwise remove
                part = part.translate({ ord(x): '_' for x in invalid_folder })
            part = part.lstrip(invalid_prefix).rstrip(invalid_suffix)
            if part in invalid_files: # reserved names
                [part := part.replace(word, '_') for word in invalid_files] # pylint: disable=expression-not-assigned
            newparts.append(part)
        fn = str(Path(*newparts))
        max_length = max(256 - len(ext), os.statvfs(__file__).f_namemax - 32 if hasattr(os, 'statvfs') else 256 - len(ext))
        while len(os.path.abspath(fn)) > max_length:
            fn = fn[:-1]
        fn += ext
        debug(f'Filename sanitize: input="{filename}" parts={parts} output="{fn}" ext={ext} max={max_length} len={len(fn)}')
        return fn

    def sequence(self, fn):
        global seq # pylint: disable=global-statement
        x = fn
        dirname = os.path.dirname(fn)
        if seq == 0:
            seq = len(os.listdir(dirname)) if os.path.exists(dirname) and os.path.isdir(dirname) else 0
        if shared.opts.save_images_add_number or '[seq]' in fn:
            if '[seq]' not in fn:
                fn = os.path.join(os.path.dirname(fn), f"[seq]-{os.path.basename(fn)}")
            for _i in range(99999): # 99999/000001
                seq += 1
                dst = fn.replace('[seq]', f'{seq:05}')
                if not os.path.exists(dst):
                    x = dst
                    break
        return x

    def apply(self, x):
        res = ''
        for m in re_pattern.finditer(x):
            text, pattern = m.groups()
            if pattern is None:
                res += text
                continue
            pattern_args = []
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break
                pattern, arg = m.groups()
                pattern_args.insert(0, arg)
            if isinstance(pattern, list):
                pattern = ' '.join(pattern)
            fun = self.replacements.get(pattern.lower(), None)
            if fun is not None:
                try:
                    debug(f'Filename apply: pattern={pattern.lower()} args={pattern_args}')
                    replacement = fun(self, *pattern_args)
                except Exception as e:
                    replacement = None
                    errors.display(e, 'namegen')
                    shared.log.error(f'Filename apply pattern: {x} {e}')
                if replacement == NOTHING:
                    continue
                if replacement is not None:
                    res += text + str(replacement).replace('/', '-').replace('\\', '-')
                    continue
            else:
                res += text + f'[{pattern}]' # reinsert unknown pattern
        return res


def get_next_sequence_number(path, basename): # pylint: disable=unused-argument
    global seq # pylint: disable=global-statement
    seq += 1
    return seq # unused
