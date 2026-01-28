import re
import os
import time
import unicodedata
import uuid
import string
import hashlib
import datetime
from pathlib import Path
from modules import shared, errors


debug= os.environ.get('SD_NAMEGEN_DEBUG', None) is not None
debug_log = errors.log.trace if debug else lambda *args, **kwargs: None
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
re_attention = re.compile(r'[\(*\[*](\w+)(:\d+(\.\d+))?[\)*\]*]|')
re_network = re.compile(r'\<\w+:(\w+)(:\d+(\.\d+))?\>|')
re_brackets = re.compile(r'[\([{})\]]')
re_leading_seq = re.compile(r'^(0*\d+)(?=[-_.\s]|$)')
seq = 0
NOTHING = object()


class FilenameGenerator:
    replacements = {
        'width': lambda self: self.width,
        'height': lambda self: self.height,
        'batch_number': lambda self: self.batch_number,
        'iter_number': lambda self: self.iter_number,
        'num': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'generation_number': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        'hasprompt': lambda self, *args: self.hasprompt(*args),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        'hash': lambda self: self.image_hash() if self.image is not None else '',
        'image_hash': lambda self: self.image_hash() if self.image is not None else '',
        'timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'epoch': lambda self: int(time.time()),
        'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),

        'model': lambda self: shared.sd_model.sd_checkpoint_info.title if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_shortname': lambda self: shared.sd_model.sd_checkpoint_info.model_name if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_name': lambda self: shared.sd_model.sd_checkpoint_info.model_name if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',
        'model_type': lambda self: shared.sd_model_type if shared.sd_loaded else '',
        'model_hash': lambda self: shared.sd_model.sd_checkpoint_info.shorthash if shared.sd_loaded and getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None else '',

        'lora': lambda self: self.p and getattr(self.p, 'extra_generation_params', {}).get('LoRA networks', ''),

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

    def __init__(self, p, seed, prompt, image=None, grid=False, width=None, height=None):
        if p is None:
            debug_log('Filename generator init skip')
        else:
            debug_log(f'Filename generator init: seed={seed} prompt="{prompt}"')
        self.p = p
        if seed is not None and int(seed) > 0:
            self.seed = seed
        elif p is not None and getattr(p, 'all_seeds', None) is not None and len(p.all_seeds) > 0:
            self.seed = p.all_seeds[0] if p.all_seeds[0] is not None and int(p.all_seeds[0]) > 0 else 0
        elif p is not None and getattr(p, 'seeds', None) is not None and len(p.seeds) > 0:
            self.seed = p.seeds[0] if p.seeds[0] is not None and int(p.seeds[0]) > 0 else 0
        else:
            self.seed = p.seed if p is not None and getattr(p, 'seed', 0) > 0 else 0
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = p.prompt if p is not None and getattr(p, 'prompt', '') != '' else ''
        if isinstance(self.prompt, list):
            self.prompt = ' '.join(self.prompt)
        self.image = image[0] if isinstance(image, list) and len(image) > 0 else image
        self.width = width if width is not None else (self.image.width if self.image is not None else (p.width if p is not None else 0))
        self.height = height if height is not None else (self.image.height if self.image is not None else (p.height if p is not None else 0))
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
        debug_log(f'Prompt sanitize: input="{prompt}" output="{sanitized}"')
        return sanitized

    def sanitize(self, filename):
        # starting reference: <https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file>
        invalid_chars = (
            "#<>\"'`"                         # ASCII quote and backtick
            "’‚‛\u2018\u2019\u201B"           # smart single quotes and variants # noqa: RUF001
            "\u02BB"                          # modifier letter turned comma
            "\u201C\u201D\u201F"              # smart double quotes and variants
            "|?*^%$\u00A0\u2013\u2014\n\t\r"  # pipes, wildcards, percent, currency, NBSP, dashes, control chars
        )
        invalid_folder = ':'
        invalid_files = ['CON', 'PRN', 'AUX', 'NUL', 'NULL', 'COM0', 'COM1', 'LPT0', 'LPT1']
        invalid_prefix = ', '
        invalid_suffix = '.,_ '
        fn, ext = os.path.splitext(unicodedata.normalize('NFKC', filename))
        fn = fn.strip()
        ext = ext.strip()
        parts = Path(fn).parts
        newparts = []
        # for ch in filename:
        #     print(repr(ch), hex(ord(ch)), unicodedata.name(ch, 'UNKNOWN'), ch in invalid_chars)
        for i, part in enumerate(parts):
            part = part.translate({ ord(x): '_' for x in invalid_chars })
            if i > 0 or (len(part) >= 2 and part[1] != invalid_folder): # skip drive, otherwise remove
                part = part.translate({ ord(x): '_' for x in invalid_folder })
            part = part.lstrip(invalid_prefix).rstrip(invalid_suffix)
            if part in invalid_files: # reserved names
                [part := part.replace(word, '_') for word in invalid_files] # pylint: disable=expression-not-assigned
            newparts.append(part)
        fn = str(Path(*newparts))
        fn = fn.replace('  ', ' ').strip()
        max_length = max(256 - len(ext), os.statvfs(__file__).f_namemax - 32 if hasattr(os, 'statvfs') else 256 - len(ext))
        while len(os.path.abspath(fn)) > max_length:
            fn = fn[:-1]
        fn += ext
        debug_log(f'Filename sanitize: input="{filename}" parts={parts} output="{fn}" ext={ext} max={max_length} len={len(fn)}')
        return fn

    def safe_int(self, s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return 0

    def sequence(self, fn):
        global seq # pylint: disable=global-statement
        x = fn
        dirname = os.path.dirname(fn)
        if seq == 0:
            files = os.listdir(dirname) if os.path.exists(dirname) and os.path.isdir(dirname) else []
            files = [f for f in files if os.path.isfile(os.path.join(dirname, f))]
            seq_files = len(files)
            seq_nums = [re_leading_seq.match(f) for f in files]
            seq_nums = [self.safe_int(m.group(1)) for m in seq_nums if m is not None]
            seq_num = max(seq_nums) if len(seq_nums) > 0 else 0
            seq = max(seq_files, seq_num)
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
        if debug:
            for k in self.replacements.keys():
                try:
                    fn = self.replacements.get(k, None)
                    debug_log(f'Namegen: key={k} value={fn(self)}')
                except Exception as e:
                    shared.log.error(f'Namegen: key={k} {e}')
                    errors.display(e, 'namegen')
        for m in re_pattern.finditer(x):
            text, pattern = m.groups()
            debug_log(f'Filename apply: text="{text}" pattern="{pattern}"')
            if isinstance(pattern, list):
                pattern = ' '.join(pattern)
            if pattern is None or not isinstance(pattern, str) or pattern.strip() == '':
                debug_log(f'Filename skip: pattern="{pattern}"')
                res += text
                continue

            _pattern = pattern
            pattern_args = []
            while True:
                m = re_pattern_arg.match(_pattern)
                if m is None:
                    break
                _pattern, arg = m.groups()
                pattern_args.insert(0, arg)

            fun = self.replacements.get(pattern.lower(), None)
            if fun is not None:
                try:
                    replacement = fun(self, *pattern_args)
                    debug_log(f'Filename apply: pattern="{pattern}" args={pattern_args} replacement="{replacement}"')
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
