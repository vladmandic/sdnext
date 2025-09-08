from dataclasses import dataclass
from installer import log


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        v.section = section_identifier
    return options_dict


class OptionInfo:
    def __init__(
            self,
            default=None,
            label="",
            component=None,
            component_args=None,
            onchange=None,
            section=None,
            refresh=None,
            folder=None,
            submit=None,
            comment_before='',
            comment_after='',
            category_id=None, # pylint: disable=unused-argument
            *args, # pylint: disable=unused-argument
            **kwargs, # pylint: disable=unused-argument
        ): # pylint: disable=keyword-arg-before-vararg
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh
        self.folder = folder
        self.comment_before = comment_before # HTML text that will be added after label in UI
        self.comment_after = comment_after # HTML text that will be added before label in UI
        self.submit = submit
        self.exclude = ['sd_model_checkpoint', 'sd_model_refiner', 'sd_vae', 'sd_unet', 'sd_text_encoder']
        self.dynamic = callable(component_args)
        args = {} if self.dynamic else (component_args or {}) # executing callable here is too expensive
        self.visible = args.get('visible', True) and len(self.label) > 2

    def needs_reload_ui(self):
        return self

    def link(self, label, uri):
        self.comment_before += f"[<a href='{uri}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, info):
        self.comment_after += f"<span class='info'>{info}</span>"
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    def validate(self, opt, value):
        if opt in self.exclude:
            return True
        args = self.component_args if self.component_args is not None else {}
        if callable(args):
            try:
                args = args()
            except Exception:
                args = {}
        choices = args.get("choices", [])
        if callable(choices):
            try:
                choices = choices()
            except Exception:
                choices = []
        if len(choices) > 0:
            if not isinstance(value, list):
                value = [value]
            for v in value:
                if v not in choices:
                    if isinstance(choices, list) and ('All' in choices or 'all' in choices): # may be added dynamically
                        continue
                    log.debug(f'Setting validation: "{opt}"="{v}" default="{self.default}" choices={choices}')
                    # return False
        minimum = args.get("minimum", None)
        maximum = args.get("maximum", None)
        try:
            if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                log.error(f'Setting validation: "{opt}"={value} default={self.default} minimum={minimum} maximum={maximum}')
                return False
        except Exception as err:
            log.error(f'Setting validation: "{opt}"={value} default={self.default} minimum={minimum} maximum={maximum} error={err}')
            return False
        return True

    def __str__(self) -> str:
        args = self.component_args if self.component_args is not None else {}
        if callable(args):
            args = args()
        choices = args.get("choices", [])
        return f'OptionInfo: label="{self.label}" section="{self.section}" component="{self.component}" default="{self.default}" refresh="{self.refresh is not None}" change="{self.onchange is not None}" args={args} choices={choices}'


@dataclass
class OptionsCategory:
    id: str
    label: str

class OptionsCategories:
    def __init__(self):
        self.mapping = {}

    def register_category(self, category_id, label):
        if category_id not in self.mapping:
            self.mapping[category_id] = OptionsCategory(category_id, label)
        return category_id


categories = OptionsCategories()
