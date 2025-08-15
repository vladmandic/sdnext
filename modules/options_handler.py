import os
import json
import threading
from typing import TYPE_CHECKING
from modules import cmd_args, errors
from modules.json_helpers import readfile, writefile
from modules.shared_legacy import LegacyOption
from installer import log
if TYPE_CHECKING:
    from modules.options import OptionInfo


cmd_opts = cmd_args.parse_args()
compatibility_opts = ['clip_skip', 'uni_pc_lower_order_final', 'uni_pc_order']


class Options():
    data = None
    data_labels = None
    filename = None
    typemap = {int: float}
    debug = os.environ.get('SD_CONFIG_DEBUG', None) is not None

    def __init__(self, options_templates:dict={}, restricted_opts:dict={}):
        self.data_labels = options_templates
        self.restricted_opts = restricted_opts
        self.data = {k: v.default for k, v in self.data_labels.items()}
        self.legacy = [k for k, v in self.data_labels.items() if isinstance(v, LegacyOption)]

    def __setattr__(self, key, value): # pylint: disable=inconsistent-return-statements
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                if cmd_opts.freeze:
                    log.warning(f'Settings are frozen: {key}')
                    return
                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    log.warning(f'Settings key is restricted: {key}')
                    return
                if self.debug:
                    log.trace(f'Settings set: {key}={value}')
                if key in self.legacy:
                    log.warning(f'Settings set: {key}={value} legacy')
                self.data[key] = value
                return
        return super(Options, self).__setattr__(key, value) # pylint: disable=super-with-arguments

    def get(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments

    def set(self, key, value):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""
        oldval = self.data.get(key, None)
        if oldval is None:
            oldval = self.data_labels[key].default
        if oldval == value:
            return False
        try:
            setattr(self, key, value)
        except RuntimeError:
            return False
        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as err:
                log.error(f'Error in onchange callback: {key} {value} {err}')
                errors.display(err, 'Error in onchange callback')
                setattr(self, key, oldval)
                return False
        return True

    def get_default(self, key):
        """returns the default value for the key"""
        data_label = self.data_labels.get(key)
        return data_label.default if data_label is not None else None

    def list(self):
        """list all visible options"""
        components = [k for k, v in self.data_labels.items() if v.visible]
        return components

    def save_atomic(self, filename=None, silent=False):
        if self.filename is None:
            self.filename = cmd_opts.config
        if filename is None:
            filename = self.filename
        filename = os.path.abspath(filename)
        if cmd_opts.freeze:
            log.warning(f'Setting: fn="{filename}" save disabled')
            return
        try:
            diff = {}
            unused_settings = []

            # if self.debug:
            #     log.debug('Settings: user')
            #     for k, v in self.data.items():
            #         log.trace(f'  Config: item={k} value={v} default={self.data_labels[k].default if k in self.data_labels else None}')

            if self.debug:
                log.debug(f'Settings: total={len(self.data.keys())} known={len(self.data_labels.keys())}')

            for k, v in self.data.items():
                if k in self.data_labels:
                    default = self.data_labels[k].default
                    if isinstance(v, list):
                        if (len(default) != len(v) or set(default) != set(v)): # list order is non-deterministic
                            diff[k] = v
                            if self.debug:
                                log.trace(f'Settings changed: {k}={v} default={default}')
                    elif self.data_labels[k].default != v:
                        diff[k] = v
                        if self.debug:
                            log.trace(f'Settings changed: {k}={v} default={default}')
                else:
                    if k not in compatibility_opts:
                        diff[k] = v
                        if not k.startswith('uiux_'):
                            unused_settings.append(k)
                        if self.debug:
                            log.trace(f'Settings unknown: {k}={v}')
            writefile(diff, filename, silent=silent)
            if self.debug:
                log.trace(f'Settings save: count={len(diff.keys())} {diff}')
            if len(unused_settings) > 0:
                log.debug(f"Settings: unused={unused_settings}")
        except Exception as err:
            log.error(f'Settings: fn="{filename}" {err}')

    def save(self, filename=None, silent=False):
        threading.Thread(target=self.save_atomic, args=(filename, silent)).start()

    def same_type(self, x, y):
        if x is None or y is None:
            return True
        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))
        return type_x == type_y

    def load(self, filename=None):
        if filename is None:
            filename = self.filename
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            log.debug(f'Settings: fn="{filename}" created')
            self.save(filename)
            return
        self.data = readfile(filename, lock=True)
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]
        unknown_settings = []
        for k, v in self.data.items():
            info: OptionInfo = self.data_labels.get(k, None)
            if info is not None:
                if not info.validate(k, v):
                    self.data[k] = info.default
            if info is not None and not self.same_type(info.default, v):
                log.warning(f"Setting validation: {k}={v} ({type(v).__name__} expected={type(info.default).__name__})")
                self.data[k] = info.default
            if info is None and k not in compatibility_opts and not k.startswith('uiux_'):
                unknown_settings.append(k)
        if len(unknown_settings) > 0:
            log.warning(f"Setting validation: unknown={unknown_settings}")

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func
        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        metadata = {
            k: {
                "is_stored": k in self.data and self.data[k] != self.data_labels[k].default, # pylint: disable=unnecessary-dict-index-lookup
                "tab_name": v.section[0]
            } for k, v in self.data_labels.items()
        }
        return json.dumps({"values": d, "metadata": metadata})

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""
        section_ids = {}
        settings_items = self.data_labels.items()
        for _k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)
        self.data_labels = dict(sorted(settings_items, key=lambda x: section_ids[x[1].section]))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """
        if value is None:
            return None
        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None
        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        elif expected_type == type(value):
            pass
        else:
            value = expected_type(value)
        return value
