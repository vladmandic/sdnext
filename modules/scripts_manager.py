import os
import re
import sys
import time
from collections import namedtuple
from dataclasses import dataclass
import gradio as gr
from modules import paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer
from installer import control_extensions


AlwaysVisible = object()
time_component = {}
time_setup = {}
debug = errors.log.trace if os.environ.get('SD_SCRIPT_DEBUG', None) is not None else lambda *args, **kwargs: None


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


@dataclass
class OnComponent:
    component: gr.blocks.Block


class Script:
    parent = None
    name = None
    filename = None
    args_from = 0
    args_to = 0
    alwayson = False
    is_txt2img = False
    is_img2img = False
    api_info = None
    group = None
    infotext_fields = None
    paste_field_names = None
    section = None
    standalone = False
    on_before_component_elem_id = [] # list of callbacks to be called before a component with an elem_id is created
    on_after_component_elem_id = [] # list of callbacks to be called after a component with an elem_id is created

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        raise NotImplementedError

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        pass # pylint: disable=unnecessary-pass

    def show(self, is_img2img): # pylint: disable=unused-argument
        """
        is_img2img is True if this function is called for the img2img interface, and False otherwise
        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.
        Usually the processing is done by calling the processing.process_images function.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass # pylint: disable=unnecessary-pass

    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def process_images(self, p, *args):
        """
        This function is called instead of main processing for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.
        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.
        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds
        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """
        pass # pylint: disable=unnecessary-pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """
        pass # pylint: disable=unnecessary-pass

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))
        return f'script_{self.parent}_{title}_{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path", "priority"])
scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


def list_scripts(scriptdirname, extension):
    tmp_list = []
    base = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(base):
        for filename in sorted(os.listdir(base)):
            tmp_list.append(ScriptFile(paths.script_path, filename, os.path.join(base, filename), '50'))
    for ext in extensions.active():
        tmp_list += ext.list_files(scriptdirname, extension)
    priority_list = []
    for script in tmp_list:
        if os.path.splitext(script.path)[1].lower() == extension and os.path.isfile(script.path):
            if script.basedir == paths.script_path:
                priority = '0'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'scripts')):
                priority = '1'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions-builtin')):
                priority = '2'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions')):
                priority = '3'
            else:
                priority = '9'
            if os.path.isfile(os.path.join(base, "..", ".priority")):
                with open(os.path.join(base, "..", ".priority"), "r", encoding="utf-8") as f:
                    priority = priority + str(f.read().strip())
                    errors.log.debug(f'Script priority override: ${script.name}:{priority}')
            else:
                priority = priority + script.priority
            priority_list.append(ScriptFile(script.basedir, script.filename, script.path, priority))
            debug(f'Adding script: folder="{script.basedir}" file="{script.filename}" full="{script.path}" priority={priority}')
    priority_sort = sorted(priority_list, key=lambda item: item.priority + item.path.lower(), reverse=False)
    return priority_sort


def list_files_with_name(filename):
    res = []
    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]
    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue
        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)
    return res


def load_scripts():
    t = timer.Timer()
    t0 = time.time()
    global current_basedir # pylint: disable=global-statement
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()
    scripts_list = list_scripts('scripts', '.py') + list_scripts(os.path.join('modules', 'face'), '.py')
    scripts_list = sorted(scripts_list, key=lambda item: item.priority + item.path.lower(), reverse=False)
    syspath = sys.path

    def register_scripts_from_module(module, scriptfile):
        for script_class in module.__dict__.values():
            if type(script_class) != type:
                continue
            debug(f'Registering script: path="{scriptfile.path}"')
            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    for scriptfile in scripts_list:
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir
            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module, scriptfile)
        except Exception as e:
            errors.display(e, f'Load script: {scriptfile.filename}')
        finally:
            current_basedir = paths.script_path
            t.record(os.path.basename(scriptfile.basedir) if scriptfile.basedir != paths.script_path else scriptfile.filename)
            sys.path = syspath

    global scripts_txt2img, scripts_img2img, scripts_control, scripts_postproc # pylint: disable=global-statement
    scripts_txt2img = ScriptRunner('txt2img')
    scripts_img2img = ScriptRunner('img2img')
    scripts_control = ScriptRunner('control')
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
    return t, time.time()-t0


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception as e:
        errors.display(e, f'Calling script: {filename}/{funcname}')
    return default


class ScriptSummary:
    def __init__(self, op):
        self.start = time.time()
        self.update = time.time()
        self.op = op
        self.time = {}

    def record(self, script):
        self.update = time.time()
        self.time[script] = round(time.time() - self.update, 2)

    def report(self):
        total = sum(self.time.values())
        if total == 0:
            return
        scripts = [f'{k}:{v}' for k, v in self.time.items() if v > 0]
        errors.log.debug(f'Script: op={self.op} total={total} scripts={scripts}')


class ScriptRunner:
    def __init__(self, name=''):
        self.name = name
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.auto_processing_scripts = []
        self.titles = []
        self.infotext_fields = []
        self.paste_field_names = []
        self.script_load_ctr = 0
        self.is_img2img = False
        self.inputs = [None]
        self.time = 0

    def add_script(self, script_class, path, is_img2img, is_control):
        try:
            script = script_class()
            script.filename = path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            if is_control: # this is messy but show is a legacy function that is not aware of control tab
                v1 = script.show(script.is_txt2img)
                v2 = script.show(script.is_img2img)
                if v1 == AlwaysVisible or v2 == AlwaysVisible:
                    visibility = AlwaysVisible
                else:
                    visibility = v1 or v2
            else:
                visibility = script.show(script.is_img2img)
            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True
            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)
        except Exception as e:
            errors.log.error(f'Script initialize: {path} {e}')

    def initialize_scripts(self, is_img2img=False, is_control=False):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.selectable_scripts.clear()
        self.alwayson_scripts.clear()
        self.titles.clear()
        self.infotext_fields.clear()
        self.paste_field_names.clear()
        self.script_load_ctr = 0
        self.is_img2img = is_img2img
        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()
        self.auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

        sorted_scripts = sorted(scripts_data, key=lambda x: x.script_class().title().lower())
        for script_class, path, _basedir, _script_module in sorted_scripts:
            self.add_script(script_class, path, is_img2img, is_control)
        sorted_scripts = sorted(self.auto_processing_scripts, key=lambda x: x.script_class().title().lower())
        for script_class, path, _basedir, _script_module in sorted_scripts:
            self.add_script(script_class, path, is_img2img, is_control)

    def prepare_ui(self):
        self.inputs = [None]

    def setup_ui(self, parent='unknown', accordion=True):
        import modules.api.models as api_models
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]

        inputs = []
        inputs_alwayson = [True]

        def create_script_ui(script: Script, inputs, inputs_alwayson):
            script.parent = parent
            script.args_from = len(inputs)
            script.args_to = len(inputs)
            controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)
            if controls is None:
                return
            script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()
            api_args = []
            for control in controls:
                debug(f'Script control: parent={script.parent} script="{script.name}" label="{control.label}" type={control} id={control.elem_id}')
                if hasattr(gr.components, 'IOComponent'):
                    if not isinstance(control, gr.components.IOComponent):
                        errors.log.error(f'Invalid script control: "{script.filename}" control={control}')
                        continue
                else:
                    if not isinstance(control, gr.components.Component):
                        errors.log.error(f'Invalid script control: "{script.filename}" control={control}')
                        continue
                control.custom_script_source = os.path.basename(script.filename)
                arg_info = api_models.ScriptArg(label=control.label or "")
                for field in ("value", "minimum", "maximum", "step", "choices"):
                    v = getattr(control, field, None)
                    if v is not None:
                        setattr(arg_info, field, v)
                api_args.append(arg_info)

            script.api_info = api_models.ItemScript(
                name=script.name,
                is_img2img=script.is_img2img,
                is_alwayson=script.alwayson,
                args=api_args,
            )
            if script.infotext_fields is not None:
                self.infotext_fields += script.infotext_fields
            if script.paste_field_names is not None:
                self.paste_field_names += script.paste_field_names
            inputs += controls
            inputs_alwayson += [script.alwayson for _ in controls]
            script.args_to = len(inputs)

        with gr.Row():
            dropdown = gr.Dropdown(label="Script", elem_id=f'{parent}_script_list', choices=["None"] + self.titles, value="None", type="index")
            inputs.insert(0, dropdown)

        with gr.Row():
            for script in self.alwayson_scripts:
                if not script.standalone:
                    continue
                if (self.name == 'control') and (script.name not in control_extensions) and (script.title() not in control_extensions):
                    errors.log.debug(f'Script: fn="{script.filename}" type={self.name} skip')
                    continue
                t0 = time.time()
                with gr.Group(elem_id=f'{parent}_script_{script.title().lower().replace(" ", "_")}', elem_classes=['group-extension']) as group:
                    create_script_ui(script, inputs, inputs_alwayson)
                script.group = group
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)

        with gr.Row():
            with gr.Accordion(label="Extensions", elem_id=f'{parent}_script_alwayson') if accordion else gr.Group():
                for script in self.alwayson_scripts:
                    if script.standalone:
                        continue
                    if (self.name == 'control') and (paths.extensions_dir in script.filename) and (script.title() not in control_extensions):
                        errors.log.debug(f'Script: fn="{script.filename}" type={self.name} skip')
                        continue
                    t0 = time.time()
                    with gr.Group(elem_id=f'{parent}_script_{script.title().lower().replace(" ", "_")}', elem_classes=['group-extension']) as group:
                        create_script_ui(script, inputs, inputs_alwayson)
                    script.group = group
                    time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)

        for script in self.selectable_scripts:
            if (self.name == 'control') and (paths.extensions_dir in script.filename) and (script.title() not in control_extensions):
                errors.log.debug(f'Script: fn="{script.filename}" type={self.name} skip')
                continue
            with gr.Group(elem_id=f'{parent}_script_{script.title().lower().replace(" ", "_")}', elem_classes=['group-scripts'], visible=False) as group:
                t0 = time.time()
                create_script_ui(script, inputs, inputs_alwayson)
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)
                script.group = group

        def select_script(script_index):
            if script_index is None:
                return [gr.update(visible=False) for script in self.selectable_scripts]
            selected_script = self.selectable_scripts[script_index - 1] if script_index > 0 else None
            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            if title == 'None': # called when an initial value is set from ui-config.json to show script's UI components
                return
            if title not in self.titles:
                errors.log.error(f'Script not found: {title}')
                return
            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field
        dropdown.change(fn=select_script, inputs=[dropdown], outputs=[script.group for script in self.selectable_scripts if script.group is not None])

        def onload_script_visibility(params):
            title = params.get('Script', None)
            if title:
                title_index = self.titles.index(title)
                visibility = title_index == self.script_load_ctr
                self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                return gr.update(visible=visibility)
            else:
                return gr.update(visible=False)

        self.infotext_fields.append((dropdown, lambda x: gr.update(value=x.get('Script', 'None'))))
        self.infotext_fields.extend([(script.group, onload_script_visibility) for script in self.selectable_scripts if script.group is not None])
        return inputs

    def run(self, p, *args):
        s = ScriptSummary('run')
        script_index = args[0] if len(args) > 0 else 0
        if script_index == 0:
            return None
        script = self.selectable_scripts[script_index-1]
        if script is None:
            return None
        if 'upscale' in script.title():
            if not hasattr(p, 'init_images') and p.task_args.get('image', None) is not None:
                p.init_images = p.task_args['image']
        parsed = []
        if hasattr(script, 'args_to') and hasattr(script, 'args_from'):
            parsed = p.per_script_args.get(script.title(), args[script.args_from:script.args_to])
        if hasattr(script, 'run'):
            processed = script.run(p, *parsed)
        else:
            processed = None
            errors.log.error(f'Script: file="{script.filename}" no run function defined')
        s.record(script.title())
        s.report()
        return processed

    def after(self, p, processed, *args):
        s = ScriptSummary('after')
        script_index = args[0] if len(args) > 0 else 0
        if script_index == 0:
            return processed
        script = self.selectable_scripts[script_index-1]
        if script is None or not hasattr(script, 'after'):
            return processed
        parsed = []
        if hasattr(script, 'args_to') and hasattr(script, 'args_from'):
            parsed = p.per_script_args.get(script.title(), args[script.args_from:script.args_to])
        after_processed = script.after(p, processed, *parsed)
        if after_processed is not None:
            processed = after_processed
        s.record(script.title())
        s.report()
        return processed

    def before_process(self, p, **kwargs):
        s = ScriptSummary('before-process')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.before_process(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f"Error running before process: {script.filename}")
            s.record(script.title())
        s.report()

    def process(self, p, **kwargs):
        s = ScriptSummary('process')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.process(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script process: {script.filename}')
            s.record(script.title())
        s.report()

    def process_images(self, p, **kwargs):
        s = ScriptSummary('process_images')
        processed = None
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    _processed = script.process_images(p, *args, **kwargs)
                    if _processed is not None:
                        processed = _processed
            except Exception as e:
                errors.display(e, f'Running script process images: {script.filename}')
            s.record(script.title())
        s.report()
        return processed

    def before_process_batch(self, p, **kwargs):
        s = ScriptSummary('before-process-batch')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.before_process_batch(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before process batch: {script.filename}')
            s.record(script.title())
        s.report()

    def process_batch(self, p, **kwargs):
        s = ScriptSummary('process-batch')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.process_batch(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script process batch: {script.filename}')
            s.record(script.title())
        s.report()

    def postprocess(self, p, processed):
        s = ScriptSummary('postprocess')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.postprocess(p, processed, *args)
            except Exception as e:
                errors.display(e, f'Running script postprocess: {script.filename}')
            s.record(script.title())
        s.report()

    def postprocess_batch(self, p, images, **kwargs):
        s = ScriptSummary('postprocess-batch')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.postprocess_batch(p, *args, images=images, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before postprocess batch: {script.filename}')
            s.record(script.title())
        s.report()

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        s = ScriptSummary('postprocess-batch-list')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.postprocess_batch_list(p, pp, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before postprocess batch list: {script.filename}')
            s.record(script.title())
        s.report()

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        s = ScriptSummary('postprocess-image')
        for script in self.alwayson_scripts:
            try:
                if hasattr(script, 'args_to') and hasattr(script, 'args_from') and (script.args_to > 0) and (script.args_to >= script.args_from):
                    args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                    script.postprocess_image(p, pp, *args)
            except Exception as e:
                errors.display(e, f'Running script postprocess image: {script.filename}')
            s.record(script.title())
        s.report()

    def before_component(self, component, **kwargs):
        s = ScriptSummary('before-component')
        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before component: {script.filename}')
            s.record(script.title())
        s.report()

    def after_component(self, component, **kwargs):
        s = ScriptSummary('after-component')
        for script in self.scripts:
            for elem_id, callback in script.on_after_component_elem_id:
                if elem_id == kwargs.get("elem_id"):
                    try:
                        callback(OnComponent(component=component))
                    except Exception as e:
                        errors.display(e, f"Running script before_component_elem_id: {script.filename}")
            try:
                script.after_component(component, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script after component: {script.filename}')
            s.record(script.title())
        s.report()

    def reload_sources(self, cache):
        s = ScriptSummary('reload-sources')
        for si, script in list(enumerate(self.scripts)):
            if hasattr(script, 'args_to') and hasattr(script, 'args_from'):
                args_from = script.args_from
                args_to = script.args_to
                filename = script.filename
                module = cache.get(filename, None)
                if module is None:
                    module = script_loading.load_module(script.filename)
                    cache[filename] = module
                for script_class in module.__dict__.values():
                    if type(script_class) == type and issubclass(script_class, Script):
                        self.scripts[si] = script_class()
                        self.scripts[si].filename = filename
                        self.scripts[si].args_from = args_from
                        self.scripts[si].args_to = args_to
                s.record(script.title())
        s.report()


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_control: ScriptRunner = None
scripts_current: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
reload_scripts = load_scripts  # compatibility alias


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)
    scripts_control.reload_sources(cache)
