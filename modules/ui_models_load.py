import os
import re
import inspect
import gradio as gr
import torch
import diffusers
from huggingface_hub import hf_hub_download
from modules import shared, errors, shared_items, sd_models, sd_checkpoint, devices, model_quant, modelloader
from modules.logger import log


debug_enabled = os.environ.get('SD_LOAD_DEBUG', None)
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None
components = []


def load_model(model: str, cls: str, repo: str, dataframes: list):
    if cls is None:
        log.error('Model load: class is None')
        return 'Model load: class is None'
    if repo is None:
        log.error('Model load: repo is None')
        return 'Model load: repo is None'
    cls = getattr(diffusers, cls, None)
    if cls is None:
        cls = diffusers.AutoPipelineForText2Image
    log.info(f'Model load: name="{model}" cls={cls.__name__} repo="{repo}"')
    kwargs = {}
    for df in dataframes:
        c = [x for x in components if x.id == df[0]]
        if len(c) != 1:
            debug_log(f'Model load component: id={df[0]} not found')
            continue
        c = c[0]
        if not c.loadable: # not loadable
            debug_log(f'Model load component: name={c.name} not loadable')
            continue
        if c.type != 'class':
            debug_log(f'Model load component: name={c.name} not class')
            continue
        if len(c.local or '') == 0 and len(c.remote or '') == 0:
            debug_log(f'Model load component: name={c.name} no local or remote')
            continue
        instance = c.load()
        if instance is not None:
            kwargs[c.name] = instance
            log.info(f'Model component: instance={instance.__class__.__name__}')
    log.info(f'Model load: name="{model}" cls={cls.__name__} repo="{repo}" preload={kwargs.keys()}')
    pipe = None
    if model == 'Current':
        for k, v in kwargs.items():
            debug_log(f'Model replace component={k}')
            setattr(shared.sd_model, k, v)
        sd_models.set_diffuser_options(shared.sd_model)
        return f'Model load: name="{model}" cls={cls.__name__} repo="{repo}" preload={kwargs.keys()}'
    else:
        try:
            pipe = cls.from_pretrained(
                repo,
                dtype=devices.dtype,
                cache_dir=shared.opts.diffusers_dir,
                **kwargs,
            )
        except Exception as e:
            log.error(f'Model load: name="{model}" {e}')
            errors.display(e, 'Model load')
            return f'Model load failed: {e}'
        if pipe is not None:
            log.info(f'Model load: name="{model}" cls={cls.__name__} repo="{repo}" instance={pipe.__class__.__name__}')
            shared.sd_model = pipe
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo)
            shared.sd_model.sd_model_hash = None
            sd_models.set_diffuser_options(shared.sd_model)
            return f'Model load: name="{model}" cls={cls.__name__} repo="{repo}" preload={kwargs.keys()}'
    return 'Model load: no model'


def unload_model():
    sd_models.unload_model_weights(op='model')
    return 'Model unloaded'


def process_huggingface_url(url):
    if url is None or len(url) == 0:
        return None, None, None, False
    url = url.replace('https://huggingface.co/', '').strip() # remove absolute url
    url = re.sub(r'/blob/[^/]+/', '/', url) # remove /blob/<branch_id>/
    parts = url.split('/')
    repo = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else url # get repo
    subfolder = None
    fn = None
    if len(parts) == 3: # can be subfolder or filename
        if '.' in parts[-1]:
            fn = parts[-1]
        else:
            subfolder = parts[-1]
    elif len(parts) > 3: # There's at least one subfolder
        subfolder = '/'.join(parts[2:-1])
        fn = parts[-1]
    download = fn is not None
    return repo, subfolder, fn, download


class Component:
    def __init__(self, signature, name=None, cls=None, val=None, local=None, remote=None, typ=None, dtype=None, quant=False, loadable=None):
        self.id = len(components) + 1
        self.name = signature.name if signature else name
        self.cls = signature.annotation if signature else cls
        self.str = str(signature.annotation) if signature else str(cls)
        self.val = signature.default if signature and signature.default is not inspect.Parameter.empty else val
        self.remote = remote
        self.repo, self.subfolder, self.local, self.download = process_huggingface_url(self.remote)
        self.local = local or self.local
        self.dtype = str(dtype or devices.dtype).rsplit('.', maxsplit=1)[-1]
        self.quant = quant
        self.revision = None
        self.enum = None
        if typ is not None:
            self.type = typ
        else:
            if self.cls in [str, int, float, bool]:
                self.type = 'variable'
            elif 'enum' in self.str:
                self.type = 'enum'
                self.enum = [v.name for v in self.cls]
            elif inspect.isclass(signature.annotation):
                self.type = 'class'
            elif inspect.ismodule(signature.annotation):
                self.type = 'module'
            elif inspect.isfunction(signature.annotation):
                self.type = 'function'
            elif 'typing.Optional' in self.str:
                self.type = 'optional'
                self.cls = signature.annotation.__args__[0]
                self.str = str(self.cls)
                self.val = None
            else:
                self.type = 'unknown'
        self.str = re.search(r"'(.*?)'", self.str).group(1) if re.search(r"'(.*?)'", self.str) else self.str
        if '.' in self.str:
            self.str = self.str.split('.')
            self.str = self.str[0] + '.' + self.str[-1]
        self.loadable = loadable if loadable is not None else (self.type == 'class' and hasattr(self.cls, 'from_pretrained'))
        if not self.loadable:
            self.dtype = None
            self.quant = None

    def __str__(self):
        return f'id={self.id} name="{self.name}" cls={self.cls} type={self.type} loadable={self.loadable} val="{self.val}" str="{self.str}" enum="{self.enum}" local="{self.local}" remote="{self.remote}" repo="{self.repo}" subfolder="{self.subfolder}" dtype={self.dtype} quant={self.quant} revision={self.revision}'

    def save(self):
        return [self.name, self.local, self.remote, self.dtype, self.quant]

    def dataframe(self):
        return [self.id, self.name, self.loadable, self.val, self.str, self.local, self.remote, self.dtype, self.quant]

    def load(self):
        if not self.loadable:
            return None
        modelloader.hf_login()

        load_args = {}
        if self.subfolder is not None:
            load_args['subfolder'] = self.subfolder
        if self.revision is not None:
            load_args['revision'] = self.revision
        if self.dtype is not None:
            load_args['torch_dtype'] = getattr(torch, self.dtype)
        if not hasattr(self.cls, 'from_pretrained'):
            debug_log(f'Model load component: name="{self.name}" cls={self.cls} not loadable')
            return None
        quant_args = model_quant.create_config(module='any', allow=self.quant)
        quant_type = model_quant.get_quant_type(quant_args)

        try:
            if self.download:
                debug_log(f'Model load component: url="{self.remote}" args={load_args} quant={quant_type}')
                self.local = hf_hub_download(
                    repo_id=self.repo,
                    subfolder=self.subfolder,
                    filename=self.local,
                    revision=self.revision,
                    cache_dir=shared.opts.hfcache_dir,
                )
                if os.path.exists(self.local):
                    self.download = False
            if self.local is not None and len(self.local) > 0:
                if not os.path.exists(self.local):
                    debug_log(f'Model load component: local="{self.local}" file not found')
                elif hasattr(self.cls, 'from_single_file') and os.path.isfile(self.local) and self.local.endswith('.safetensors'):
                    debug_log(f'Model load component: local="{self.local}" type=file args={load_args} quant={quant_type}')
                    return self.cls.from_single_file(self.local, **load_args, **quant_args, cache_dir=shared.opts.hfcache_dir)
                elif os.path.isfile(self.local) and self.local.endswith('.gguf'):
                    debug_log(f'Model load component: local="{self.local}" type=gguf args={load_args} quant={quant_type}')
                    from modules import ggml
                    return ggml.load_gguf(self.local, cls=self.cls, compute_dtype=self.dtype)
                else:
                    debug_log(f'Model load component: local="{self.local}" type=folder args={load_args} quant={quant_type}')
                    return self.cls.from_pretrained(self.local, **load_args, **quant_args, cache_dir=shared.opts.hfcache_dir)
            elif self.repo is not None and len(self.repo) > 0:
                debug_log(f'Model load component: repo="{self.repo}" args={load_args} quant={quant_type}')
                return self.cls.from_pretrained(self.repo, **load_args, **quant_args, cache_dir=shared.opts.hfcache_dir)
            elif self.val is not None and len(self.val) > 0:
                debug_log(f'Model load component: default="{self.val}" args={load_args} quant={quant_type}')
                return self.cls.from_pretrained(self.val, **load_args, **quant_args, cache_dir=shared.opts.hfcache_dir)
            else:
                debug_log(f'Model load component: name="{self.name}" cls={self.cls} no handler')
                return None
        except Exception as e:
            log.error(f'Model load component: name="{self.name}" {e}')
            errors.display(e, 'Model load component')
        return None


def create_ui(gr_status, gr_file):
    def get_components(cls):
        if cls is None:
            return []
        signature = inspect.signature(cls.__init__, follow_wrapped=True)
        components.clear()
        for param in signature.parameters.values():
            if param.name == 'self' or param.name == 'args' or param.name == 'kwargs':
                continue
            component = Component(param)
            debug_log(f'Model component: {str(component)}')
            components.append(component)
        return components

    def get_model(model):
        if model == 'Current':
            cls = shared.sd_model.__class__
        else:
            cls = shared_items.pipelines.get(model, None)
        if cls is None:
            cls = diffusers.AutoPipelineForText2Image
        name = cls.__name__
        repo = shared_items.get_repo(name) or shared_items.get_repo(model)
        link = f'Link<br><br><a href="https://huggingface.co/{repo}" target="_blank">{repo}</a>' if repo else ''
        get_components(cls)
        dataframes = [c.dataframe() for c in components]
        log.debug(f'Model select: name="{model}" cls={name} repo="{repo}" link={link} components={len(components)}')
        return [name, repo, link, dataframes]

    def update_component(dataframes):
        for df in dataframes:
            c = [x for x in components if x.id == df[0]]
            if len(c) != 1:
                continue
            c = c[0]
            c.local = df[5].strip()
            c.remote = df[6].strip()
            c.dtype = df[7]
            c.quant = df[8]
            if c.remote and len(c.remote) > 0:
                c.repo, c.subfolder, c.local, c.download = process_huggingface_url(c.remote)

    # TODO loader: load receipe
    def load_receipe(file_select):
        if file_select is not None and 'name' in file_select:
            fn = file_select['name']
            log.debug(f'Load receipe: fn={fn}')
        return ['Load receipe not implemented yet', gr.update(label='Receipe .json file', file_types=['json'], visible=True)]

    # TODO loader: save receipe
    def save_receipe(model: str, repo: str):
        receipe = {
            'model': model,
            'repo': repo,
            'components': []
        }
        for c in components:
            if c.loadable:
                receipe['components'].append(c.save())
        # with open('/tmp/receipe.json', 'w', encoding='utf8') as f:
        #    json.dump(receipe, f, indent=2)
        return 'Save receipe not implemented yet'

    with gr.Row():
        gr.HTML('<h2>&nbsp<a href="https://vladmandic.github.io/sdnext-docs/Loader" target="_blank">Custom model loader</a><br></h2>')
    with gr.Row():
        choices = list(shared_items.pipelines)
        choices = ['Current' if x.startswith('Custom') else x for x in choices]
        model = gr.Dropdown(label="Model type", choices=choices, value='Autodetect')
        cls = gr.Textbox(label="Model class", placeholder="Class name", interactive=False)
    with gr.Row():
        repo = gr.Textbox(label="Model repo", placeholder="Repo name", interactive=True)
        link = gr.HTML(value="")
    with gr.Row():
        headers =  ['ID', 'Name', 'Loadable', 'Default', 'Class', 'Local', 'Remote', 'Dtype', 'Quant']
        datatype = ['number', 'str', 'bool', 'str', 'str', 'str', 'str', 'str', 'bool']
        dataframes = gr.DataFrame(
            value=None,
            label=None,
            show_label=False,
            interactive=True,
            wrap=True,
            headers=headers,
            datatype=datatype,
            type='array',
            elem_id="model_loader_df",
        )
        dataframes.change(fn=update_component, inputs=[dataframes], outputs=[])

    model.change(get_model, inputs=[model], outputs=[cls, repo, link, dataframes])

    with gr.Row():
        btn_load_receipe = gr.Button(value="Load receipe")
        btn_save_receipe = gr.Button(value="Save receipe")
    with gr.Row():
        btn_load_model = gr.Button(value="Load model")
        btn_unload_model = gr.Button(value="Unload model")

    btn_load_receipe.click(fn=load_receipe, inputs=[gr_file], outputs=[gr_status, gr_file])
    btn_save_receipe.click(fn=save_receipe, inputs=[model, repo], outputs=[gr_status])
    btn_load_model.click(fn=load_model, inputs=[model, cls, repo, dataframes], outputs=[gr_status])
    btn_unload_model.click(fn=unload_model, inputs=[], outputs=[gr_status])
