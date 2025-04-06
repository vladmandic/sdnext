import os
import inspect
import gradio as gr
from modules import shared, shared_items


debug_enabled = os.environ.get('SD_LOAD_DEBUG', None)
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None


class Component():
    def __init__(self, param):
        self.name = param.name
        self.cls = param.annotation
        self.str = str(param.annotation)
        self.val = param.default if param.default is not inspect.Parameter.empty else None
        self.enum = None
        if self.cls in [str, int, float, bool]:
            self.type = 'variable'
        elif 'enum' in self.str:
            self.type = 'enum'
            self.enum = [v.name for v in self.cls]
        elif inspect.isclass(param.annotation):
            self.type = 'class'
        elif inspect.ismodule(param.annotation):
            self.type = 'module'
        elif inspect.isfunction(param.annotation):
            self.type = 'function'
        elif 'typing.Optional' in self.str:
            self.type = 'optional'
            self.cls = param.annotation.__args__[0]
            self.str = str(self.cls)
            self.val = None
        else:
            self.type = 'unknown'
        self.loadable = self.type = 'class' and hasattr(self.cls, 'from_pretrained')

    def __str__(self):
        return f'name="{self.name}" type={self.type} cls={self.cls} str={self.str} val={self.val} loadable={self.loadable} enum={self.enum}'

    def dataframe(self):
        return [self.name, self.loadable, self.val, self.str, self.enum]


def create_ui():
    def get_components(cls):
        if cls is None:
            return []
        signature = inspect.signature(cls.__init__, follow_wrapped=True)
        components = []
        for param in signature.parameters.values():
            if param.name == 'self':
                continue
            component = Component(param)
            debug_log(f'Component: {str(component)}')
            components.append(component.dataframe())
        return components

    def get_model(model):
        cls = shared_items.pipelines.get(model, None)
        name = cls.__name__
        repo = shared_items.get_repo(name) or shared_items.get_repo(model)
        link = f'Link<br><br><a href="https://huggingface.co/{repo}" target="_blank">{repo}</a>' if repo else ''
        components = get_components(cls)
        shared.log.debug(f'Model select: name="{model}" cls={name} repo={repo} link={link} components={len(components)}')
        return [name, repo, link, components]

    with gr.Row():
        model = gr.Dropdown(label="Model type", choices=['Select'] + list(shared_items.pipelines), value='Select')
        cls = gr.Textbox(label="Model class", placeholder="Class name", interactive=False)
    with gr.Row():
        repo = gr.Textbox(label="Model repo", placeholder="Repo name", interactive=True)
        link = gr.HTML(value="", interactive=False)
    with gr.Row():
        headers = ['Name', 'Loadable', 'Value', 'Class', 'Choices']
        datatype = ['str', 'bool', 'str', 'str', 'str']
        components = gr.DataFrame(
            value=None,
            label=None,
            show_label=False,
            interactive=True,
            wrap=True,
            headers=headers,
            datatype=datatype,
            max_rows=None,
            max_cols=None,
            # row_count=(20, 'fixed'),
            # col_count=(5, 'fixed'),
            type='array',
            elem_id="model_loader_df",
        )

    model.change(get_model, inputs=[model], outputs=[cls, repo, link, components])

    gr.Markdown("""
        - Model repo is required to access base model config
        - Default model repo is provided for common models
        - Model repo can be overriden to any valid repo on huggingface
        - Any loadable model component without set value will be loaded from default repo
        - Any loadable model component with set value will be loaded from that value
        - Value can be local path to safetensors file or path on huggingface
    """, elem_id="model_loader_md")

    with gr.Row():
        btn_load_receipe = gr.Button(value="Load receipe") # pylint: disable=unused-variable
        btn_save_receipe = gr.Button(value="Save receipe") # pylint: disable=unused-variable
    with gr.Row():
        btn_load_model = gr.Button(value="Load model") # pylint: disable=unused-variable
        btn_unload_model = gr.Button(value="Unload model") # pylint: disable=unused-variable
