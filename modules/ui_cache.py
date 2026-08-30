from functools import partial
import gradio as gr
import diffusers.hooks # pylint: disable=unused-import
from modules import ui_common


_stored_args = {}
methods = {
    'None': {},
    'Context Parallel': {},
    'Faster Cache': {},
    'First Block Cache': {},
    'Layer Skip': {},
    'Mag Cache': {},
    'Pyramid Attention Broadcast': {},
    'TaylorSeer Cache': {},
    'Text KV Cache': {},
}


def get_modular_args():
    return _stored_args


def get_cache_methods():
    from modules.processing_helpers import is_modular
    if is_modular():
        return list(methods.keys())
    return ['None']


def create_cache_inputs(tab):
    with gr.Accordion(open=False, label='Cache', elem_id=f"{tab}_cache", elem_classes=["small-accordion"]):
        with gr.Group():
            with gr.Row(elem_id=f"{tab}_cache_row", elem_classes=['flexbox']):
                cache_name = gr.Dropdown(choices=get_cache_methods(), value='None', label='Method', elem_id=f"{tab}_cache")
                _cache_check = ui_common.create_refresh_button(cache_name, get_cache_methods)

            acc_context_parallel = gr.Accordion(open=True, label='Context Parallel', elem_classes=["small-accordion"], visible=False)
            with acc_context_parallel:
                gr.HTML(value="<p>TODO: Context Parallel</p>")
                args_context_parallel = []

            acc_faster_cache = gr.Accordion(open=True, label='Faster Cache', elem_classes=["small-accordion"], visible=False)
            with acc_faster_cache:
                gr.HTML(value="<p>TODO: Faster Cache</p>")
                args_faster_cache = []

            acc_first_block_cache = gr.Accordion(open=True, label='First Block Cache', elem_classes=["small-accordion"], visible=False)
            with acc_first_block_cache:
                gr.HTML(value="<p>TODO: First Block Cache</p>")
                args_first_block_cache = []

            acc_layer_skip = gr.Accordion(open=True, label='Layer Skip', elem_classes=["small-accordion"], visible=False)
            with acc_layer_skip:
                gr.HTML(value="<p>TODO: Layer Skip</p>")
                args_layer_skip = []

            acc_mag_cache = gr.Accordion(open=True, label='Mag Cache', elem_classes=["small-accordion"], visible=False)
            with acc_mag_cache:
                gr.HTML(value="<p>TODO: Mag Cache</p>")
                args_mag_cache = []

            acc_pyramid_attention_broadcast = gr.Accordion(open=True, label='Pyramid Attention Broadcast', elem_classes=["small-accordion"], visible=False)
            with acc_pyramid_attention_broadcast:
                gr.HTML(value="<p>TODO: Pyramid Attention Broadcast</p>")
                args_pyramid_attention_broadcast = []

            acc_taylorseer_cache = gr.Accordion(open=True, label='TaylorSeer Cache', elem_classes=["small-accordion"], visible=False)
            with acc_taylorseer_cache:
                gr.HTML(value="<p>TODO: TaylorSeer Cache</p>")
                args_taylorseer_cache = []

            acc_text_kv_cache = gr.Accordion(open=True, label='Text KV Cache', elem_classes=["small-accordion"], visible=False)
            with acc_text_kv_cache:
                gr.HTML(value="<p>TODO: Text KV Cache</p>")
                args_text_kv_cache = []

            def adv_visibility(guidance_name):
                _stored_args['cache_name'] = guidance_name
                return [
                    gr.update(visible=guidance_name == 'Context Parallel'),
                    gr.update(visible=guidance_name == 'Faster Cache'),
                    gr.update(visible=guidance_name == 'First Block Cache'),
                    gr.update(visible=guidance_name == 'Layer Skip'),
                    gr.update(visible=guidance_name == 'Mag Cache'),
                    gr.update(visible=guidance_name == 'Pyramid Attention Broadcast'),
                    gr.update(visible=guidance_name == 'TaylorSeer Cache'),
                    gr.update(visible=guidance_name == 'Text KV Cache')
                ]
            cache_name.change(fn=adv_visibility,
                              inputs=[cache_name],
                              outputs=[acc_context_parallel, acc_faster_cache, acc_first_block_cache, acc_layer_skip, acc_mag_cache, acc_pyramid_attention_broadcast, acc_taylorseer_cache, acc_text_kv_cache],
                             )

    modular_args = args_context_parallel + args_faster_cache + args_first_block_cache + args_layer_skip + args_mag_cache + args_pyramid_attention_broadcast + args_taylorseer_cache + args_text_kv_cache
    def update_stored(component, name):
        _stored_args[name] = component
    for component in modular_args:
        label = getattr(component, 'label', None)
        value = getattr(component, 'value', None)
        name = label.lower().replace(' ', '_') if label is not None else None
        _stored_args[name] = value
        component.change(fn=partial(update_stored, name=name), inputs=[component], outputs=[])

    return [cache_name]
