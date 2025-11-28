import gradio as gr


def create_ui():
    with gr.Row():
        btn_refresh = gr.Button("Refresh", elem_id='btn_history_refresh')
    with gr.Row():
        _history_table = gr.HTML('', elem_id='history_table')
    with gr.Row():
        _history_timeline = gr.HTML('', elem_id='history_timeline')
    btn_refresh.click(_js='refreshHistory', fn=None, inputs=[], outputs=[], show_progress='hidden')
