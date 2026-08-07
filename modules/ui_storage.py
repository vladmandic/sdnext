import gradio as gr


def create_ui():
    types = ['All', 'Images', 'Videos', 'Models', 'Data', 'Cache', 'Code', 'Other']
    with gr.Row():
        btn_refresh = gr.Button("Calculate", elem_id='btn_storage_refresh')
        storage_type = gr.Dropdown(label="Storage type", elem_id='storage_type', choices=types, value=[types[0]], multiselect=True)
    with gr.Row():
        _storage_table = gr.HTML('', elem_id='storage_table')
    with gr.Row():
        _storage_timeline = gr.HTML('', elem_id='storage_timeline')
    btn_refresh.click(_js='refreshStorage', fn=None, inputs=[storage_type], outputs=[], show_progress='full')
