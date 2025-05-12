import time
import gradio as gr
from modules import shared


def refresh():
    def ts(t):
        try:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
        except Exception:
            return ''

    items = []
    for item in shared.state.state_history:
        items.append([
            item['id'],
            item['job'],
            item['op'],
            ts(item['start']),
            ts(item['end']),
            len(item['outputs']),
        ])
    shared.log.info(f"History: records={len(items)}")
    if len(items) == 0:
        items = None
    return gr.Dataframe.update(value=items)


def select(evt: gr.SelectData, _data):
    if evt.index is None or len(evt.index) == 0:
        return []
    row = evt.index[0]
    item = shared.state.state_history[row] if row < len(shared.state.state_history) else {}
    shared.log.debug(f"History: select={row}:{len(shared.state.state_history)} {item}")
    files = item.get('outputs', [])
    return gr.Files.update(value=files)


def create_ui():
    with gr.Row():
        btn_refresh = gr.Button("Refresh", elem_id='btn_history_refresh')
    with gr.Row():
        history_table = gr.DataFrame(
            value=None,
            headers=['ID', 'Job', 'Op', 'Start', 'End', 'Outputs'],
            label='History data',
            show_label=True,
            interactive=False,
            wrap=True,
            overflow_row_behaviour='paginate',
            max_rows=50,
            elem_id='history_table',
        )
    with gr.Row():
        history_files = gr.Files(
            label="Task files",
            interactive=False,
            elem_id='history_files',
        )
    btn_refresh.click(fn=refresh, inputs=[], outputs=[history_table])
    history_table.select(fn=select, inputs=[history_table], outputs=[history_files])
