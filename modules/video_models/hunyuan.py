from modules import shared


def load(selected):
    msg = f'Video load: model="{selected.name}" repo="{selected.repo}" dit="{selected.dit}"'
    shared.log.info(msg)
    return msg


def generate(*args, **kwargs):
    # TODO hunyuanvideo: check if loaded
    shared.log.debug(f'Video generate: args={args} kwargs={kwargs}')
    return [], '', '', 'TBD'
