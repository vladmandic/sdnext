import os
from installer import log, git


ENSO_REPO = "https://github.com/CalamitousFelicitousness/enso.git"
ENSO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extensions-builtin", "enso")


def install():
    """Clone the Enso frontend repo if not already present."""
    if os.path.isdir(ENSO_DIR):
        return
    log.info(f'Enso: installing to "{ENSO_DIR}"')
    git(f'clone "{ENSO_REPO}" "{ENSO_DIR}"')


def update():
    """Pull latest changes for the Enso frontend repo."""
    if not os.path.isdir(ENSO_DIR):
        return
    log.info('Enso: updating')
    git('pull', folder=ENSO_DIR)
