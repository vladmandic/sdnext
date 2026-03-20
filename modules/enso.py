import os
from installer import log, git, run_extension_installer
from modules.paths import extensions_dir


ENSO_REPO = "https://github.com/CalamitousFelicitousness/enso.git"
# ENSO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extensions-builtin", "enso")
ENSO_DIR = os.path.join(extensions_dir, "enso")


def install():
    """Clone the Enso frontend repo if not already present."""
    if os.path.isdir(ENSO_DIR):
        return
    log.info(f'Enso: folder="{ENSO_DIR}" installing')
    git(f'clone "{ENSO_REPO}" "{ENSO_DIR}"')
    run_extension_installer(ENSO_DIR)


def update():
    """Pull latest changes for the Enso frontend repo."""
    if not os.path.isdir(ENSO_DIR):
        return
    log.info(f'Enso: folder="{ENSO_DIR}" updating')
    git('pull', folder=ENSO_DIR)
    run_extension_installer(ENSO_DIR)
