# compatibility with extensions that import scripts directly
from modules import scripts_manager
from modules.scripts_manager import * # noqa: F403 # pylint: disable=wildcard-import


scripts_txt2img = None
scripts_img2img = None
scripts_control = None
scripts_current = None
scripts_postproc = None


def register_runners():
    global scripts_txt2img, scripts_img2img, scripts_control, scripts_current, scripts_postproc
    scripts_txt2img = scripts_manager.scripts_txt2img
    scripts_img2img = scripts_manager.scripts_img2img
    scripts_control = scripts_manager.scripts_control
    scripts_current = scripts_manager.scripts_current
    scripts_postproc = scripts_manager.scripts_postproc
