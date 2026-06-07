import os
import platform
from modules.logger import log


def cleanup():
    if os.environ.get('SD_PLATFORM_DEBUG', None) is None:
        return
    if platform.system() == "Linux":
        log.warning(f'Platform: {platform.system()} cleanup')
        from modules.platform_linux import LinuxUtils
        LinuxUtils.advise_mmap()
        LinuxUtils.release_mmap()
        LinuxUtils.advise_cache()
        LinuxUtils.malloc_trim()
        LinuxUtils.get_smaps()
        LinuxUtils.get_status()
    else:
        log.warning(f'Platform: {platform.system()} not supported')
