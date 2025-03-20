import os
import sys
import site
import ctypes
import shutil
import zipfile
import urllib.request
from typing import Union
from modules import rocm


DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'cufft.dll': 'cufft64_10.dll',
    'cufftw.dll': 'cufftw64_10.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIPSDK_TARGETS = ['rocblas.dll', 'rocsolver.dll', 'hipfft.dll',]
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)

hipBLASLt_available = False
MIOpen_available = False

path = os.path.abspath(os.environ.get('ZLUDA', '.zluda'))
default_agent: Union[rocm.Agent, None] = None
hipBLASLt_enabled = False

nightly = os.environ.get("ZLUDA_NIGHTLY", "0") == "1"
skip_arch_test = os.environ.get("ZLUDA_SKIP_ARCH_TEST", "0") == "1"


def set_default_agent(agent: rocm.Agent):
    global default_agent # pylint: disable=global-statement
    default_agent = agent

    is_nightly = False
    try:
        nvcuda = ctypes.windll.LoadLibrary(os.path.join(path, 'nvcuda.dll'))
        nvcuda.zluda_get_nightly_flag.restype = ctypes.c_int
        nvcuda.zluda_get_nightly_flag.argtypes = []
        is_nightly = nvcuda.zluda_get_nightly_flag() == 1
    except Exception:
        pass

    global hipBLASLt_available, hipBLASLt_enabled # pylint: disable=global-statement
    hipBLASLt_available = is_nightly and os.path.exists(rocm.blaslt_tensile_libpath)
    hipBLASLt_enabled = hipBLASLt_available and os.path.exists(os.path.join(rocm.path, "bin", "hipblaslt.dll"))

    global MIOpen_available # pylint: disable=global-statement
    MIOpen_available = is_nightly and os.path.exists(os.path.join(rocm.path, "bin", "MIOpen.dll"))


def is_reinstall_needed() -> bool: # ZLUDA<3.8.7
    return not os.path.exists(os.path.join(path, 'cufftw.dll'))


def install() -> None:
    if os.path.exists(path):
        return

    platform = "windows"
    commit = os.environ.get("ZLUDA_HASH", "ae0540beb129ffd140226ce956b386619b38f84c")
    if nightly:
        platform = "nightly-" + platform
    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{commit}/ZLUDA-{platform}-rocm{rocm.version[0]}-amd64.zip', '_zluda')
    with zipfile.ZipFile('_zluda', 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if not info.is_dir():
                info.filename = os.path.basename(info.filename)
                archive.extract(info, path)
    os.remove('_zluda')


def uninstall() -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def set_blaslt_enabled(enabled: bool):
    global hipBLASLt_enabled # pylint: disable=global-statement
    hipBLASLt_enabled = enabled


def get_blaslt_enabled() -> bool:
    return hipBLASLt_enabled


def link_or_copy(src: os.PathLike, dst: os.PathLike):
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copyfile(src, dst)


def make_copy() -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(path, v)):
            link_or_copy(os.path.join(path, k), os.path.join(path, v))

    if hipBLASLt_enabled and not os.path.exists(os.path.join(path, 'cublasLt64_11.dll')):
        link_or_copy(os.path.join(path, 'cublasLt.dll'), os.path.join(path, 'cublasLt64_11.dll'))

    if MIOpen_available and not os.path.exists(os.path.join(path, 'cudnn64_9.dll')):
        link_or_copy(os.path.join(path, 'cudnn.dll'), os.path.join(path, 'cudnn64_9.dll'))


def load() -> None:
    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"
    os.environ["ZLUDA_NVRTC_LIB"] = os.path.join([v for v in site.getsitepackages() if v.endswith("site-packages")][0], "torch", "lib", "nvrtc64_112_0.dll")

    for v in HIPSDK_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(path, v))

    if hipBLASLt_enabled:
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "0")
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', 'hipblaslt.dll'))
        ctypes.windll.LoadLibrary(os.path.join(path, 'cublasLt64_11.dll'))
    else:
        os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

    if MIOpen_available:
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', 'MIOpen.dll'))
        ctypes.windll.LoadLibrary(os.path.join(path, 'cudnn64_9.dll'))

    def conceal():
        import torch # pylint: disable=unused-import
        platform = sys.platform
        sys.platform = ""
        from torch.utils import cpp_extension
        sys.platform = platform
        cpp_extension.IS_WINDOWS = platform == "win32"
        cpp_extension.IS_MACOS = False
        cpp_extension.IS_LINUX = platform.startswith('linux')
        def _join_rocm_home(*paths) -> str:
            return os.path.join(cpp_extension.ROCM_HOME, *paths)
        cpp_extension._join_rocm_home = _join_rocm_home # pylint: disable=protected-access
    rocm.conceal = conceal
