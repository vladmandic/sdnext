import os
import sys
import ssl
import site
import ctypes
import shutil
import zipfile
import urllib.request
from installer import args
from modules.logger import log
from modules import rocm


DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'cufft.dll': 'cufft64_10.dll',
    'cufftw.dll': 'cufftw64_10.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIPSDK_TARGETS = ['rocblas.dll', 'rocsolver.dll', 'rocsparse.dll', 'hipfft.dll',]

MIOpen_enabled = False

path = os.path.abspath(os.environ.get('ZLUDA', '.zluda'))
default_agent: rocm.Agent | None = None
hipBLASLt_enabled = False


class ZLUDAResult(ctypes.Structure):
    _fields_ = [
        ('return_code', ctypes.c_int),
        ('value', ctypes.c_ulonglong),
    ]


class ZLUDALibrary:
    internal: ctypes.CDLL

    def __init__(self, internal: ctypes.CDLL):
        self.internal = internal


class Core(ZLUDALibrary):
    def __init__(self, internal: ctypes.CDLL):
        internal.zluda_get_hip_object.restype = ZLUDAResult
        internal.zluda_get_hip_object.argtypes = [ctypes.c_void_p, ctypes.c_int]

        try:
            internal.zluda_get_nightly_flag.restype = ctypes.c_int
            internal.zluda_get_nightly_flag.argtypes = []
        except AttributeError:
            internal.zluda_get_nightly_flag = lambda: 0

        super().__init__(internal)

    def to_hip_stream(self, zluda_object: ctypes.c_void_p):
        return self.internal.zluda_get_hip_object(zluda_object, 1).value

    def get_nightly_flag(self) -> int:
        return self.internal.zluda_get_nightly_flag()


core = None
ml = None


def set_default_agent(agent: rocm.Agent):
    global default_agent # pylint: disable=global-statement
    default_agent = agent


def is_reinstall_needed() -> bool: # ZLUDA<3.9.4
    return os.path.exists(os.path.join(path, 'cudart.dll'))


def install():
    if os.path.exists(path):
        return

    platform = "windows"
    commit = os.environ.get("ZLUDA_HASH", "5e717459179dc272b7d7d23391f0fad66c7459cf")
    if os.environ.get("ZLUDA_NIGHTLY", "0") == "1":
        log.warning("Environment variable 'ZLUDA_NIGHTLY' will be removed. Please use command-line argument '--use-nightly' instead.")
        args.use_nightly = True
    if args.use_nightly:
        platform = "nightly-" + platform
    log.debug(f'Install ZLUDA: rocm={rocm.version} platform={platform} commit={commit}')
    ssl._create_default_https_context = ssl._create_unverified_context # pylint: disable=protected-access
    try:
        urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{commit}/ZLUDA-{platform}-rocm{rocm.version[0]}-amd64.zip', '_zluda')
        if not os.path.exists('_zluda'):
            raise RuntimeError('ZLUDA download failed')
        with zipfile.ZipFile('_zluda', 'r') as archive:
            infos = archive.infolist()
            for info in infos:
                if not info.is_dir():
                    info.filename = os.path.basename(info.filename)
                    archive.extract(info, path)
    except Exception as e:
        raise RuntimeError(f'Install ZLUDA: {e}') from e
    finally:
        if os.path.exists('_zluda'):
            os.remove('_zluda')


def uninstall():
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


def load():
    assert isinstance(rocm.environment, rocm.ROCmEnvironment)
    global core, ml, hipBLASLt_enabled, MIOpen_enabled # pylint: disable=global-statement
    core = Core(ctypes.windll.LoadLibrary(os.path.join(path, 'nvcuda.dll')))
    ml = ZLUDALibrary(ctypes.windll.LoadLibrary(os.path.join(path, 'nvml.dll')))
    is_nightly = core.get_nightly_flag() == 1
    hipBLASLt_enabled = is_nightly and os.path.exists(rocm.blaslt_tensile_libpath) and os.path.exists(os.path.join(rocm.environment.path, "bin", "hipblaslt.dll")) and default_agent is not None and default_agent.blaslt_supported
    MIOpen_enabled = is_nightly and os.path.exists(os.path.join(rocm.environment.path, "bin", "MIOpen.dll"))

    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(path, v)):
            link_or_copy(os.path.join(path, k), os.path.join(path, v))

    if hipBLASLt_enabled and not os.path.exists(os.path.join(path, 'cublasLt64_11.dll')):
        link_or_copy(os.path.join(path, 'cublasLt.dll'), os.path.join(path, 'cublasLt64_11.dll'))

    if MIOpen_enabled and not os.path.exists(os.path.join(path, 'cudnn64_9.dll')):
        link_or_copy(os.path.join(path, 'cudnn.dll'), os.path.join(path, 'cudnn64_9.dll'))

    log.info(f"ZLUDA load: path='{path}' nightly={bool(core.get_nightly_flag())}")

    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"
    os.environ["ZLUDA_NVRTC_LIB"] = os.path.join([v for v in site.getsitepackages() if v.endswith("site-packages")][0], "torch", "lib", "nvrtc64_112_0.dll")

    for v in HIPSDK_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(rocm.environment.path, 'bin', v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(path, v))

    if hipBLASLt_enabled:
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "0")
        ctypes.windll.LoadLibrary(os.path.join(rocm.environment.path, 'bin', 'hipblaslt.dll'))
        ctypes.windll.LoadLibrary(os.path.join(path, 'cublasLt64_11.dll'))
    else:
        os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

    if MIOpen_enabled:
        ctypes.windll.LoadLibrary(os.path.join(rocm.environment.path, 'bin', 'MIOpen.dll'))
        ctypes.windll.LoadLibrary(os.path.join(path, 'cudnn64_9.dll'))

    def postinstall():
        import torch
        torch.version.hip = rocm.version

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
    rocm.postinstall = postinstall

    from modules.zluda import zluda_init
    rocm.rocm_init = zluda_init
