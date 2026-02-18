import os
import sys
import glob
import ctypes
import shutil
import subprocess
from types import ModuleType
from typing import Union, overload, TYPE_CHECKING
from enum import Enum
from functools import wraps
if TYPE_CHECKING:
    import torch


rocm_sdk: Union[ModuleType, None] = None


def resolve_link(path_: str) -> str:
    if not os.path.islink(path_):
        return path_
    return resolve_link(os.readlink(path_))


def dirname(path_: str, r: int = 1) -> str:
    for _ in range(0, r):
        path_ = os.path.dirname(path_)
    return path_


def spawn(command: Union[str, list[str]], cwd: os.PathLike = '.') -> str:
    process = subprocess.run(command, cwd=cwd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return process.stdout.decode(encoding="utf8", errors="ignore")


def load_library_global(path_: str):
    ctypes.CDLL(path_, mode=ctypes.RTLD_GLOBAL)


class Environment:
    pass


# rocm is installed system-wide
class ROCmEnvironment(Environment):
    path: str

    def __init__(self, path: str):
        self.path = path


# rocm-sdk package is installed
class PythonPackageEnvironment(Environment):
    hip: ctypes.CDLL

    def __init__(self, rocm_sdk_module: ModuleType):
        spec = rocm_sdk_module._dist_info.ALL_PACKAGES['core'].get_py_package() # pylint: disable=protected-access
        lib = rocm_sdk_module._dist_info.ALL_LIBRARIES['amdhip64'] # pylint: disable=protected-access
        pattern = os.path.join(os.path.dirname(spec.origin), lib.windows_relpath if sys.platform == "win32" else lib.posix_relpath, lib.dll_pattern if sys.platform == "win32" else lib.so_pattern)
        candidates = glob.glob(pattern)
        if len(candidates) == 0:
            raise FileNotFoundError("Could not find amdhip64 in rocm-sdk package")
        self.hip = ctypes.CDLL(candidates[0])


class MicroArchitecture(Enum):
    GCN = "gcn"
    RDNA = "rdna"
    CDNA = "cdna"


class Agent:
    name: str
    gfx_version: int
    arch: MicroArchitecture
    is_apu: bool
    blaslt_supported: bool

    @staticmethod
    def parse_gfx_version(name: str) -> int:
        result = 0
        for i in range(3, len(name)):
            if name[i].isdigit():
                result *= 0x10
                result += ord(name[i]) - 48
                continue
            if name[i] in "abcdef":
                result *= 0x10
                result += ord(name[i]) - 87
                continue
            break
        return result

    @overload
    def __init__(self, name: str): ...
    @overload
    def __init__(self, device: 'torch.types.Device'): ...

    def __init__(self, arg):
        if isinstance(arg, str):
            name = arg
        else: # assume arg is device-like object
            import torch
            name = getattr(torch.cuda.get_device_properties(arg), "gcnArchName", "gfx0000")
        self.name = name.split(':')[0]
        self.gfx_version = Agent.parse_gfx_version(self.name)
        if self.gfx_version > 0x1000:
            self.arch = MicroArchitecture.RDNA
        elif self.gfx_version in (0x908, 0x90a, 0x942,):
            self.arch = MicroArchitecture.CDNA
        else:
            self.arch = MicroArchitecture.GCN
        self.is_apu = (self.gfx_version & 0xFFF0 == 0x1150) or self.gfx_version in (0x801, 0x902, 0x90c, 0x1013, 0x1033, 0x1035, 0x1036, 0x1103,)
        self.blaslt_supported = False if blaslt_tensile_libpath is None else os.path.exists(os.path.join(blaslt_tensile_libpath, f"TensileLibrary_lazy_{self.name}.dat"))

    def __str__(self) -> str:
        return self.name

    @property
    def therock(self) -> Union[str, None]:
        if (self.gfx_version & 0xFFF0) == 0x1200:
            return "v2/gfx120X-all"
        if (self.gfx_version & 0xFFF0) == 0x1100:
            return "v2/gfx110X-all"
        if self.gfx_version == 0x1150:
            return "v2-staging/gfx1150"
        if self.gfx_version == 0x1151:
            return "v2/gfx1151"
        if self.gfx_version == 0x1152:
            return "v2-staging/gfx1152"
        if self.gfx_version == 0x1153:
            return "v2-staging/gfx1153"
        if self.gfx_version in (0x1030, 0x1031, 0x1032, 0x1034,):
            return "v2-staging/gfx103X-dgpu"
        #if (self.gfx_version & 0xFFF0) == 0x1010:
        #    return "gfx101X-dgpu"
        #if (self.gfx_version & 0xFFF0) == 0x900:
        #    return "gfx90X-dcgpu"
        #if (self.gfx_version & 0xFFF0) == 0x940:
        #    return "gfx94X-dcgpu"
        #if self.gfx_version == 0x950:
        #    return "gfx950-dcgpu"
        return None

    def get_gfx_version(self) -> Union[str, None]:
        if self.gfx_version is None:
            return None
        if self.gfx_version >= 0x1100 and self.gfx_version < 0x1200:
            return "11.0.0"
        elif self.gfx_version != 0x1030 and self.gfx_version >= 0x1000 and self.gfx_version < 0x1100:
            # gfx1010 users had to override gfx version to 10.3.0 in Linux
            # it is unknown whether overriding is needed in ZLUDA
            return "10.3.0"
        return None


def find() -> Union[ROCmEnvironment, None]:
    hip_path = shutil.which("hipconfig")
    if hip_path is not None:
        return ROCmEnvironment(dirname(resolve_link(hip_path), 2))

    if sys.platform == "win32":
        hip_path = os.environ.get("HIP_PATH", None)
        if hip_path is not None:
            return ROCmEnvironment(hip_path)

        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        hip_path = rf'{program_files}\AMD\ROCm'
        if not os.path.exists(hip_path):
            return None

        class Version:
            major: int
            minor: int

            def __init__(self, string: str):
                self.major, self.minor = [int(v) for v in string.strip().split(".")]

            def __gt__(self, other):
                return self.major * 10 + other.minor > other.major * 10 + other.minor

            def __str__(self):
                return f"{self.major}.{self.minor}"

        latest = None
        versions = os.listdir(hip_path)
        for s in versions:
            item = None
            try:
                item = Version(s)
            except Exception:
                continue
            if latest is None:
                latest = item
                continue
            if item > latest:
                latest = item

        if latest is None:
            return None

        return ROCmEnvironment(os.path.join(hip_path, str(latest)))
    else:
        if not os.path.exists("/opt/rocm"):
            return None
        return ROCmEnvironment(resolve_link("/opt/rocm"))


def get_version() -> str:
    if isinstance(environment, ROCmEnvironment):
        # We don't load the hip library that will not be used by PyTorch.
        if sys.platform == "win32":
            # ROCm is system-wide installed. Assume the version is the folder name. (e.g. C:\Program Files\AMD\ROCm\6.4)
            # hipconfig requires Perl
            return os.path.basename(environment.path) or os.path.basename(os.path.dirname(environment.path))
        else:
            arr = spawn("hipconfig --version", cwd=os.path.join(environment.path, 'bin')).split(".")
            return f'{arr[0]}.{arr[1]}' if len(arr) >= 2 else None
    elif isinstance(environment, PythonPackageEnvironment):
        # If rocm-sdk package is installed, the hip library may be used by PyTorch.
        ver = ctypes.c_int()
        environment.hip.hipRuntimeGetVersion(ctypes.byref(ver))
        major = ver.value // 10000000
        minor = (ver.value // 100000) % 100
        #patch = version.value % 100000
        return f"{major}.{minor}"
    else:
        return None


def get_flash_attention_command(agent: Agent) -> str:
    default = "git+https://github.com/ROCm/flash-attention"
    if agent.gfx_version >= 0x1100 and agent.gfx_version < 0x1200 and os.environ.get("FLASH_ATTENTION_USE_TRITON_ROCM", "false").lower() != "true":
        # use the navi_rotary_fix fork because the original doesn't support rotary_emb for transformers
        # original: "git+https://github.com/ROCm/flash-attention@howiejay/navi_support"
        default = "git+https://github.com/Disty0/flash-attention@navi_rotary_fix"
    return "--no-build-isolation " + os.environ.get("FLASH_ATTENTION_PACKAGE", default)


def refresh():
    global rocm_sdk, environment, blaslt_tensile_libpath, is_installed, version # pylint: disable=global-statement
    try:
        import rocm_sdk
        environment = PythonPackageEnvironment(rocm_sdk)
        try:
            target_family = rocm_sdk._dist_info.determine_target_family() # pylint: disable=protected-access
            spec = rocm_sdk._dist_info.ALL_PACKAGES['libraries'].get_py_package(target_family) # pylint: disable=protected-access
            blaslt_tensile_libpath = os.path.join(os.path.dirname(spec.origin), "bin", "hipblaslt", "library")
        except Exception:
            blaslt_tensile_libpath = None
        spawn(["rocm-sdk", "init"])
    except ImportError:
        rocm_sdk = None
        environment = find()
        if environment is not None:
            blaslt_tensile_libpath = os.path.join(environment.path, "bin" if sys.platform == "win32" else "lib", "hipblaslt", "library")

    if environment is not None:
        blaslt_tensile_libpath = os.environ.get("HIPBLASLT_TENSILE_LIBPATH", blaslt_tensile_libpath)
        is_installed = True
        version = get_version()


if sys.platform == "win32":
    import tempfile

    def get_agents() -> list[Agent]:
        name = None
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            name = f.name
            f.write(CODE_AMDGPU_ARCH)
            f.flush()
        out = spawn([sys.executable, name])
        os.unlink(name)
        out = out.strip()
        if out == "":
            return []
        return [Agent(x.split(' ')[-1].strip()) for x in out.split("\n")]

    def postinstall():
        import torch
        if torch.version.hip is None:
            os.environ.pop("ROCM_HOME", None)
            os.environ.pop("ROCM_PATH", None)
            paths = os.environ["PATH"].split(";")
            paths_no_rocm = []
            for path_ in paths:
                if "rocm" not in path_.lower():
                    paths_no_rocm.append(path_)
            os.environ["PATH"] = ";".join(paths_no_rocm)
            return

    def rocm_init():
        try:
            import torch
            import numpy as np
            from installer import log
            from modules.devices import get_hip_agent
            from modules.rocm_triton_windows import apply_triton_patches

            build_targets = torch.cuda.get_arch_list()
            agents = get_agents()
            log.debug(f'ROCm: agents={agents}')
            if all(available.name not in build_targets for available in agents):
                log.warning('ROCm: torch-rocm is installed, but none of build targets are available')
                # use cpu instead of crashing
                torch.cuda.is_available = lambda: False

            agent = get_hip_agent()
            log.debug(f'ROCm: selected={agents}')
            if not agent.blaslt_supported:
                log.warning(f'ROCm: hipBLASLt unavailable agent={agent}')

            if sys.platform == "win32":
                apply_triton_patches()

            original_cholesky_ex = torch.linalg.cholesky_ex
            @wraps(original_cholesky_ex)
            def cholesky_ex(A: torch.Tensor, upper=False, check_errors=False, out=None) -> torch.return_types.linalg_cholesky_ex:
                assert not check_errors
                return_device = A.device
                L = torch.from_numpy(np.linalg.cholesky(A.to("cpu").numpy(), upper=upper)).to(return_device)
                info = torch.tensor(0, dtype=torch.int32, device=return_device)
                if out is not None:
                    out[0].copy_(L)
                    out[1].copy_(info)
                return torch.return_types.linalg_cholesky_ex((L, info), {})
            torch.linalg.cholesky_ex = cholesky_ex

            original_cholesky = torch.linalg.cholesky
            @wraps(original_cholesky)
            def cholesky(A: torch.Tensor, upper=False, out=None) -> torch.Tensor:
                return_device = A.device
                L = torch.from_numpy(np.linalg.cholesky(A.to("cpu").numpy(), upper=upper)).to(return_device)
                if out is not None:
                    out.copy_(L)
                return L
            torch.linalg.cholesky = cholesky
        except Exception as e:
            return False, e
        return True, None

    is_wsl: bool = False
else: # sys.platform != "win32"
    def get_agents() -> list[Agent]:
        try:
            _agents = spawn("rocm_agent_enumerator").split("\n")
            _agents = [x for x in _agents if x and x != 'gfx000']
        except Exception: # old version of ROCm WSL doesn't have rocm_agent_enumerator
            _agents = spawn("rocminfo").split("\n")
            _agents = [x.strip().split(" ")[-1] for x in _agents if x.startswith('  Name:') and "CPU" not in x]
        return [Agent(x) for x in _agents]

    def postinstall():
        if is_wsl:
            try:
                if shutil.which("conda") is not None:
                    # Preload stdc++ library. This will bypass Anaconda stdc++ library.
                    # (hsa-runtime64 depends on stdc++)
                    load_library_global("/lib/x86_64-linux-gnu/libstdc++.so.6")
                # Preload rocr4wsl. The user don't have to replace the library file.
                load_library_global("/opt/rocm/lib/libhsa-runtime64.so")
            except OSError:
                pass

    def rocm_init():
        try:
            import torch
            from installer import log
            from modules.devices import get_hip_agent

            agent = get_hip_agent()
            if not agent.blaslt_supported:
                log.debug(f'ROCm: hipBLASLt unavailable agent={agent}')
        except Exception as e:
            return False, e
        return True, None

    is_wsl: bool = os.environ.get('WSL_DISTRO_NAME', 'unknown' if spawn('wslpath -w /') else None) is not None

environment: Union[Environment, None] = None
blaslt_tensile_libpath: Union[str, None] = None
is_installed: bool = False
version: Union[str, None] = None
refresh()

# amdgpu-arch.exe written in Python
CODE_AMDGPU_ARCH = """
import os
import sys
import ctypes
import ctypes.wintypes
import contextlib
hipDeviceProp = ctypes.c_byte * 1472
@contextlib.contextmanager
def mute(fd):
    s = os.dup(fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(s, fd)
        os.close(s)
class HIP:
    def __init__(self):
        ctypes.windll.kernel32.LoadLibraryA.restype = ctypes.wintypes.HMODULE
        ctypes.windll.kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
        self.handle = None
        path = os.environ.get("windir", "C:\\\\Windows") + "\\\\System32\\\\amdhip64_7.dll"
        if not os.path.isfile(path):
            path = os.environ.get("windir", "C:\\\\Windows") + "\\\\System32\\\\amdhip64_6.dll"
        if not os.path.isfile(path):
            path = os.environ.get("windir", "C:\\\\Windows") + "\\\\System32\\\\amdhip64.dll"
        assert os.path.isfile(path)
        self.handle = ctypes.windll.kernel32.LoadLibraryA(path.encode('utf-8'))
        ctypes.windll.kernel32.GetLastError.restype = ctypes.wintypes.DWORD
        ctypes.windll.kernel32.GetLastError.argtypes = []
        assert ctypes.windll.kernel32.GetLastError() == 0
        ctypes.windll.kernel32.GetProcAddress.restype = ctypes.c_void_p
        ctypes.windll.kernel32.GetProcAddress.argtypes = [ctypes.wintypes.HMODULE, ctypes.c_char_p]
        hipInit = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint)(
            ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipInit"))
        with mute(sys.stdout.fileno()):
            hipInit(0)
        self.hipGetDeviceCount = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))(
            ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceCount"))
        self.hipGetDeviceProperties = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(hipDeviceProp), ctypes.c_int)(
            ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceProperties"))
    def get_device_count(self) -> int:
        count = ctypes.c_int()
        assert self.hipGetDeviceCount(ctypes.byref(count)) == 0
        return count.value
    def get_device_properties(self, device_id) -> bytes:
        prop = hipDeviceProp()
        assert self.hipGetDeviceProperties(ctypes.byref(prop), device_id) == 0
        return bytes(prop)
if __name__ == "__main__":
    hip = HIP()
    count = hip.get_device_count()
    archs: list[str | None] = [None] * count
    for i in range(count):
        prop = hip.get_device_properties(i)
        name = ""
        idx = 0
        while idx < len(prop):
            try:
                idx = prop.index(0x67, idx) + 1
            except ValueError:
                break
            if prop[idx] != 0x66:
                continue
            if prop[idx + 1] != 0x78:
                continue
            idx = idx + 2
            while prop[idx] != 0x00:
                c = prop[idx]
                idx += 1
                if (c < 0x30 or c > 0x39) and (c < 0x61 or c > 0x66):
                    name = ""
                    continue
                name += chr(c)
            break
        if name:
            archs[i] = "gfx" + name
    del hip
    for arch in archs:
        if arch is not None:
            print(arch)
"""
