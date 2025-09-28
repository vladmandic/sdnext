import os
import sys
import ctypes
import shutil
import subprocess
from typing import Union, List
from enum import Enum
from functools import wraps


def resolve_link(path_: str) -> str:
    if not os.path.islink(path_):
        return path_
    return resolve_link(os.readlink(path_))


def dirname(path_: str, r: int = 1) -> str:
    for _ in range(0, r):
        path_ = os.path.dirname(path_)
    return path_


def spawn(command: str, cwd: os.PathLike = '.') -> str:
    process = subprocess.run(command, cwd=cwd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    def __init__(self):
        import _rocm_sdk_core
        if sys.platform == "win32":
            path = os.path.join(_rocm_sdk_core.__path__[0], "bin", "amdhip64_7.dll")
        else:
            raise NotImplementedError
        # This library will be loaded/used by PyTorch. So it won't make conflicts.
        self.hip = ctypes.CDLL(path)


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

    def __init__(self, name: str):
        self.name = name
        self.gfx_version = Agent.parse_gfx_version(name)
        if self.gfx_version > 0x1000:
            self.arch = MicroArchitecture.RDNA
        elif self.gfx_version in (0x908, 0x90a, 0x942,):
            self.arch = MicroArchitecture.CDNA
        else:
            self.arch = MicroArchitecture.GCN
        self.is_apu = (self.gfx_version & 0xFFF0 == 0x1150) or self.gfx_version in (0x801, 0x902, 0x90c, 0x1013, 0x1033, 0x1035, 0x1036, 0x1103,)
        self.blaslt_supported = os.path.exists(os.path.join(blaslt_tensile_libpath, f"Kernels.so-000-{name}.hsaco" if sys.platform == "win32" else f"extop_{name}.co"))

    @property
    def therock(self) -> str:
        if (self.gfx_version & 0xFFF0) == 0x1100:
            return "gfx110X-dgpu"
        if self.gfx_version == 0x1151:
            return "gfx1151"
        if (self.gfx_version & 0xFFF0) == 0x1200:
            return "gfx120X-all"
        if (self.gfx_version & 0xFFF0) == 0x940:
            return "gfx94X-dcgpu"
        if self.gfx_version == 0x950:
            return "gfx950-dcgpu"
        raise Exception(f"Unsupported GPU architecture: {self.name}")

    def get_gfx_version(self) -> Union[str, None]:
        if self.gfx_version >= 0x1100 and self.gfx_version < 0x1200:
            return "11.0.0"
        elif self.gfx_version != 0x1030 and self.gfx_version >= 0x1000 and self.gfx_version < 0x1100:
            # gfx1010 users had to override gfx version to 10.3.0 in Linux
            # it is unknown whether overriding is needed in ZLUDA
            return "10.3.0"
        return None


def find() -> Union[Environment, None]:
    try: # TheRock
        import _rocm_sdk_core # pylint: disable=unused-import
        return PythonPackageEnvironment()
    except ImportError:
        pass

    # system-wide installation
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
    else:
        # If rocm-sdk package is installed, the hip library may be used by PyTorch.
        version = ctypes.c_int()
        environment.hip.hipRuntimeGetVersion(ctypes.byref(version))
        major = version.value // 10000000
        minor = (version.value // 100000) % 100
        #patch = version.value % 100000
        return f"{major}.{minor}"


def get_flash_attention_command(agent: Agent) -> str:
    default = "git+https://github.com/ROCm/flash-attention"
    if agent.gfx_version >= 0x1100 and agent.gfx_version < 0x1200 and os.environ.get("FLASH_ATTENTION_USE_TRITON_ROCM", "false").lower() != "true":
        # use the navi_rotary_fix fork because the original doesn't support rotary_emb for transformers
        # original: "git+https://github.com/ROCm/flash-attention@howiejay/navi_support"
        default = "git+https://github.com/Disty0/flash-attention@navi_rotary_fix"
    return "--no-build-isolation " + os.environ.get("FLASH_ATTENTION_PACKAGE", default)


if sys.platform == "win32":
    def get_agents() -> List[Agent]:
        if isinstance(environment, ROCmEnvironment):
            out = spawn("amdgpu-arch", cwd=os.path.join(environment.path, 'bin'))
        else:
            # Assume that amdgpu-arch is in PATH (venv/Scripts/amdgpu-arch.exe)
            out = spawn("amdgpu-arch")
        out = out.strip()
        return [Agent(x.split(' ')[-1].strip()) for x in out.split("\n")]

    def driver_get_agents() -> List[Agent]:
        # unsafe and experimental feature
        from modules import windows_hip_ffi
        hip = windows_hip_ffi.HIP()
        count = hip.get_device_count()
        agents = [None] * count
        for i in range(count):
            prop = hip.get_device_properties(i)
            name = prop.gcnArchName.decode('utf-8').strip('\x00')
            agents[i] = Agent(name)
        del hip
        return agents

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

            cholesky_ex_gpu = torch.linalg.cholesky_ex
            @wraps(cholesky_ex_gpu)
            def cholesky_ex(A: torch.Tensor, upper=False, check_errors=False, out=None) -> torch.return_types.linalg_cholesky_ex:
                if A.device.type != 'cpu':
                    return cholesky_ex_gpu(A, upper=upper, check_errors=check_errors, out=out)

                assert not check_errors
                L = torch.from_numpy(np.linalg.cholesky(A.numpy(), upper=upper))
                info = torch.tensor(0, dtype=torch.int32, device='cpu')
                if out is not None:
                    out[0].copy_(L)
                    out[1].copy_(info)
                return torch.return_types.linalg_cholesky_ex((L, info), {})
            torch.linalg.cholesky_ex = cholesky_ex

            cholesky_gpu = torch.linalg.cholesky
            @wraps(cholesky_gpu)
            def cholesky(A: torch.Tensor, upper=False, out=None) -> torch.Tensor:
                if A.device.type != 'cpu':
                    return cholesky_gpu(A, upper=upper, out=out)
                L = torch.from_numpy(np.linalg.cholesky(A.numpy(), upper=upper))
                if out is not None:
                    out.copy_(L)
                return L
            torch.linalg.cholesky = cholesky
        except Exception as e:
            return False, e
        return True, None

    is_wsl: bool = False
else:
    def get_agents() -> List[Agent]:
        try:
            agents = spawn("rocm_agent_enumerator").split("\n")
            agents = [x for x in agents if x and x != 'gfx000']
        except Exception: # old version of ROCm WSL doesn't have rocm_agent_enumerator
            agents = spawn("rocminfo").split("\n")
            agents = [x.strip().split(" ")[-1] for x in agents if x.startswith('  Name:') and "CPU" not in x]
        return [Agent(x) for x in agents]

    def postinstall():
        if is_wsl:
            try:
                if shutil.which("conda") is not None:
                    # Preload stdc++ library. This will bypass Anaconda stdc++ library.
                    load_library_global("/lib/x86_64-linux-gnu/libstdc++.so.6")
                # Preload rocr4wsl. The user don't have to replace the library file.
                load_library_global("/opt/rocm/lib/libhsa-runtime64.so")
            except OSError:
                pass

    def rocm_init():
        return True, None

    is_wsl: bool = os.environ.get('WSL_DISTRO_NAME', 'unknown' if spawn('wslpath -w /') else None) is not None
environment = None
blaslt_tensile_libpath = ""
is_installed = False
version = None

def refresh():
    global environment, blaslt_tensile_libpath, is_installed, version
    environment = find()
    if environment is not None:
        if isinstance(environment, ROCmEnvironment):
            blaslt_tensile_libpath = os.environ.get("HIPBLASLT_TENSILE_LIBPATH", os.path.join(environment.path, "bin" if sys.platform == "win32" else "lib", "hipblaslt", "library"))
        is_installed = True
        version = get_version()
refresh()
