import sys
from enum import Enum
from modules.logger import log
from modules import devices


class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    MIGraphX = "MIGraphXExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"


EP_TO_NAME = {
    ExecutionProvider.CPU: "gpu-cpu", # ???
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-cuda", # test required
    ExecutionProvider.ROCm: "gpu-rocm", # test required
    ExecutionProvider.MIGraphX: "gpu-migraphx", # test required
    ExecutionProvider.OpenVINO: "gpu-openvino", # test required
}
TORCH_DEVICE_TO_EP = {
    "cpu": ExecutionProvider.CPU if devices.backend != "openvino" else ExecutionProvider.OpenVINO,
    "cuda": ExecutionProvider.CUDA,
    "xpu": ExecutionProvider.OpenVINO,
    "privateuseone": ExecutionProvider.DirectML,
    "meta": None,
}


try:
    import onnxruntime as ort
    available_execution_providers: list[ExecutionProvider] = ort.get_available_providers()
except Exception as e:
    log.error(f'ONNX import error: {e}')
    available_execution_providers = []
    ort = None


def get_default_execution_provider() -> ExecutionProvider:
    if devices.backend == "cpu":
        return ExecutionProvider.CPU
    elif devices.backend == "directml":
        return ExecutionProvider.DirectML
    elif devices.backend == "cuda":
        return ExecutionProvider.CUDA
    elif devices.backend == "rocm":
        return ExecutionProvider.ROCm
    elif devices.backend == "ipex" or devices.backend == "openvino":
        return ExecutionProvider.OpenVINO
    return ExecutionProvider.CPU


def get_execution_provider_options():
    from modules.shared import cmd_opts, opts
    execution_provider_options = { "device_id": int(cmd_opts.device_id or 0) }
    if opts.onnx_execution_provider == ExecutionProvider.ROCm:
        if ExecutionProvider.ROCm in available_execution_providers:
            execution_provider_options["tunable_op_enable"] = True
            execution_provider_options["tunable_op_tuning_enable"] = True
    elif opts.onnx_execution_provider == ExecutionProvider.OpenVINO:
        from modules.intel.openvino import get_device as get_raw_openvino_device
        device = get_raw_openvino_device()
        if "HETERO:" not in device:
            if opts.olive_float16:
                device = f"{device}_FP16"
            else:
                device = f"{device}_FP32"
        else:
            device = ""
            available_devices = opts.openvino_devices.copy()
            available_devices.remove("CPU")
            for hetero_device in available_devices:
                if opts.olive_float16:
                    device = f"{device},{hetero_device}_FP16"
                else:
                    device = f"{device},{hetero_device}_FP32"
            if "CPU" in opts.openvino_devices:
                if opts.olive_float16:
                    device = f"{device},CPU_FP16"
                else:
                    device = f"{device},CPU_FP32"
            device = f"HETERO:{device[1:]}"

        execution_provider_options["device_type"] = device
        del execution_provider_options["device_id"]
    return execution_provider_options


def get_provider() -> tuple:
    from modules.shared import opts
    return (opts.onnx_execution_provider, get_execution_provider_options(),)


def install_execution_provider(ep: ExecutionProvider):
    import importlib  # pylint: disable=deprecated-module
    from installer import installed, install, uninstall
    res = "<br><pre>"
    res += uninstall(["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-training", "onnxruntime-openvino"], quiet=True)
    packages = ["onnxruntime"] # Failed to load olive: cannot import name '__version__' from 'onnxruntime'
    if ep == ExecutionProvider.DirectML:
        packages.append("onnxruntime-directml")
    elif ep == ExecutionProvider.CUDA:
        packages.append("onnxruntime-gpu")
    elif ep == ExecutionProvider.ROCm:
        if "linux" not in sys.platform:
            log.warning("ROCMExecutionProvider is not supported on Windows.")
            return
        packages.append("--pre onnxruntime-training --index-url https://pypi.lsh.sh/60 --extra-index-url https://pypi.org/simple")
    elif ep == ExecutionProvider.OpenVINO:
        packages.append("openvino")
        packages.append("onnxruntime-openvino")
    log.info(f'ONNX install: {packages}')
    for package in packages:
        res += install(package)
    res += '</pre><br>'
    res += 'Server restart required'
    log.info("Server restart required")
    try:
        importlib.reload(ort)
    except Exception:
        pass
    return res
