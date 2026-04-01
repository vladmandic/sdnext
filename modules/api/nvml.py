try:
    from installer import install
    from modules.logger import log
except Exception:
    def install(*args, **kwargs): # pylint: disable=unused-argument
        pass
    import logging
    log = logging.getLogger(__name__)


nvml_initialized = False
warned = False


def warn_once(msg):
    global warned # pylint: disable=global-statement
    if not warned:
        log.error(msg)
        warned = True

def get_reason(val):
    throttle = {
        1: 'gpu idle',
        2: 'applications clocks setting',
        4: 'sw power cap',
        8: 'hw slowdown',
        16: 'sync boost',
        32: 'sw thermal slowdown',
        64: 'hw thermal slowdown',
        128: 'hw power brake slowdown',
        256: 'display clock setting',
    }
    reason = ', '.join([throttle[i] for i in throttle if i & val])
    return reason if len(reason) > 0 else 'ok'


def get_nvml():
    global nvml_initialized # pylint: disable=global-statement
    if warned:
        return []
    try:
        from modules.memstats import ram_stats
        if not nvml_initialized:
            install('nvidia-ml-py', quiet=True)
            import pynvml # pylint: disable=redefined-outer-name
            pynvml.nvmlInit()
            log.debug('NVML initialized')
            nvml_initialized = True
        else:
            import pynvml
        devices = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            dev = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(dev)
            except Exception:
                name = ''
            load = pynvml.nvmlDeviceGetUtilizationRates(dev)
            mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
            ram = ram_stats()
            data = {
                "CUDA": f'Version {pynvml.nvmlSystemGetCudaDriverVersion()} Compute {pynvml.nvmlDeviceGetCudaComputeCapability(dev)}',
                "Driver": pynvml.nvmlSystemGetDriverVersion(),
                "Hardware": f'VBIOS {pynvml.nvmlDeviceGetVbiosVersion(dev)} ROM {pynvml.nvmlDeviceGetInforomImageVersion(dev)}',
                "PCI link": f'Gen.{pynvml.nvmlDeviceGetCurrPcieLinkGeneration(dev)} x{pynvml.nvmlDeviceGetCurrPcieLinkWidth(dev)}',
                "Power": f'{round(pynvml.nvmlDeviceGetPowerUsage(dev)/1000, 2)} W / {round(pynvml.nvmlDeviceGetEnforcedPowerLimit(dev)/1000, 2)} W',
                "GPU clock": f'{pynvml.nvmlDeviceGetClockInfo(dev, 0)} Mhz / {pynvml.nvmlDeviceGetMaxClockInfo(dev, 0)} Mhz',
                "SM clock": f'{pynvml.nvmlDeviceGetClockInfo(dev, 1)} Mhz / {pynvml.nvmlDeviceGetMaxClockInfo(dev, 1)} Mhz',
                "VRAM clock": f'{pynvml.nvmlDeviceGetClockInfo(dev, 2)} Mhz / {pynvml.nvmlDeviceGetMaxClockInfo(dev, 2)} Mhz',
                "VRAM usage": f'{round(100 * mem.used / mem.total)}% | {round(mem.used / 1024 / 1024)} MB used | {round(mem.free / 1024 / 1024)} MB free | {round(mem.total / 1024 / 1024)} MB total',
                "RAM usage": f'{round(100 * ram["used"] / ram["total"])}% | {round(1024 * ram["used"])} MB used | {round(1024 * ram["free"])} MB free | {round(1024 * ram["total"])} MB total',
                "System load": f'GPU {load.gpu}% | VRAM {load.memory}% | Temp {pynvml.nvmlDeviceGetTemperature(dev, 0)}C | Fan {pynvml.nvmlDeviceGetFanSpeed(dev)}%',
                'State': get_reason(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(dev)),
            }
            chart = [load.memory, load.gpu]
            devices.append({
                'name': name,
                'data': data,
                'chart': chart,
            })
        # log.debug(f'nmvl: {devices}')
        return devices
    except Exception as e:
        warn_once(f'NVML: {e}')
        return []


if __name__ == '__main__':
    nvml_initialized = True
    import pynvml # pylint: disable=redefined-outer-name
    pynvml.nvmlInit()
    from rich import print as rprint
    for gpu in get_nvml():
        rprint(gpu)
