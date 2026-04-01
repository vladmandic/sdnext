import math
import json
import subprocess as sp
from enum import IntFlag


try:
    from modules.logger import log
except Exception:
    import logging
    log = logging.getLogger(__name__)


try:
    from modules.rocm import version as rocm_version
except Exception:
    rocm_version = "unknown"


# ThrottleStatus is from leuc/amdgpu_metrics.py
class ThrottleStatus(IntFlag):
    # linux/drivers/gpu/drm/amd/pm/inc/amdgpu_smu.h
    PPT0 = 1 << 0
    PPT1 = 1 << 1
    PPT2 = 1 << 2
    PPT3 = 1 << 3
    SPL = 1 << 4
    FPPT = 1 << 5
    SPPT = 1 << 6
    SPPT_APU = 1 << 7
    TDC_GFX = 1 << 16
    TDC_SOC = 1 << 17
    TDC_MEM = 1 << 18
    TDC_VDD = 1 << 19
    TDC_CVIP = 1 << 20
    EDC_CPU = 1 << 21
    EDC_GFX = 1 << 22
    APCC = 1 << 23
    TEMP_GPU = 1 << 32
    TEMP_CORE = 1 << 33
    TEMP_MEM = 1 << 34
    TEMP_EDGE = 1 << 35
    TEMP_HOTSPOT = 1 << 36
    TEMP_SOC = 1 << 37
    TEMP_VR_GFX = 1 << 38
    TEMP_VR_SOC = 1 << 39
    TEMP_VR_MEM0 = 1 << 40
    TEMP_VR_MEM1 = 1 << 41
    TEMP_LIQUID0 = 1 << 42
    TEMP_LIQUID1 = 1 << 43
    VRHOT0 = 1 << 44
    VRHOT1 = 1 << 45
    PROCHOT_CPU = 1 << 46
    PROCHOT_GFX = 1 << 47
    PPM = 1 << 56
    FIT = 1 << 57

    def active(self):
        members = self.__class__.__members__
        return (m for m in members if getattr(self, m)._value_ & self.value != 0) # pylint: disable=protected-access

    def __iter__(self):
        return self.active()

    def __str__(self):
        return ', '.join(self.active())


def get_rocm_smi():
    try:
        rocm_smi_data = json.loads(sp.check_output(("rocm-smi", "-a", "--json")))
        driver_version = rocm_smi_data.pop("system", {"Driver version": "unknown"}).get("Driver version")

        devices = []
        for key in rocm_smi_data.keys():
            load = {
                'gpu': rocm_smi_data[key].get('GPU use (%)', 'unknown'),
                'memory': rocm_smi_data[key].get("GPU Memory Allocated (VRAM%)", "unknown"),
                'temp': rocm_smi_data[key].get('Temperature (Sensor edge) (C)', 'unknown'),
                'temp_junction': rocm_smi_data[key].get('Temperature (Sensor junction) (C)', 'unknown'),
                'temp_memory': rocm_smi_data[key].get('Temperature (Sensor memory) (C)', 'unknown'),
                'fan': rocm_smi_data[key].get('Fan speed (%)', 'unknown'),
            }

            data = {
                "ROCm": f'version {rocm_version} agent {rocm_smi_data[key].get("GFX Version", "unknown")}',
                "Driver": driver_version,
                "Hardware": f'VBIOS {rocm_smi_data[key].get("VBIOS version", "unknown")}',
                "PCI link": f'Gen.{int(math.log2(float(rocm_smi_data[key].get("pcie_link_speed (0.1 GT/s)", 10)) / 10))} x{rocm_smi_data[key].get("pcie_link_width (Lanes)", "unknown")}',
                "Power": f'{round(float(rocm_smi_data[key].get("Average Graphics Package Power (W)", 0)), 2)} W / {round(float(rocm_smi_data[key].get("Max Graphics Package Power (W)", 0)), 2)} W',
                "GPU clock": f'{rocm_smi_data[key].get("average_gfxclk_frequency (MHz)", 0)} Mhz / {rocm_smi_data[key].get("Valid sclk range", "0").split(" - ")[-1].removesuffix("Mhz")} Mhz',
                "VRAM clock": f'{rocm_smi_data[key].get("current_uclk (MHz)", 0)} Mhz / {rocm_smi_data[key].get("Valid mclk range", "0").split(" - ")[-1].removesuffix("Mhz")} Mhz',
                "VRAM usage": f'{load["memory"]}% Used | {rocm_smi_data[key].get("GPU Memory Read/Write Activity (%)", "unknown")}% Activity',
                "GPU usage": f'GPU {load["gpu"]}% | Fan {load["fan"]}%',
                "GPU temp": f'Edge {load["temp"]}C | Junction {load["temp_junction"]}C | Memory {load["temp_memory"]}C',
                'Throttle reason': str(ThrottleStatus(int(rocm_smi_data[key].get("throttle_status", 0)))),
            }
            name = rocm_smi_data[key].get('Device Name', 'unknown')
            chart = [load["memory"], load["gpu"]]
            devices.append({
                'name': name,
                'data': data,
                'chart': chart,
            })
        return devices
    except Exception as e:
        log.error(f'ROCm SMI: {e}')
        return []


if __name__ == '__main__':
    from rich import print as rprint
    for gpu in get_rocm_smi():
        rprint(gpu)
