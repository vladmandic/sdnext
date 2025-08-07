import torch
from installer import log


device = None


def get_gpu_status():
    global device # pylint: disable=global-statement
    if device is None:
        try:
            device = torch.cuda.get_device_name(torch.cuda.current_device())
            log.info(f'GPU monitoring: device={device}')
        except Exception:
            device = ''
    # per vendor modules
    if 'nvidia' in device.lower():
        from modules.api import nvml
        return nvml.get_nvml()
    return []


"""
Resut should always be: list[ResGPU]
class ResGPU(BaseModel):
    name: str = Field(title="GPU Name")
    data: dict = Field(title="Name/Value data")
    chart: list[float, float] = Field(title="Exactly two items to place on chart")
"""
