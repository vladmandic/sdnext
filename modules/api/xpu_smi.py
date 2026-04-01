try:
    from modules.logger import log
except Exception:
    import logging
    log = logging.getLogger(__name__)


def get_xpu_smi():
    try:
        import torch
        from modules.memstats import ram_stats

        devices = []
        mem = torch.xpu.memory_stats()
        ram = ram_stats()
        cap = torch.xpu.get_device_capability()
        prop = torch.xpu.get_device_properties()
        load = {
            'gpu': 0, # no interface to get gpu load
            'memory': mem['active_bytes.all.allocated'] // (1024**3), # no interface to get gpu memory so use torch instead
        }
        total = prop.total_memory // (1024**2)
        data = {
            'Version': cap['version'],
            'Driver': prop.driver_version,
            'Platform': prop.platform_name,
            'ID': hex(prop.device_id).removeprefix("0x"),
            'Compute Units': prop.max_compute_units,
            "VRAM usage": f'{round(100 * load["memory"] / total)}% | {load["memory"]} MB used | {total - load["memory"]} MB free | {total} MB total',
            "RAM usage": f'{round(100 * ram["used"] / ram["total"])}% | {round(1024 * ram["used"])} MB used | {round(1024 * ram["free"])} MB free | {round(1024 * ram["total"])} MB total',
        }
        chart = [load["memory"], load["gpu"]]
        devices.append({
            'name': torch.xpu.get_device_name(),
            'data': data,
            'chart': chart,
        })
        return devices
    except Exception as e:
        log.error(f'XPU SMI: {e}')
        return []


if __name__ == '__main__':
    from rich import print as rprint
    for gpu in get_xpu_smi():
        rprint(gpu)
