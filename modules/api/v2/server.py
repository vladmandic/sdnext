import os
import re
import time
from fastapi import APIRouter
from modules.api.v2.models import (
    ResServerInfoV2, VersionInfoV2, ServerCapabilities, ServerModelInfo,
    ResMemoryV2, RamMemoryV2, CudaMemoryV2, MemoryUsage, MemoryPeakUsage, MemoryWarnings,
    ResGpuV2, GpuMetrics,
    ResSystemInfoV2,
)

router = APIRouter(prefix="/sdapi/v2", tags=["Server"])


def _detect_video_capability() -> bool:
    """Video generation is always available."""
    return True


@router.get("/server-info", response_model=ResServerInfoV2)
async def get_server_info_v2():
    """Return server version, backend, platform, capabilities, and loaded model."""
    import installer
    from modules import shared, devices
    from modules.sd_models import model_data
    ver = installer.get_version()
    model_name = getattr(shared.opts, 'sd_model_checkpoint', None)
    model_type = type(model_data.sd_model).__name__ if model_data.sd_model is not None else None
    capabilities = ServerCapabilities(video=_detect_video_capability())
    return ResServerInfoV2(
        version=VersionInfoV2(**{k: str(v) for k, v in ver.items() if k in VersionInfoV2.model_fields}),
        backend=shared.backend.name if hasattr(shared.backend, 'name') else str(shared.backend),
        platform=devices.get_device_name() if hasattr(devices, 'get_device_name') else str(shared.device),
        capabilities=capabilities,
        model=ServerModelInfo(name=model_name, type=model_type),
    )


@router.get("/memory", response_model=ResMemoryV2)
async def get_memory_v2():
    """Return RAM and CUDA memory usage statistics."""
    from modules import shared
    try:
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram = RamMemoryV2(free=int(ram_total - res.rss), used=int(res.rss), total=int(ram_total))
    except Exception as err:
        ram = RamMemoryV2(error=str(err))
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            system = MemoryUsage(free=s[0], used=s[1] - s[0], total=s[1])
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = MemoryPeakUsage(current=s['allocated_bytes.all.current'], peak=s['allocated_bytes.all.peak'])
            reserved = MemoryPeakUsage(current=s['reserved_bytes.all.current'], peak=s['reserved_bytes.all.peak'])
            active = MemoryPeakUsage(current=s['active_bytes.all.current'], peak=s['active_bytes.all.peak'])
            inactive = MemoryPeakUsage(current=s['inactive_split_bytes.all.current'], peak=s['inactive_split_bytes.all.peak'])
            warnings = MemoryWarnings(retries=s['num_alloc_retries'], oom=s['num_ooms'])
            cuda = CudaMemoryV2(system=system, active=active, allocated=allocated, reserved=reserved, inactive=inactive, events=warnings)
        else:
            cuda = CudaMemoryV2(error='unavailable')
    except Exception as err:
        cuda = CudaMemoryV2(error=str(err))
    return ResMemoryV2(ram=ram, cuda=cuda)


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_metrics(raw: dict[str, str]) -> GpuMetrics:
    """Parse GPU metrics from raw string values. Format depends on modules/api/gpu.py get_gpu_status()."""
    metrics = GpuMetrics()
    system_load = raw.get("System load") or raw.get("GPU usage", "")
    if system_load:
        m = re.search(r'GPU\s+(\d+)%', system_load)
        if m:
            metrics.load_gpu = float(m.group(1))
        m = re.search(r'VRAM\s+(\d+)%', system_load)
        if m:
            metrics.load_vram = float(m.group(1))
        m = re.search(r'Temp\s+(\d+)', system_load)
        if m:
            metrics.temperature = float(m.group(1))
        m = re.search(r'Fan\s+(\d+)%', system_load)
        if m:
            metrics.fan_speed = float(m.group(1))
    gpu_temp = raw.get("GPU temp", "")
    if gpu_temp and metrics.temperature is None:
        m = re.search(r'Edge\s+(\d+)', gpu_temp)
        if m:
            metrics.temperature = float(m.group(1))
    power = raw.get("Power", "")
    if power:
        parts = re.findall(r'([\d.]+)\s*W', power)
        if len(parts) >= 2:
            metrics.power_current = float(parts[0])
            metrics.power_limit = float(parts[1])
    vram_usage = raw.get("VRAM usage", "")
    if vram_usage:
        used_m = re.search(r'(\d+)\s*MB\s*used', vram_usage)
        total_m = re.search(r'(\d+)\s*MB\s*total', vram_usage)
        if used_m:
            metrics.vram_used = int(used_m.group(1)) * 1024 * 1024
        if total_m:
            metrics.vram_total = int(total_m.group(1)) * 1024 * 1024
    return metrics


@router.get("/gpu", response_model=list[ResGpuV2])
async def get_gpu_v2():
    """Return GPU name, utilization metrics, and driver details for each detected GPU."""
    from modules.api.gpu import get_gpu_status
    raw_list = get_gpu_status()
    result = []
    for raw in raw_list:
        data = raw.get('data', {})
        chart = raw.get('chart', [])
        metrics = _parse_metrics(data)
        result.append(ResGpuV2(
            name=raw.get('name', ''),
            metrics=metrics,
            details={k: str(v) for k, v in data.items()},
            chart_vram_pct=_safe_float(chart[0]) if len(chart) > 0 else None,
            chart_gpu_pct=_safe_float(chart[1]) if len(chart) > 1 else None,
        ))
    return result


@router.get("/system-info", response_model=ResSystemInfoV2, response_model_exclude_none=True)
async def get_system_info_v2():
    """Return full system diagnostics: versions, platform, GPU, device config, libraries, and flags."""
    import installer
    from modules import shared, devices, loader
    ver = installer.get_version()
    plat = installer.get_platform()
    gpu = devices.get_gpu_info()
    packages = loader.get_packages()
    flags = []
    for flag in ['medvram', 'lowvram', 'no_half', 'no_half_vae', 'api_only']:
        if getattr(shared.cmd_opts, flag, False):
            flags.append(flag)
    return ResSystemInfoV2(
        version={k: str(v) for k, v in ver.items()},
        uptime=time.strftime('%c', time.localtime(shared.state.server_start)) if hasattr(shared.state, 'server_start') else '',
        timestamp=time.strftime('%c'),
        platform={k: str(v) for k, v in plat.items()} if isinstance(plat, dict) else {"platform": str(plat)},
        torch=str(packages.get('torch', '')),
        gpu={k: str(v) for k, v in gpu.items()} if isinstance(gpu, dict) else {"gpu": str(gpu)},
        device={
            "device": str(getattr(devices, 'device', '')),
            "dtype": str(getattr(devices, 'dtype', '')),
            "dtype_vae": str(getattr(devices, 'dtype_vae', '')),
            "dtype_unet": str(getattr(devices, 'dtype_unet', '')),
        },
        libs={k: str(packages.get(k, '')) for k in ['torch', 'diffusers', 'transformers', 'gradio', 'safetensors', 'accelerate'] if packages.get(k)},
        backend=shared.backend.name if hasattr(shared.backend, 'name') else str(shared.backend),
        pipeline=str(getattr(shared, 'native', '')),
        cross_attention=str(getattr(shared.opts, 'cross_attention_optimization', '')),
        flags=flags,
    )
