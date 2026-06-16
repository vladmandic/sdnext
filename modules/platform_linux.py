import os
import ctypes
from modules.logger import log


class LinuxUtils():
    @staticmethod
    def get_status() -> dict[str, float] | None:
        lines = []
        status = {}
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return status
        for line in lines:
            key, _sep, value = line.partition(":")
            parts = value.strip().split()
            if not parts:
                continue
            try:
                status[key] = parts[0]
            except ValueError:
                continue
        log.debug(f'Linux status: {status}')
        return status

    @staticmethod
    def get_smaps(limit: int = 8) -> list[dict[str, float | str]] | None:
        try:
            with open("/proc/self/smaps", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return None
        entries = []
        current = None
        for raw_line in lines:
            line = raw_line.rstrip()
            if not line:
                continue
            if "-" in line and line[:1].isalnum() and line.split(maxsplit=1)[0].count("-") == 1:
                if current is not None:
                    entries.append(current)
                parts = line.split(maxsplit=5)
                current = {
                    "path": parts[5] if len(parts) > 5 else "[anonymous]",
                    "rss": 0,
                    "pss": 0,
                    "private": 0,
                    "shared": 0,
                }
                continue
            if current is None or ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            value = value.strip().split()
            if not value:
                continue
            try:
                amount = int(value[0])
            except ValueError:
                continue
            if key == "Rss":
                current["rss"] += amount
            elif key == "Pss":
                current["pss"] += amount
            elif key in {"Private_Clean", "Private_Dirty"}:
                current["private"] += amount
            elif key in {"Shared_Clean", "Shared_Dirty"}:
                current["shared"] += amount
        if current is not None:
            entries.append(current)
        merged = {}
        for entry in entries:
            path = entry["path"]
            if path not in merged:
                merged[path] = entry.copy()
            else:
                merged[path]["rss"] += entry["rss"]
                merged[path]["pss"] += entry["pss"]
                merged[path]["private"] += entry["private"]
                merged[path]["shared"] += entry["shared"]
            top = sorted(merged.values(), key=lambda item: item["rss"], reverse=True)[:limit]
        for entry in top:
            entry["rss"] = round(entry["rss"] / 1024 / 1024, 3)
            entry["pss"] = round(entry["pss"] / 1024 / 1024, 3)
            entry["private"] = round(entry["private"] / 1024 / 1024, 3)
            entry["shared"] = round(entry["shared"] / 1024 / 1024, 3)
        log.debug(f'Linux smaps: top={top}')
        return top

    @staticmethod
    def malloc_trim() -> bool | None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            status = bool(libc.malloc_trim(0))
            log.debug(f"Linux trim: status={status}")
        except (AttributeError, OSError):
            log.debug("Linux trim: not supported")

    @staticmethod
    def advise_mmap():
        """Mark mmaps as temporary so OS prioritizes dropping them."""
        MADV_COLD = 5  # Linux 5.4+, mark as unlikely to be used
        libc = ctypes.CDLL('libc.so.6')
        advised = 0
        with open('/proc/self/maps', 'r', encoding='utf-8') as f:
            for line in f:
                if 'blobs' in line or '/dev/zero' in line:
                    try:
                        addr, size = line.split()[0].split('-')
                        addr = int(addr, 16)
                        size = int(size, 16) - addr
                        libc.madvise(ctypes.c_void_p(addr), size, MADV_COLD)
                        advised += 1
                    except Exception:
                        log.error(f"Linux mmap advise: {line.strip()}")
        log.debug(f"Linux mmap advise: num={advised}")

    @staticmethod
    def release_mmap():
        """Use madvise to drop safetensors blob mmaps from page cache."""
        try:
            libc = ctypes.CDLL('libc.so.6')
            # Get all memory mappings for this process
            dropped = []
            with open('/proc/self/maps', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6:
                        path = parts[5]
                        if 'blobs' in path or '/dev/zero' in path:
                            if path in dropped:
                                continue
                            try:
                                addr, size = line.split()[0].split('-')
                                addr = int(addr, 16)
                                size = int(size, 16) - addr
                                if libc.madvise(ctypes.c_void_p(addr), size, 4) == 0:
                                    dropped.append(path)
                            except Exception:
                                log.error(f"Linux mmap release: {line.strip()}")
            log.debug(f"Linux mmap release: {dropped}")
        except Exception as e:
            log.error(f"Linux mmap release: {e}")


    @staticmethod
    def advise_cache():
        """Advise OS to drop cache for safetensors blobs."""
        from modules.shared import opts
        try:
            if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
                for root, _dirs, files in os.walk(opts.hfcache_dir, topdown=False):
                    for f in files:
                        if f.startswith(('blobs', 'snapshots')):
                            try:
                                path = os.path.join(root, f)
                                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                                os.close(fd)
                            except Exception:
                                log.error(f"Linux cache: {path}")
            log.debug("Linux cache: advised")
        except Exception as e:
            log.error(f"Linux cache: {e}")
