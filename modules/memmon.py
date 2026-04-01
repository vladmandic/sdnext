from collections import defaultdict
import torch


class MemUsageMonitor:
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.data = defaultdict(int)
        if not torch.cuda.is_available():
            self.disabled = True
        else:
            try:
                torch.cuda.mem_get_info(self.device.index if self.device.index is not None else torch.cuda.current_device())
                torch.cuda.memory_stats(self.device)
            except Exception:
                self.disabled = True

    def cuda_mem_get_info(self): # legacy for extensions only
        if self.disabled:
            return 0, 0
        return torch.cuda.mem_get_info(self.device.index if self.device.index is not None else torch.cuda.current_device())

    def reset(self):
        if not self.disabled:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                self.data['retries'] = 0
                self.data['oom'] = 0
                # torch.cuda.reset_accumulated_memory_stats(self.device)
                # torch.cuda.reset_max_memory_allocated(self.device)
                # torch.cuda.reset_max_memory_cached(self.device)
            except Exception:
                pass

    def read(self):
        if not self.disabled:
            try:
                self.data["free"], self.data["total"] = torch.cuda.mem_get_info(self.device.index if self.device.index is not None else torch.cuda.current_device())
                self.data["used"] = self.data["total"] - self.data["free"]
                torch_stats = torch.cuda.memory_stats(self.device)
                self.data["active"] = torch_stats.get("active.all.current", torch_stats.get("active_bytes.all.current", -1))
                self.data["active_peak"] = torch_stats.get("active_bytes.all.peak", -1)
                self.data["reserved"] = torch_stats.get("reserved_bytes.all.current", -1)
                self.data["reserved_peak"] = torch_stats.get("reserved_bytes.all.peak", -1)
                self.data['retries'] = torch_stats.get("num_alloc_retries", -1)
                self.data['oom'] = torch_stats.get("num_ooms", -1)
            except Exception:
                self.disabled = True
        return self.data

    def summary(self):
        from modules.memstats import ram_stats
        gpu = ''
        cpu = ''
        gpu = ''
        if not self.disabled:
            mem_mon_read = self.read()
            ooms = mem_mon_read.pop("oom")
            retries = mem_mon_read.pop("retries")
            vram = {k: v//1048576 for k, v in mem_mon_read.items()}
            if 'active_peak' in vram:
                peak = max(vram['active_peak'], vram['reserved_peak'], vram['used'])
                used = round(100.0 * peak / vram['total']) if vram['total'] > 0 else 0
            else:
                peak = 0
                used = 0
            if peak > 0:
                gpu += f"| GPU {peak} MB"
                gpu += f" {used}%" if used > 0 else ''
                gpu += f" | retries {retries} oom {ooms}" if retries > 0 or ooms > 0 else ''
        ram = ram_stats()
        if ram['used'] > 0:
            cpu += f"| RAM {ram['used']} GB"
            cpu += f" {round(100.0 * ram['used'] / ram['total'])}%" if ram['total'] > 0 else ''
        return f'{gpu} {cpu}'
