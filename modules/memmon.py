from collections import defaultdict
import torch


class MemUsageMonitor():
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
