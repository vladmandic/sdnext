import re
import sys
import os
import types
from collections import deque
import psutil
import torch
from modules import shared, errors, devices
from modules.logger import log


fail_once = False
ram = {}
gpu = {}
mem = {}
process = None
docker_limit = None
runpod_limit = None


def gb(val: float):
    return round(val / 1024 / 1024 / 1024, 2)


def get_docker_limit():
    global docker_limit # pylint: disable=global-statement
    if docker_limit is not None:
        return docker_limit
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', encoding='utf8') as f:
            docker_limit = float(f.read())
    except Exception:
        docker_limit = sys.float_info.max
    if docker_limit == 0:
        docker_limit = sys.float_info.max
    return docker_limit


def get_runpod_limit():
    global runpod_limit # pylint: disable=global-statement
    if runpod_limit is not None:
        return runpod_limit
    runpod_limit = float(os.environ.get('RUNPOD_MEM_GB', 0)) * 1024 * 1024 * 1024
    if runpod_limit == 0:
        runpod_limit = sys.float_info.max
    return runpod_limit


def ram_stats():
    global process, fail_once # pylint: disable=global-statement
    try:
        if process is None:
            process = psutil.Process(os.getpid())
        res = process.memory_info()
        if 'total' not in ram:
            process = psutil.Process(os.getpid())
            ram_total = 100 * res.rss / process.memory_percent()
            ram_total = min(ram_total, get_docker_limit(), get_runpod_limit())
            ram['total'] = gb(ram_total)
        ram['rss'] = gb(res.rss)
    except Exception as e:
        ram['total'] = 0
        ram['rss'] = 0
        ram['error'] = str(e)
        if not fail_once:
            log.error(f'RAM stats: {e}')
            errors.display(e, 'RAM stats')
            fail_once = True
    try:
        vmem = psutil.virtual_memory()
        ram['used'] = gb(vmem.used) if hasattr(vmem, 'used') else 0
        ram['free'] = gb(vmem.free) if hasattr(vmem, 'free') else 0
        ram['avail'] = gb(vmem.available) if hasattr(vmem, 'available') else 0
        ram['buffers'] = gb(vmem.buffers) if hasattr(vmem, 'buffers') else 0
        ram['cached'] = gb(vmem.cached) if hasattr(vmem, 'cached') else 0
    except Exception as e:
        ram['used'] = 0
        ram['free'] = 0
        ram['avail'] = 0
        ram['buffers'] = 0
        ram['cached'] = 0
        ram['error'] = str(e)
        if not fail_once:
            log.error(f'RAM stats: {e}')
            errors.display(e, 'RAM stats')
            fail_once = True
    return ram


def gpu_stats():
    global fail_once # pylint: disable=global-statement
    try:
        free, total = torch.cuda.mem_get_info()
        gpu['used'] = gb(total - free)
        gpu['total'] = gb(total)
        stats = dict(torch.cuda.memory_stats())
        if stats.get('num_ooms', 0) > 0:
            shared.state.oom = True
        gpu['active'] = gb(stats.get('active_bytes.all.current', 0))
        gpu['peak'] = gb(stats.get('active_bytes.all.peak', 0))
        gpu['retries'] = stats.get('num_alloc_retries', 0)
        gpu['oom'] = stats.get('num_ooms', 0)
    except Exception as e:
        gpu['total'] = 0
        gpu['used'] = 0
        gpu['error'] = str(e)
        if not fail_once:
            log.warning(f'GPU stats: {e}')
            # errors.display(e, 'GPU stats')
            fail_once = True
    return gpu


def memory_stats():
    mem['ram'] = ram_stats()
    mem['gpu'] = gpu_stats()
    mem['job'] = shared.state.job
    try:
        mem['gpu']['swap'] = round(mem['gpu']['active'] - mem['gpu']['used']) if mem['gpu']['active'] > mem['gpu']['used'] else 0
    except Exception:
        mem['gpu']['swap'] = 0
    return mem


def reset_stats():
    try:
        torch.cuda.reset_memory_stats()
    except Exception:
        pass


class Object:
    pattern = r"'(.*?)'"

    def get_size(self, obj, seen=None):
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0  # Avoid double counting
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum(self.get_size(k, seen) + self.get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset, deque)):
            size += sum(self.get_size(i, seen) for i in obj)
        return size

    def __init__(self, name, obj):
        self.id = id(obj)
        self.name = name
        self.fn = sys._getframe(2).f_code.co_name
        self.refcount = sys.getrefcount(obj)
        if torch.is_tensor(obj):
            self.type = obj.dtype
            self.size = obj.element_size() * obj.nelement()
        else:
            self.type = re.findall(self.pattern, str(type(obj)))[0]
            self.size = self.get_size(obj)
    def __str__(self):
        return f'{self.fn}.{self.name} type={self.type} size={self.size} ref={self.refcount}'


def get_objects(gcl=None, threshold:int=1024*1024):
    devices.torch_gc(force=True)
    if gcl is None:
        # gcl = globals()
        gcl = {}
        log.trace(f'Memory: modules={len(sys.modules)}')
        for _module_name, module in sys.modules.items():
            try:
                if not isinstance(module, types.ModuleType):
                    continue
                namespace = vars(module)
                gcl.update(namespace)
            except Exception:
                pass # Some modules may not allow introspection
    objects = []
    seen = []

    log.trace(f'Memory: items={len(gcl)} threshold={threshold}')
    for name, obj in gcl.items():
        if id(obj) in seen:
            continue
        seen.append(id(obj))
        if name == '__name__':
            name = obj
        elif name.startswith('__'):
            continue
        try:
            o = Object(name, obj)
            if o.size >= threshold:
                objects.append(o)
        except Exception:
            pass

    objects = sorted(objects, key=lambda x: x.size, reverse=True)
    for obj in objects:
        log.trace(f'Memory: {obj}')

    return objects
