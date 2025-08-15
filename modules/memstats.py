import re
import sys
import os
import psutil
import torch
from modules import shared, errors


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
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r', encoding='utf8') as f:
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
        ram['used'] = gb(res.rss)
        ram['free'] = round(ram['total'] - ram['used'])
    except Exception as e:
        ram['total'] = 0
        ram['used'] = 0
        ram['error'] = str(e)
        if not fail_once:
            shared.log.error(f'RAM stats: {e}')
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
            shared.log.error(f'GPU stats: {e}')
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

    def __init__(self, name, obj):
        self.id = id(obj)
        self.name = name
        self.fn = sys._getframe(2).f_code.co_name
        self.size = sys.getsizeof(obj)
        self.refcount = sys.getrefcount(obj)
        if torch.is_tensor(obj):
            self.type = obj.dtype
            self.size = obj.element_size() * obj.nelement()
        else:
            self.type = re.findall(self.pattern, str(type(obj)))[0]
            self.size = sys.getsizeof(obj)
    def __str__(self):
        return f'{self.fn}.{self.name} type={self.type} size={self.size} ref={self.refcount}'


def get_objects(gcl={}, threshold:int=0):
    objects = []
    seen = []

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
        shared.log.trace(obj)

    return objects
