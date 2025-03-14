import re
import sys
import os
import psutil
import torch
from modules import shared, errors

fail_once = False
mem = {}
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
    return docker_limit


def get_runpod_limit():
    global runpod_limit # pylint: disable=global-statement
    if runpod_limit is not None:
        return runpod_limit
    runpod_limit = float(os.environ.get('RUNPOD_MEM_GB', 0))
    if runpod_limit == 0: # RUNPOD_MEM_GB can be set to 0 for no limit
        runpod_limit = sys.float_info.max
    return runpod_limit


def memory_stats():
    global fail_once # pylint: disable=global-statement
    mem.clear()
    try:
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram_total = min(ram_total, get_docker_limit(), get_runpod_limit())
        ram = { 'used': gb(res.rss), 'total': gb(ram_total) }
        mem.update({ 'ram': ram })
    except Exception as e:
        if not fail_once:
            shared.log.error(f'Memory stats: {e}')
            errors.display(e, 'Memory stats')
            fail_once = True
        mem.update({ 'ram': { 'error': str(e) } })
    try:
        s = torch.cuda.mem_get_info()
        gpu = { 'used': gb(s[1] - s[0]), 'total': gb(s[1]) }
        s = dict(torch.cuda.memory_stats())
        if s.get('num_ooms', 0) > 0:
            shared.state.oom = True
        mem.update({
            'gpu': gpu,
            'retries': s.get('num_alloc_retries', 0),
            'oom': s.get('num_ooms', 0)
        })
        return mem
    except Exception:
        pass
    return mem


def memory_cache():
    return mem


def ram_stats():
    try:
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram_total = min(ram_total, get_docker_limit(), get_runpod_limit())
        ram = { 'used': gb(res.rss), 'total': gb(ram_total) }
        return ram
    except Exception:
        return { 'used': 0, 'total': 0 }


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
