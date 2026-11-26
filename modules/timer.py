import os
import time
import sys


try:
    default_min_time = float(os.environ.get('SD_MIN_TIMER', '0.1'))
except Exception:
    default_min_time = 0.1


def now():
    return time.time()


class Timer:
    def __init__(self, profile=False):
        self.start = time.time()
        self.records = {}
        self.excluded = {}
        self.total = 0
        self.total_override = None
        self.profile = profile

    def elapsed(self, reset=True):
        end = time.time()
        res = end - self.start
        if reset:
            self.start = end
        return res

    def add(self, name, t, include_total=True):
        if name not in self.records:
            self.records[name] = 0
        self.records[name] += t
        if not include_total:
            self.excluded[name] = self.excluded.get(name, 0) + t
        self.total = self.get_total()

    def rm(self, name):
        if name in self.records:
            del self.records[name]
        self.excluded.pop(name, None)
        self.total = self.get_total()

    def get(self, name):
        return self.records.get(name, 0)

    def set(self, name, t, include_total=True):
        self.records[name] = t
        if include_total:
            self.excluded.pop(name, None)
        else:
            self.excluded[name] = t
        self.total = self.get_total()

    def set_total(self, t):
        self.total_override = t
        self.total = t

    def ts(self, name, t):
        elapsed = time.time() - t
        self.add(name, elapsed)

    def merge(self, other):
        total = self.get_total() + other.get_total()
        has_total_override = self.total_override is not None or other.total_override is not None
        for k, v in other.records.items():
            if k not in self.records:
                self.records[k] = 0
            self.records[k] += v
        for k, v in other.excluded.items():
            self.excluded[k] = self.excluded.get(k, 0) + v
        if has_total_override:
            self.total_override = total
        self.total = self.get_total()

    def record(self, category=None, extra_time=0, reset=True):
        e = self.elapsed(reset)
        if category is None:
            category = sys._getframe(1).f_code.co_name # pylint: disable=protected-access
        self.add(category, e + extra_time)

    def summary(self, min_time=default_min_time, max_time=-1, total=True):
        if self.profile:
            min_time = -1
        self.total = self.get_total()
        res = f"total={self.total:.2f} " if total else ''
        additions = [x for x in self.records.items() if x[1] >= min_time]
        if max_time > 0:
            additions = [x for x in additions if x[1] <= max_time]
        if not total:
            additions = [x for x in additions if x[0] != 'total']
        additions = sorted(additions, key=lambda x: x[1], reverse=True)
        if not additions:
            return res
        res += " ".join([f"{category}={time_taken:.2f}" for category, time_taken in additions])
        return res

    def max(self):
        if not self.records:
            return 0
        return max(self.records.values())

    def get_total(self):
        if self.total_override is not None:
            return self.total_override
        return sum(self.records.values()) - sum(self.excluded.values())

    def dct(self, min_time=default_min_time, no_total=False):
        self.total = self.get_total()
        records = dict(self.records)
        if no_total:
            records.pop('total', None)
        else:
            records['total'] = self.total
        if self.profile:
            res = {k: round(v, 4) for k, v in records.items()}
        else:
            res = {k: round(v, 2) for k, v in records.items() if v >= min_time}
        res = {k: v for k, v in sorted(res.items(), key=lambda x: x[1], reverse=True)} # noqa: C416 # pylint: disable=unnecessary-comprehension
        return res

    def reset(self):
        self.records.clear()
        self.__init__(self.profile)

startup = Timer()
process = Timer()
video = Timer()
launch = Timer()
init = Timer()
load = Timer()
dynamo = Timer()
blocks = Timer()
autotune = Timer(profile=True)
