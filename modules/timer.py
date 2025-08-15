import os
import time
import sys


try:
    default_min_time = float(os.environ.get('SD_MIN_TIMER', '0.05'))
except Exception:
    default_min_time = 0.1


class Timer:
    def __init__(self):
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.profile = False

    def elapsed(self, reset=True):
        end = time.time()
        res = end - self.start
        if reset:
            self.start = end
        return res

    def add(self, name, t):
        if name not in self.records:
            self.records[name] = 0
        self.records[name] += t

    def ts(self, name, t):
        elapsed = time.time() - t
        self.add(name, elapsed)

    def record(self, category=None, extra_time=0, reset=True):
        e = self.elapsed(reset)
        if category is None:
            category = sys._getframe(1).f_code.co_name # pylint: disable=protected-access
        if category not in self.records:
            self.records[category] = 0
        self.records[category] += e + extra_time
        self.total += e + extra_time

    def summary(self, min_time=default_min_time, total=True):
        if self.profile:
            min_time = -1
        self.total = sum(self.records.values())
        res = f"total={self.total:.2f} " if total else ''
        additions = [x for x in self.records.items() if x[1] >= min_time]
        additions = sorted(additions, key=lambda x: x[1], reverse=True)
        if not additions:
            return res
        res += " ".join([f"{category}={time_taken:.2f}" for category, time_taken in additions])
        return res

    def get_total(self):
        return sum(self.records.values())

    def dct(self, min_time=default_min_time):
        if self.profile:
            res = {k: round(v, 4) for k, v in self.records.items()}
        self.total = sum(self.records.values())
        self.records['total'] = self.total
        res = {k: round(v, 2) for k, v in self.records.items() if v >= min_time}
        res = {k: v for k, v in sorted(res.items(), key=lambda x: x[1], reverse=True)} # noqa: C416 # pylint: disable=unnecessary-comprehension
        return res

    def reset(self):
        self.__init__()

startup = Timer()
process = Timer()
launch = Timer()
init = Timer()
load = Timer()
