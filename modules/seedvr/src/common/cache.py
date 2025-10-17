from typing import Callable


class Cache:
    """Caching reusable args for faster inference"""

    def __init__(self, disable=False, prefix="", cache=None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key: str, fn: Callable):
        if self.disable:
            return fn()

        key = self.prefix + key
        try:
            result = self.cache[key]
        except KeyError:
            result = fn()
            self.cache[key] = result
        return result

    def namespace(self, namespace: str):
        return Cache(
            disable=self.disable,
            prefix=self.prefix + namespace + ".",
            cache=self.cache,
        )

    def get(self, key: str):
        key = self.prefix + key
        return self.cache[key]
