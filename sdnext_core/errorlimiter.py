from typing import final


class ErrorLimiterTrigger(RuntimeError):
    def __init__(self, name: str, *args):
        super().__init__(*args)
        self.name = name


class ErrorLimiterWarning(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@final
class ErrorLimiter:
    _store: dict[str, int] = {}

    @classmethod
    def start(cls, name: str, limit: int = 5):
        cls._store[name] = limit

    @classmethod
    def update(cls, name: str):
        if name in cls._store.keys():
            cls._store[name] = cls._store[name] - 1
            if cls._store[name] <= 0:
                raise ErrorLimiterTrigger(name)
        else:
            raise ErrorLimiterWarning(f"ErrorLimiter for '{name}' was called before setup")
