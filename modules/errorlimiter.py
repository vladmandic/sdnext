from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar
from threading import Lock

if TYPE_CHECKING:
    from collections.abc import Iterable

_instance_id = 0
_lock = Lock()


def _make_unique(name: str):
    global _instance_id
    with _lock:  # Guard against race conditions
        new_name = f"{name}__{_instance_id}"
        _instance_id += 1
        return new_name


class ErrorLimiterTrigger(BaseException):  # Use BaseException to avoid being caught by "except Exception:".
    name: str
    identifier: str | None

    def __init__(self, name: str, *args):
        super().__init__(*args)
        if "__" in name:
            self.name, self.identifier = name.rsplit("__", 1)
            if self.name == "":  # Edge case if the only "__" was at the beginning of the name
                self.name = self.identifier
                self.identifier = None
        else:
            self.name = name  # Possible if implemented manually
            self.identifier = None


class ErrorLimiterAbort(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(msg)


class ErrorLimiter:
    _store: ClassVar[dict[str, int]] = {}

    @classmethod
    def start(cls, name: str, limit: int = 5):
        cls._store[name] = limit

    @classmethod
    def notify(cls, name: str | Iterable[str]):  # Can be manually triggered if execution is spread across multiple files
        if isinstance(name, str):
            name = (name,)
        for key in name:
            if key in cls._store.keys():
                cls._store[key] = cls._store[key] - 1
                if cls._store[key] <= 0:
                    raise ErrorLimiterTrigger(key)

    @classmethod
    def end(cls, name: str):
        cls._store.pop(name)


@contextmanager
def limit_errors(name: str, limit: int = 5):
    """Limiter for aborting execution after being triggered a specified number of times (default 5).

    >>> with limit_errors("identifier", limit=5) as elimit:
    >>>     while do_thing():
    >>>         if (something_bad):
    >>>             print("Something bad happened")
    >>>             elimit()  # In this example, raises ErrorLimiterAbort on the 5th call
    >>>         try:
    >>>             something_broken()
    >>>         except Exception:
    >>>             print("Encountered an exception")
    >>>             elimit()  # Count is shared across all calls

    Args:
        name (str): Identifier.
        limit (int, optional): Abort after `limit` number of triggers. Defaults to 5.

    Raises:
        ErrorLimiterAbort: Subclass of RuntimeException.

    Yields:
        Callable: Notification function to indicate that an error occurred.
    """
    name_id = _make_unique(name)
    try:
        ErrorLimiter.start(name_id, limit)
        yield lambda: ErrorLimiter.notify(name_id)
    except ErrorLimiterTrigger as e:
        raise ErrorLimiterAbort(f"HALTING. Too many errors during '{e.name}'") from None
    finally:
        ErrorLimiter.end(name_id)
