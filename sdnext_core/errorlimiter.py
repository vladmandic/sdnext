from contextlib import contextmanager


class ErrorLimiterTrigger(BaseException): # Use BaseException to avoid being caught by "except Exception:".
    def __init__(self, name: str, *args):
        super().__init__(*args)
        self.name = name


class ErrorLimiterAbort(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(msg)


class ErrorLimiter:
    _store: dict[str, int] = {}

    @classmethod
    def start(cls, name: str, limit: int = 5):
        cls._store[name] = limit

    @classmethod
    def notify(cls, name: str): # Can be manually triggered if execution is spread across multiple files
        if name in cls._store.keys():
            cls._store[name] = cls._store[name] - 1
            if cls._store[name] <= 0:
                raise ErrorLimiterTrigger(name)
        else:
            raise RuntimeError(f"ErrorLimiter for '{name}' was called before setup")

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
    >>>             elimit() # In this example, raises ErrorLimiterAbort on the 5th call
    >>>         try:
    >>>             something_broken()
    >>>         except Exception:
    >>>             print("Encountered an exception")
    >>>             elimit() # Count is shared across all calls

    Args:
        name (str): Identifier.
        limit (int, optional): Abort after `limit` number of triggers. Defaults to 5.

    Raises:
        ErrorLimiterAbort: Subclass of RuntimeException.
    """
    try:
        ErrorLimiter.start(name, limit)
        yield lambda: ErrorLimiter.notify(name)
    except ErrorLimiterTrigger as e:
        raise ErrorLimiterAbort(f"HALTING. Too many errors during {e.name}") from None
    finally:
        ErrorLimiter.end(name)
