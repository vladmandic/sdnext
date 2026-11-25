from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar


T = TypeVar('T')


class SynchronousExecutor(Executor):
    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        future: Future[T] = Future()
        future.set_running_or_notify_cancel()
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            future.set_exception(e)
        else:
            future.set_result(result)
        return future


def get_executor(max_workers: int, *, thread_name_prefix: str = '') -> Executor:
    if max_workers == 0:
        return SynchronousExecutor()
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
