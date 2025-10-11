import functools
import threading
from typing import Callable
import torch

from .distributed import barrier_if_distributed, get_global_rank, get_local_rank
from .logger import get_logger

logger = get_logger(__name__)


def log_on_entry(func: Callable) -> Callable:
    """
    Functions with this decorator will log the function name at entry.
    When using multiple decorators, this must be applied innermost to properly capture the name.
    """

    def log_on_entry_wrapper(*args, **kwargs):
        logger.info(f"Entering {func.__name__}")
        return func(*args, **kwargs)

    return log_on_entry_wrapper


def barrier_on_entry(func: Callable) -> Callable:
    """
    Functions with this decorator will start executing when all ranks are ready to enter.
    """

    def barrier_on_entry_wrapper(*args, **kwargs):
        barrier_if_distributed()
        return func(*args, **kwargs)

    return barrier_on_entry_wrapper


def _conditional_execute_wrapper_factory(execute: bool, func: Callable) -> Callable:
    """
    Helper function for local_rank_zero_only and global_rank_zero_only.
    """

    def conditional_execute_wrapper(*args, **kwargs):
        # Only execute if needed.
        result = func(*args, **kwargs) if execute else None
        # All GPUs must wait.
        barrier_if_distributed()
        # Return results.
        return result

    return conditional_execute_wrapper


def _asserted_wrapper_factory(condition: bool, func: Callable, err_msg: str = "") -> Callable:
    """
    Helper function for some functions with special constraints,
    especially functions called by other global_rank_zero_only / local_rank_zero_only ones,
    in case they are wrongly invoked in other scenarios.
    """

    def asserted_execute_wrapper(*args, **kwargs):
        assert condition, err_msg
        result = func(*args, **kwargs)
        return result

    return asserted_execute_wrapper


def local_rank_zero_only(func: Callable) -> Callable:
    """
    Functions with this decorator will only execute on local rank zero.
    """
    return _conditional_execute_wrapper_factory(get_local_rank() == 0, func)


def global_rank_zero_only(func: Callable) -> Callable:
    """
    Functions with this decorator will only execute on global rank zero.
    """
    return _conditional_execute_wrapper_factory(get_global_rank() == 0, func)


def assert_only_global_rank_zero(func: Callable) -> Callable:
    """
    Functions with this decorator are only accessible to processes with global rank zero.
    """
    return _asserted_wrapper_factory(
        get_global_rank() == 0, func, err_msg="Not accessible to processes with global_rank != 0"
    )


def assert_only_local_rank_zero(func: Callable) -> Callable:
    """
    Functions with this decorator are only accessible to processes with local rank zero.
    """
    return _asserted_wrapper_factory(
        get_local_rank() == 0, func, err_msg="Not accessible to processes with local_rank != 0"
    )


def new_thread(func: Callable) -> Callable:
    """
    Functions with this decorator will run in a new thread.
    The function will return the thread, which can be joined to wait for completion.
    """

    def new_thread_wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return new_thread_wrapper


def log_runtime(func: Callable) -> Callable:
    """
    Functions with this decorator will logging the runtime.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        torch.distributed.barrier()
        result = func(*args, **kwargs)
        torch.distributed.barrier()
        return result

    return wrapped
