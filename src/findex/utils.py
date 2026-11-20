import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger("findex")

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для вимірювання часу виконання функцій та методів."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            logger.info("%s executed in %.4f s", func.__qualname__, elapsed)

    return wrapper
