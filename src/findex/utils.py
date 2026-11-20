import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger("findex")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            wrapper.last_elapsed = elapsed  # type: ignore[attr-defined]
            logger.info("%s executed in %.4f s", func.__qualname__, elapsed)

    wrapper.last_elapsed = 0.0  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
