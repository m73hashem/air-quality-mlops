import time
from collections.abc import Callable
from functools import wraps
from typing import Any
import logging

logger = logging.getLogger(__name__)

def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Measure and report the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info("%s completed in %.4f seconds",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper
