import logging
import time

from air_quality.logging_config import (
    JsonFormatter,
    set_correlation_id,
)
from air_quality.utils import timed


def test_timed_decorator():
    @timed
    def sample_function():
        time.sleep(0.01)
        return "done"

    result = sample_function()

    assert result == "done"


def test_correlation_id_in_log():
    set_correlation_id("test-request-123")

    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    assert '"correlation_id": "test-request-123"' in formatted

