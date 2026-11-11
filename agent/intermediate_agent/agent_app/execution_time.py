"""A per-question time budget checked around model, tool, and search calls."""

from contextlib import contextmanager
from contextvars import ContextVar
import math
from time import perf_counter

from langchain.agents.middleware import wrap_model_call, wrap_tool_call

from .flow_logging import logger


DEFAULT_MAX_SECONDS = 180.0
_deadline: ContextVar[float | None] = ContextVar("intermediate_agent_deadline", default=None)


class RunDeadlineExceeded(TimeoutError):
    """The question has used its allotted time."""


def validate_max_seconds(max_seconds: float) -> None:
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or
            not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be a positive, finite number")


@contextmanager
def question_time_limit(max_seconds: float):
    """Start a fresh deadline for one question and restore the old context."""
    validate_max_seconds(max_seconds)
    proposed_deadline = perf_counter() + max_seconds
    parent_deadline = _deadline.get()
    token = _deadline.set(min(parent_deadline, proposed_deadline)
                          if parent_deadline is not None else proposed_deadline)
    logger.info("Question time limit: %.2fs", max_seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def check_deadline() -> None:
    deadline = _deadline.get()
    if deadline is not None and perf_counter() >= deadline:
        logger.warning("Question time limit reached")
        raise RunDeadlineExceeded("Question time limit reached")


@wrap_model_call
def enforce_model_deadline(request, handler):
    check_deadline()
    response = handler(request)
    check_deadline()
    return response


@wrap_tool_call
def enforce_tool_deadline(request, handler):
    check_deadline()
    result = handler(request)
    check_deadline()
    return result
