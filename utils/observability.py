from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_trace_debug: ContextVar[bool] = ContextVar("trace_debug", default=False)


def get_trace_id() -> str:
    return _trace_id.get()


def is_trace_debug() -> bool:
    return _trace_debug.get()


@contextmanager
def trace_context(trace_id: str, trace_debug: bool = False):
    trace_token = _trace_id.set(trace_id or "-")
    debug_token = _trace_debug.set(bool(trace_debug))
    try:
        yield
    finally:
        _trace_id.reset(trace_token)
        _trace_debug.reset(debug_token)