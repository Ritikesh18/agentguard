"""@guard decorator for wrapping agent functions."""

import functools
import logging
import threading
from typing import Optional, Callable, Any
import asyncio

from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer
from agentguard.exceptions import BudgetExceededError

logger = logging.getLogger(__name__)

# Thread-local storage so each thread tracks its own active enforcer.
# This makes @guard safe under concurrent / multi-threaded agent workloads.
_local = threading.local()


def _get_local_enforcer() -> Optional[Enforcer]:
    return getattr(_local, "enforcer", None)


def _set_local_enforcer(enforcer: Optional[Enforcer]) -> None:
    _local.enforcer = enforcer


def guard(
    max_tokens: Optional[int] = None,
    max_usd: Optional[float] = None,
    on_breach: Callable | str = "kill",
    model: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Callable:
    """
    Decorator to enforce budget limits on a function.

    All LLM calls made inside the decorated function are budget-enforced.
    Works with the OpenAI integration: if openai has been patched via
    ``patch_openai`` the global enforcer is used; otherwise the active
    enforcer set by this decorator is respected automatically.

    Args:
        max_tokens: Hard token limit across all LLM calls
        max_usd: Hard USD limit across all LLM calls
        on_breach: Policy for handling breaches ('kill'|'warn'|'pause'|callable)
        model: Model identifier for pricing lookup (optional, auto-detected)
        agent_id: Label for multi-agent tracking
        session_id: Session identifier for grouping runs

    Returns:
        Decorated function with budget enforcement

    Example:
        @guard(max_usd=5.00, on_breach='kill')
        def run_agent(query: str) -> str:
            # All LLM calls here are budget-enforced
            response = openai.chat.completions.create(...)
            return response.choices[0].message.content
    """
    if max_tokens is None and max_usd is None:
        raise ValueError("Must specify either max_tokens or max_usd")

    def decorator(func: Callable) -> Callable:
        # Each decorated function gets its own fresh tracker/enforcer per call
        # so that budgets reset between invocations (not shared across calls).

        # Handle both sync and async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                tracker = BudgetTracker(
                    max_tokens=max_tokens,
                    max_usd=max_usd,
                    session_id=session_id,
                )
                enforcer = Enforcer(tracker, on_breach)
                old_enforcer = _get_local_enforcer()
                _set_local_enforcer(enforcer)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except BudgetExceededError:
                    raise
                finally:
                    _set_local_enforcer(old_enforcer)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                tracker = BudgetTracker(
                    max_tokens=max_tokens,
                    max_usd=max_usd,
                    session_id=session_id,
                )
                enforcer = Enforcer(tracker, on_breach)
                old_enforcer = _get_local_enforcer()
                _set_local_enforcer(enforcer)
                try:
                    result = func(*args, **kwargs)
                    return result
                except BudgetExceededError:
                    raise
                finally:
                    _set_local_enforcer(old_enforcer)

            return sync_wrapper

    return decorator


def get_active_enforcer() -> Optional[Enforcer]:
    """Get the currently active enforcer for this thread (used by integrations)."""
    return _get_local_enforcer()
