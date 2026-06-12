"""Lightweight Anthropic integration helpers for AgentGuard.

This module provides helper to wrap Anthropic-style completion callables
with pre-call enforcement and post-call recording. We avoid hard SDK
dependencies and accept any callable that behaves like a completion
function (accepts `prompt`/`messages` and returns an object with `.completion`
and `.usage` attributes or a dict with similar keys).
"""

from typing import Any, Callable, Optional
import functools
import logging

from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer

logger = logging.getLogger(__name__)


def wrap_completion_fn(
    fn: Callable[..., Any],
    max_tokens: Optional[int] = None,
    max_usd: Optional[float] = None,
    on_breach: str | Callable = "kill",
    session_id: Optional[str] = None,
) -> Callable[..., Any]:
    """Return a wrapped callable that enforces budget around `fn`.

    Usage:
        from agentguard.integrations.anthropic import wrap_completion_fn

        wrapped = wrap_completion_fn(client.completions.create, max_usd=5.0)
        resp = wrapped(prompt="Hello")

    The wrapper will attempt to extract messages/prompt and model from
    kwargs or positional args to compute token estimates. Post-call it
    will try to read `.usage` or dict keys to record actual tokens.
    """

    tracker = BudgetTracker(max_tokens=max_tokens, max_usd=max_usd, session_id=session_id)
    enforcer = Enforcer(tracker, on_breach)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Try to build a minimal messages list for token counting
        messages = []
        if "messages" in kwargs:
            messages = kwargs["messages"]
        elif "prompt" in kwargs:
            messages = [{"role": "user", "content": kwargs["prompt"]}]
        elif args:
            # If first arg looks like a prompt string
            if isinstance(args[0], str):
                messages = [{"role": "user", "content": args[0]}]

        model = kwargs.get("model", "claude")

        # Pre-check
        enforcer.pre_call_check(messages, model)

        # Call downstream
        resp = fn(*args, **kwargs)

        # Try to normalize response usage
        try:
            enforcer.post_call_record(resp, model)
        except Exception as e:
            logger.debug(f"Anthropic wrapper: failed to record usage: {e}")

        return resp

    return wrapped


__all__ = ["wrap_completion_fn"]
