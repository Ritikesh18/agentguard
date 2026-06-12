"""Lightweight AWS Bedrock integration helpers for AgentGuard.

Provides `wrap_bedrock_fn` to wrap Bedrock client callables with pre/post
budget enforcement. The wrapper is agnostic to the client implementation
shape and attempts to extract `input`/`messages` and `.usage` similarly
to other adapters.
"""

from typing import Any, Callable, Optional
import functools
import logging

from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer

logger = logging.getLogger(__name__)


def wrap_bedrock_fn(
    fn: Callable[..., Any],
    max_tokens: Optional[int] = None,
    max_usd: Optional[float] = None,
    on_breach: str | Callable = "kill",
    session_id: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap a Bedrock-style call with budget checks.

    Example usage:
        wrapped = wrap_bedrock_fn(client.invoke_model, max_usd=2.0)
        resp = wrapped(modelId='amazon.titan', input='Hello')
    """

    tracker = BudgetTracker(max_tokens=max_tokens, max_usd=max_usd, session_id=session_id)
    enforcer = Enforcer(tracker, on_breach)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Build a message-like list
        messages = []
        if "messages" in kwargs:
            messages = kwargs["messages"]
        elif "input" in kwargs:
            messages = [{"role": "user", "content": kwargs["input"]}]
        elif args:
            if isinstance(args[0], str):
                messages = [{"role": "user", "content": args[0]}]

        model = kwargs.get("model") or kwargs.get("modelId") or "bedrock"

        enforcer.pre_call_check(messages, model)

        resp = fn(*args, **kwargs)

        try:
            enforcer.post_call_record(resp, model)
        except Exception as e:
            logger.debug(f"Bedrock wrapper: failed to record usage: {e}")

        return resp

    return wrapped


__all__ = ["wrap_bedrock_fn"]
