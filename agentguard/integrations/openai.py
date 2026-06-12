"""Raw OpenAI SDK integration for AgentGuard."""

import logging
from typing import Optional, Callable
import functools

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer
from agentguard.decorator import get_active_enforcer

logger = logging.getLogger(__name__)

# Global state for OpenAI patching
_global_enforcer: Optional[Enforcer] = None
_original_create = None


def patch_openai(
    max_tokens: Optional[int] = None,
    max_usd: Optional[float] = None,
    on_breach: Callable | str = "kill",
    session_id: Optional[str] = None,
) -> None:
    """
    Patch openai.chat.completions.create to enforce budgets.

    Call this once at startup. All subsequent openai calls will be budget-enforced.
    No changes needed to existing code.

    Args:
        max_tokens: Hard token limit
        max_usd: Hard USD limit
        on_breach: Policy for handling breaches
        session_id: Optional session identifier

    Example:
        from agentguard.integrations.openai import patch_openai
        import openai

        patch_openai(max_usd=10.00, on_breach='kill')

        # All subsequent openai calls are now budget-enforced
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
    """
    global _global_enforcer, _original_create

    if _global_enforcer is not None:
        logger.warning("OpenAI already patched. Skipping re-patch.")
        return

    # Create tracker and enforcer
    tracker = BudgetTracker(
        max_tokens=max_tokens,
        max_usd=max_usd,
        session_id=session_id,
    )
    _global_enforcer = Enforcer(tracker, on_breach)

    # Patch the create method
    _original_create = openai.chat.completions.create

    @functools.wraps(_original_create)
    def patched_create(*args, **kwargs):
        """Patched create method with budget enforcement."""
        # Extract model and messages
        model = kwargs.get("model") or (args[0] if args else None)
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)

        if not messages:
            # If no messages provided, just call original
            return _original_create(*args, **kwargs)

        # Prefer the enforcer from @guard (thread-local) if active,
        # otherwise fall back to the global patch enforcer.
        active = get_active_enforcer()
        enforcer = active if active is not None else _global_enforcer

        # Pre-call check
        enforcer.pre_call_check(messages, model)

        # Make the actual call
        response = _original_create(*args, **kwargs)

        # Post-call record
        enforcer.post_call_record(response, model)

        return response

    # Replace the method
    openai.chat.completions.create = patched_create
    logger.info("OpenAI patched successfully with budget enforcement")


def unpatch_openai() -> None:
    """Restore original openai.chat.completions.create."""
    global _global_enforcer, _original_create

    if _original_create is None:
        logger.warning("OpenAI not patched. Skipping un-patch.")
        return

    openai.chat.completions.create = _original_create
    _global_enforcer = None
    _original_create = None
    logger.info("OpenAI un-patched")


def get_openai_enforcer() -> Optional[Enforcer]:
    """Get the global enforcer used for OpenAI patching."""
    global _global_enforcer
    return _global_enforcer
