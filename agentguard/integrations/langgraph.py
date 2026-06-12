"""LangGraph / LangChain GuardCallback integration for AgentGuard.

This module provides `GuardCallback` which can be attached to a LangChain/LangGraph
callbacks list. It enforces pre-call budget checks and records usage post-call.

The class is framework-agnostic: if `langchain` is installed it subclasses
`BaseCallbackHandler`, otherwise it remains a plain Python class with the
same hook method names (`on_llm_start`, `on_llm_end`, `on_llm_error`).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING, Callable
import logging

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from langchain.callbacks.base import BaseCallbackHandler
else:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except Exception:  # pragma: no cover - optional dependency
        BaseCallbackHandler = object  # type: ignore

from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer

logger = logging.getLogger(__name__)


class GuardCallback(BaseCallbackHandler):
    """
    Callback handler to integrate AgentGuard with LangGraph/LangChain.

    Args:
        max_usd: Global USD budget (optional)
        on_breach: Policy string or callable ('kill'|'warn'|'pause'|callable)
        per_agent_budgets: Optional dict mapping agent_name -> usd_limit
        agent_id_key: Key to extract agent id from metadata/kwargs (defaults to 'name')
    """

    def __init__(
        self,
        max_usd: Optional[float] = None,
        on_breach: str | Callable = "kill",
        per_agent_budgets: Optional[Dict[str, float]] = None,
        agent_id_key: str = "name",
    ) -> None:
        self.max_usd = max_usd
        self.on_breach = on_breach
        self.agent_id_key = agent_id_key

        # Global tracker/enforcer
        self._global_tracker = BudgetTracker(max_usd=max_usd) if max_usd is not None else None
        self._global_enforcer = (
            Enforcer(self._global_tracker, on_breach) if self._global_tracker else None
        )

        # Per-agent trackers and enforcers
        self._per_agent_trackers: Dict[str, BudgetTracker] = {}
        self._per_agent_enforcers: Dict[str, Enforcer] = {}
        if per_agent_budgets:
            for agent, usd in per_agent_budgets.items():
                tracker = BudgetTracker(max_usd=usd)
                self._per_agent_trackers[agent] = tracker
                self._per_agent_enforcers[agent] = Enforcer(tracker, on_breach)

    # LangChain hook: called when an LLM starts
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs) -> None:
        """Called before an LLM call is made. Enforce budgets here."""
        # Extract agent id (prefer kwargs then serialized metadata)
        agent_id = kwargs.get(self.agent_id_key) or serialized.get("metadata", {}).get(
            self.agent_id_key
        )
        model = serialized.get("model") or kwargs.get("model") or "gpt-3.5-turbo"

        messages = []
        # Convert prompts to message-like dicts for token counting
        for p in prompts:
            if isinstance(p, dict):
                messages.append(p)
            else:
                messages.append({"role": "user", "content": str(p)})

        # Prefer per-agent enforcer if available
        enforcer = None
        if agent_id and agent_id in self._per_agent_enforcers:
            enforcer = self._per_agent_enforcers[agent_id]
        elif self._global_enforcer:
            enforcer = self._global_enforcer

        if enforcer is None:
            # Nothing to enforce
            return

        try:
            enforcer.pre_call_check(messages, model, agent_id)
        except Exception as e:
            # If enforcer raises BudgetExceededError it will propagate
            logger.debug(f"GuardCallback: pre_call_check raised: {e}")
            raise

    # LangChain hook: called after LLM returns
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Record the actual usage after the LLM returned."""
        # Try to determine model and agent id
        model = getattr(response, "model", None) or kwargs.get("model") or "gpt-3.5-turbo"
        agent_id = kwargs.get(self.agent_id_key) or getattr(response, "metadata", {}).get(
            self.agent_id_key
        )

        # Response should have .usage with prompt_tokens and completion_tokens
        enforcer = None
        if agent_id and agent_id in self._per_agent_enforcers:
            enforcer = self._per_agent_enforcers[agent_id]
        elif self._global_enforcer:
            enforcer = self._global_enforcer

        if enforcer is None:
            return

        try:
            enforcer.post_call_record(response, model, agent_id)
        except Exception as e:
            logger.warning(f"GuardCallback: failed to record usage: {e}")

    # LangChain hook: called on error
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        # Do not count failed calls
        logger.debug(f"GuardCallback: LLM error: {error}")


__all__ = ["GuardCallback"]
