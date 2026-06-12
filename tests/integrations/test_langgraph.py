"""Unit tests for GuardCallback integration."""

from unittest.mock import MagicMock
import pytest

from agentguard.integrations.langgraph import GuardCallback
from agentguard.core.budget import BudgetTracker
from agentguard.exceptions import BudgetExceededError


def make_mock_response(prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp.usage = usage
    resp.model = "gpt-4o"
    return resp


def test_guardcallback_enforces_global_budget():
    cb = GuardCallback(max_usd=0.0001, on_breach="kill")

    # Create a prompt that will exceed the tiny budget
    prompts = ["word " * 1000]
    serialized = {"model": "gpt-4o", "metadata": {}}

    with pytest.raises(BudgetExceededError):
        cb.on_llm_start(serialized, prompts)


def test_guardcallback_records_post_call():
    cb = GuardCallback(max_usd=10.0, on_breach="warn")

    # No pre-call breach
    prompts = ["Hello"]
    serialized = {"model": "gpt-4o", "metadata": {}}

    cb.on_llm_start(serialized, prompts)

    # Mock response
    resp = make_mock_response(100, 50)
    cb.on_llm_end(resp)

    # Should have recorded usage in the global tracker
    tracker = cb._global_tracker
    assert tracker.tokens_used == 150
    assert tracker.usd_spent > 0


def test_guardcallback_per_agent_budgets():
    per_agent = {"researcher": 0.001, "writer": 0.001}
    cb = GuardCallback(per_agent_budgets=per_agent, on_breach="kill")

    # Simulate call from researcher (should breach with large prompt)
    serialized = {"model": "gpt-4o", "metadata": {}}
    prompts = ["word " * 2000]

    # Provide agent id via kwargs
    with pytest.raises(BudgetExceededError):
        cb.on_llm_start(serialized, prompts, name="researcher")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
