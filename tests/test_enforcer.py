"""Tests for Enforcer and policy handling."""

import pytest
from unittest.mock import MagicMock

from agentguard.core.enforcer import Enforcer
from agentguard.core.budget import BudgetTracker
from agentguard.exceptions import BudgetExceededError


class TestEnforcer:
    """Test Enforcer pre-call enforcement."""

    def test_enforcer_init(self):
        """Enforcer initializes with tracker and policy."""
        tracker = BudgetTracker(max_usd=10.0)
        enforcer = Enforcer(tracker, "kill")

        assert enforcer.tracker is tracker
        assert enforcer.policy_handler is not None

    def test_pre_call_check_no_breach(self):
        """Pre-call check passes when under limit."""
        tracker = BudgetTracker(max_tokens=1000)
        enforcer = Enforcer(tracker, "kill")

        messages = [{"role": "user", "content": "Hello"}]
        # Should not raise
        enforcer.pre_call_check(messages, "gpt-4o")

    def test_pre_call_check_kill_policy_raises(self):
        """Kill policy raises BudgetExceededError on breach."""
        tracker = BudgetTracker(max_tokens=10)
        enforcer = Enforcer(tracker, "kill")

        messages = [{"role": "user", "content": "Hello world " * 100}]

        with pytest.raises(BudgetExceededError) as exc_info:
            enforcer.pre_call_check(messages, "gpt-4o")

        assert exc_info.value.breach_type == "token_limit"

    def test_post_call_record(self):
        """Post-call recording extracts usage from response."""
        tracker = BudgetTracker(max_usd=10.0)
        enforcer = Enforcer(tracker, "warn")

        # Mock response
        response = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50

        record = enforcer.post_call_record(response, "gpt-4o")

        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50
        assert tracker.tokens_used == 150

    def test_token_counting_openai(self):
        """Token counting works for OpenAI models."""
        enforcer = Enforcer(BudgetTracker(max_usd=100), "warn")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]

        tokens = enforcer._count_tokens(messages, "gpt-4o")

        # Should estimate reasonable token count
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_token_counting_claude(self):
        """Token counting approximates for Claude models."""
        enforcer = Enforcer(BudgetTracker(max_usd=100), "warn")

        messages = [
            {"role": "user", "content": "Hello world this is a test"},
        ]

        tokens = enforcer._count_tokens(messages, "claude-3-5-sonnet")

        # Should estimate reasonable token count
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_token_counting_with_agent_id(self):
        """Pre-call check accepts agent_id."""
        tracker = BudgetTracker(max_tokens=1000)
        enforcer = Enforcer(tracker, "warn")

        messages = [{"role": "user", "content": "Hello"}]

        # Should not raise
        enforcer.pre_call_check(messages, "gpt-4o", agent_id="agent_a")

    def test_warn_policy_logs_warning(self, caplog):
        """Warn policy logs warning instead of raising."""
        tracker = BudgetTracker(max_tokens=10)
        enforcer = Enforcer(tracker, "warn")

        messages = [{"role": "user", "content": "Hello world " * 100}]

        # Should not raise, just warn
        enforcer.pre_call_check(messages, "gpt-4o")

        # Warning should be logged
        # (caplog check would go here if configured)

    def test_multiple_models_different_pricing(self):
        """Different models calculate costs differently."""
        tracker = BudgetTracker(max_usd=100.0)
        enforcer = Enforcer(tracker, "warn")

        # gpt-4o is expensive
        response1 = MagicMock()
        response1.usage.prompt_tokens = 1000
        response1.usage.completion_tokens = 0
        record1 = enforcer.post_call_record(response1, "gpt-4o")

        # Reset
        tracker2 = BudgetTracker(max_usd=100.0)
        enforcer2 = Enforcer(tracker2, "warn")

        # gpt-4o-mini is cheaper
        response2 = MagicMock()
        response2.usage.prompt_tokens = 1000
        response2.usage.completion_tokens = 0
        record2 = enforcer2.post_call_record(response2, "gpt-4o-mini")

        # Mini should be cheaper
        assert record2.cost_usd < record1.cost_usd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
