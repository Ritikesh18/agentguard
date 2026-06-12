"""Tests for BudgetTracker and related classes."""

import pytest
from agentguard.core.budget import BudgetTracker, CallRecord, BudgetSummary
from agentguard.exceptions import BudgetExceededError


class TestBudgetTracker:
    """Test BudgetTracker functionality."""

    def test_init_requires_limit(self):
        """Must specify at least one limit."""
        with pytest.raises(ValueError):
            BudgetTracker()

    def test_init_with_token_limit(self):
        """Can initialize with token limit only."""
        tracker = BudgetTracker(max_tokens=1000)
        assert tracker.max_tokens == 1000
        assert tracker.max_usd is None

    def test_init_with_usd_limit(self):
        """Can initialize with USD limit only."""
        tracker = BudgetTracker(max_usd=10.0)
        assert tracker.max_usd == 10.0
        assert tracker.max_tokens is None

    def test_init_with_both_limits(self):
        """Can initialize with both limits."""
        tracker = BudgetTracker(max_tokens=1000, max_usd=10.0)
        assert tracker.max_tokens == 1000
        assert tracker.max_usd == 10.0

    def test_record_call_updates_totals(self):
        """record_call accumulates tokens and USD."""
        tracker = BudgetTracker(max_usd=100.0)

        record = tracker.record_call(
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o",
        )

        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50
        assert tracker.tokens_used == 150
        assert tracker.usd_spent > 0

    def test_multiple_calls_accumulate(self):
        """Multiple calls accumulate correctly."""
        tracker = BudgetTracker(max_usd=100.0)

        tracker.record_call(100, 50, "gpt-4o")
        tracker.record_call(200, 100, "gpt-4o")

        assert tracker.tokens_used == 450  # 100+50+200+100
        assert len(tracker._calls) == 2

    def test_per_agent_tracking(self):
        """Per-agent budgets are tracked separately."""
        tracker = BudgetTracker(max_usd=100.0)

        tracker.record_call(100, 50, "gpt-4o", agent_id="agent_a")
        tracker.record_call(200, 100, "gpt-4o", agent_id="agent_b")

        summary = tracker.summary()
        assert "agent_a" in summary.per_agent_breakdown
        assert "agent_b" in summary.per_agent_breakdown
        assert summary.per_agent_breakdown["agent_a"]["tokens"] == 150
        assert summary.per_agent_breakdown["agent_b"]["tokens"] == 300

    def test_check_pre_call_no_breach(self):
        """Pre-call check passes when under limit."""
        tracker = BudgetTracker(max_tokens=1000)

        result = tracker.check_pre_call(100, "gpt-4o")

        assert result.is_breach is False

    def test_check_pre_call_token_breach(self):
        """Pre-call check detects token limit breach."""
        tracker = BudgetTracker(max_tokens=100)

        # Already used 80 tokens
        tracker.record_call(50, 30, "gpt-4o")

        # Next call would exceed
        result = tracker.check_pre_call(50, "gpt-4o")

        assert result.is_breach is True
        assert result.breach_type == "token_limit"

    def test_check_pre_call_usd_breach(self):
        """Pre-call check detects USD limit breach."""
        tracker = BudgetTracker(max_usd=0.04)

        # Already spent most budget - use many tokens
        tracker.record_call(5000, 2500, "gpt-4o")  # ~$0.0375

        # Next call would exceed
        result = tracker.check_pre_call(5000, "gpt-4o")

        assert result.is_breach is True
        assert result.breach_type == "usd_limit"

    def test_summary_format(self):
        """Summary returns correct structure."""
        tracker = BudgetTracker(max_tokens=1000, max_usd=10.0)
        tracker.record_call(100, 50, "gpt-4o")

        summary = tracker.summary()

        assert isinstance(summary, BudgetSummary)
        assert summary.total_tokens == 150
        assert summary.calls_made == 1
        assert summary.max_tokens == 1000
        assert summary.max_usd == 10.0

    def test_summary_to_dict(self):
        """Summary serializes to dict."""
        tracker = BudgetTracker(max_usd=10.0)
        tracker.record_call(100, 50, "gpt-4o")

        summary = tracker.summary()
        data = summary.to_dict()

        assert isinstance(data, dict)
        assert "total_tokens" in data
        assert "total_usd" in data

    def test_summary_to_json(self):
        """Summary serializes to JSON."""
        tracker = BudgetTracker(max_usd=10.0)
        tracker.record_call(100, 50, "gpt-4o")

        summary = tracker.summary()
        json_str = summary.to_json()

        assert isinstance(json_str, str)
        assert "total_tokens" in json_str

    def test_usd_remaining(self):
        """USD remaining is calculated correctly."""
        tracker = BudgetTracker(max_usd=1.0)
        tracker.record_call(1000, 500, "gpt-4o-mini")  # ~$0.00045

        remaining = tracker.usd_remaining
        assert 0 < remaining < 1.0

    def test_thread_safety_multiple_records(self):
        """Record calls are thread-safe."""
        import threading

        tracker = BudgetTracker(max_usd=100.0)

        def add_calls():
            for _ in range(10):
                tracker.record_call(10, 5, "gpt-4o")

        threads = [threading.Thread(target=add_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 5 threads * 10 calls * (10+5 tokens) = 750 tokens total
        assert tracker.tokens_used == 750

    def test_breach_count_increments(self):
        """Breach count increments on each pre-call check."""
        tracker = BudgetTracker(max_tokens=100)

        # Fill to near limit
        tracker.record_call(80, 20, "gpt-4o")

        # Check breach
        tracker.check_pre_call(50, "gpt-4o")
        assert tracker._breach_count == 1

        # Check breach again
        tracker.check_pre_call(50, "gpt-4o")
        assert tracker._breach_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
