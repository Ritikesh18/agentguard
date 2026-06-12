"""Tests for the @guard decorator."""

import asyncio
import threading
import pytest
from unittest.mock import MagicMock, patch

from agentguard.decorator import guard, get_active_enforcer
from agentguard.exceptions import BudgetExceededError


class TestGuardDecorator:
    """Test @guard decorator behaviour."""

    def test_requires_at_least_one_limit(self):
        """guard() with no limits raises ValueError at decoration time."""
        with pytest.raises(ValueError):

            @guard()
            def fn():
                pass

    def test_sync_function_executes_normally(self):
        """Decorated sync function runs and returns value."""

        @guard(max_usd=10.0)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_async_function_executes_normally(self):
        """Decorated async function runs and returns value."""

        @guard(max_usd=10.0)
        async def async_add(a, b):
            return a + b

        assert asyncio.run(async_add(2, 3)) == 5

    def test_active_enforcer_set_during_call(self):
        """get_active_enforcer() returns enforcer inside guarded function."""
        captured = []

        @guard(max_usd=10.0)
        def fn():
            captured.append(get_active_enforcer())

        assert get_active_enforcer() is None  # nothing active before call
        fn()
        assert get_active_enforcer() is None  # nothing active after call
        assert captured[0] is not None  # enforcer was active during call

    def test_active_enforcer_cleared_after_exception(self):
        """Enforcer is cleaned up even when the function raises."""

        @guard(max_usd=10.0)
        def failing():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing()

        assert get_active_enforcer() is None

    def test_budget_exceeded_error_propagates(self):
        """BudgetExceededError raised inside guard propagates correctly."""
        from agentguard.core.enforcer import Enforcer

        @guard(max_usd=10.0, on_breach="kill")
        def fn():
            enforcer = get_active_enforcer()
            # Force a breach by making the tracker think it already spent the budget
            enforcer.tracker._total_usd = 9.99
            # Now check a call that will tip it over
            from agentguard.core.budget import BreachResult

            result = BreachResult(is_breach=True, breach_type="usd_limit", overage_usd=0.01)
            enforcer.policy_handler.handle(result, enforcer.tracker, 10, "gpt-4o")

        with pytest.raises(BudgetExceededError):
            fn()

    def test_fresh_tracker_per_call(self):
        """Each call to a guarded function gets a fresh budget tracker."""
        trackers = []

        @guard(max_usd=10.0)
        def fn():
            trackers.append(get_active_enforcer().tracker)

        fn()
        fn()

        # Should be two different tracker objects
        assert trackers[0] is not trackers[1]

    def test_thread_safety_independent_enforcers(self):
        """Concurrent calls use independent enforcers per thread."""
        results = {}

        @guard(max_usd=10.0)
        def fn(thread_id):
            results[thread_id] = get_active_enforcer()

        threads = [threading.Thread(target=fn, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 5 threads should have captured an enforcer
        assert len(results) == 5
        # All enforcers should be distinct objects
        enforcer_ids = [id(e) for e in results.values()]
        assert len(set(enforcer_ids)) == 5

    def test_guard_integrates_with_patched_openai(self):
        """@guard enforcer is used when OpenAI is patched and @guard is active."""
        openai = pytest.importorskip("openai", reason="openai not installed")
        from agentguard.integrations.openai import patch_openai, unpatch_openai

        # Patch OpenAI with a large global budget
        patch_openai(max_usd=100.0, on_breach="kill")

        try:
            # Mock the underlying create so no real HTTP call is made
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            with patch(
                "agentguard.integrations.openai._original_create",
                return_value=mock_response,
            ):

                @guard(max_usd=0.000001, on_breach="kill")  # tiny per-call budget
                def agent():
                    # This call should hit the @guard enforcer (tiny budget), not the global one
                    openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "word " * 500}],
                    )

                # The @guard budget (0.000001) should be exceeded before the global (100.0)
                with pytest.raises(BudgetExceededError):
                    agent()

        finally:
            unpatch_openai()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
