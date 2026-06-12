"""Tests for pause policy behavior."""

import threading
import time

import pytest

from agentguard.core.policies import PauseManager
from agentguard.core.budget import BudgetTracker
from agentguard.core.enforcer import Enforcer


def test_pause_policy_blocks_and_resumes():
    tracker = BudgetTracker(max_usd=0.00001)  # tiny budget to force breach
    enforcer = Enforcer(tracker, "pause")

    # Create a large message to trigger USD breach
    messages = [{"role": "user", "content": "word " * 5000}]
    model = "gpt-4o"

    finished = threading.Event()

    def caller():
        try:
            enforcer.pre_call_check(messages, model)
        finally:
            finished.set()

    t = threading.Thread(target=caller)
    t.start()

    # Wait for the pause to be registered
    waited = 0.0
    while waited < 1.0 and not PauseManager._pauses:
        time.sleep(0.05)
        waited += 0.05

    assert PauseManager._pauses, "Pause should be registered"
    assert not finished.is_set(), "Caller should be paused"

    # Resume all paused callers
    PauseManager.resume_all()

    # Wait for caller to finish
    t.join(timeout=2.0)
    assert finished.is_set(), "Caller should have resumed and finished"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
