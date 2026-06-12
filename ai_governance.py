"""Demo: AgentGuard governance script.

Shows three patterns:
  1. @guard decorator — budget resets each call, breach kills execution
  2. Budget context manager — shared budget across multiple calls, warn on breach
  3. Simulated breach — how BudgetExceededError surfaces and what it carries

Run with:  python ai_governance.py
No API key required — all LLM calls are simulated via the enforcer directly.
"""

from agentguard import guard, Budget
from agentguard.core.enforcer import Enforcer
from agentguard.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# Pattern 1: @guard decorator
# ---------------------------------------------------------------------------


@guard(max_usd=0.005, on_breach="kill")
def research_agent(prompt: str) -> str:
    """Simulates a research agent that makes an LLM call."""
    enforcer: Enforcer = __import__(
        "agentguard.decorator", fromlist=["get_active_enforcer"]
    ).get_active_enforcer()
    # Simulate: record a small call (100 prompt + 50 completion tokens on gpt-4o-mini)
    enforcer.tracker.record_call(100, 50, "gpt-4o-mini", agent_id="researcher")
    return f"Research result for: {prompt}"


# ---------------------------------------------------------------------------
# Pattern 2: Budget context manager with shared budget
# ---------------------------------------------------------------------------


def run_pipeline():
    with Budget(max_usd=0.001, on_breach="warn") as b:
        # Simulate two cheap calls — together they may breach the tiny budget
        b.enforcer.tracker.record_call(200, 100, "gpt-4o-mini", agent_id="planner")
        b.enforcer.tracker.record_call(300, 150, "gpt-4o-mini", agent_id="writer")
        return b.summary()


# ---------------------------------------------------------------------------
# Pattern 3: Explicit breach to show error fields
# ---------------------------------------------------------------------------


def simulate_breach():
    b = Budget(max_usd=0.00001, on_breach="kill")
    # Pre-fill the tracker so the next check tips it over
    b.tracker.record_call(1000, 500, "gpt-4o")
    breach = b.tracker.check_pre_call(500, "gpt-4o")
    if breach.is_breach:
        b.enforcer.policy_handler.handle(breach, b.tracker, 500, "gpt-4o", agent_id="loop-agent")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("=" * 55)
    print("AgentGuard Governance Demo")
    print("=" * 55)

    # 1. Decorator pattern
    print("\n[1] @guard decorator (max $0.005, kill on breach)")
    result = research_agent("climate change mitigation")
    print(f"    Result  : {result}")
    print(f"    (budget enforcement was active during the call)")

    # 2. Context manager pattern
    print("\n[2] Budget context manager (max $0.001, warn on breach)")
    summary = run_pipeline()
    print(f"    Tokens  : {summary.total_tokens}")
    print(f"    Cost    : ${summary.total_usd:.6f}")
    print(f"    Breaches: {summary.breaches}")

    # 3. Kill-on-breach pattern
    print("\n[3] Simulated budget breach (kill policy)")
    try:
        simulate_breach()
    except BudgetExceededError as e:
        print(f"    Caught BudgetExceededError")
        print(f"    breach_type : {e.breach_type}")
        print(f"    usd_spent   : ${e.usd_spent:.6f}")
        print(f"    budget_limit: ${e.budget_limit}")
        print(f"    agent_id    : {e.agent_id}")

    print("\n" + "=" * 55)
    print("All three patterns demonstrated successfully.")


if __name__ == "__main__":
    main()
