"""Example showing setting per-agent budgets with GuardCallback.

This example is illustrative; integrate with your LangGraph graph and pass
agent identifiers via metadata or kwargs (e.g., name="researcher").
"""

from agentguard.integrations.langgraph import GuardCallback


def setup_multiagent_guard(graph):
    per_agent = {
        "researcher": 1.00,
        "writer": 0.50,
        "reviewer": 0.25,
    }
    guard_cb = GuardCallback(per_agent_budgets=per_agent, on_breach="kill")
    graph.callbacks = getattr(graph, "callbacks", []) + [guard_cb]
    return guard_cb


if __name__ == "__main__":
    print("Attach multi-agent GuardCallback to your graph. Use agent name in metadata.")
