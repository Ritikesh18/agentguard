"""Simple LangGraph / LangChain example using GuardCallback.

This is a minimal example showing how to attach GuardCallback to a graph's
callbacks list. This script is illustrative; it assumes you have a graph
or chain object to attach callbacks to.
"""

from agentguard.integrations.langgraph import GuardCallback


def attach_guard_to_graph(graph):
    """Attach GuardCallback to a hypothetical graph's callbacks list."""
    guard_cb = GuardCallback(max_usd=1.00, on_breach='kill')
    # Example: graph.callbacks = existing + [guard_cb]
    graph.callbacks = getattr(graph, 'callbacks', []) + [guard_cb]
    return guard_cb


if __name__ == '__main__':
    print("This is an example. Import GuardCallback and attach to your graph's callbacks list.")
    print("See tests/integrations/test_langgraph.py for a programmatic example.")
