"""Small demo showing how to integrate AgentGuard into a governance script.

This script demonstrates using the `Budget` context manager and the
`guard` decorator to protect an example agent run. It's intentionally
minimal so you can adapt it into your own `ai_governance.py` flow.
"""

from agentguard import guard, Budget
import time


@guard(max_usd=0.01, on_breach="warn")
def simple_agent_run(prompt: str) -> str:
	# Replace with actual LLM call; here we simulate an API call and usage
	time.sleep(0.01)
	return f"Echo: {prompt}"


def main():
	with Budget(max_usd=0.02, on_breach="warn") as b:
		print("Starting governed run")
		out = simple_agent_run("Hello governance")
		print(out)
		print("Tokens used:", b.tokens_used)
		print("USD spent:", b.usd_spent)


if __name__ == "__main__":
	main()
