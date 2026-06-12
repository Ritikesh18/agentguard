"""Example showing how to wrap an Anthropic-like callable with AgentGuard."""

from agentguard.integrations.anthropic import wrap_completion_fn


def fake_anthropic_create(prompt: str):
    class Resp:
        def __init__(self):
            self.completion = "Fake response"
            self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    return Resp()


wrapped = wrap_completion_fn(fake_anthropic_create, max_usd=1.0, on_breach="warn")


if __name__ == "__main__":
    r = wrapped(prompt="Hello Anthropic")
    print(r.completion)
