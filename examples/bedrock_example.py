"""Example showing how to wrap a Bedrock-like callable with AgentGuard."""

from agentguard.integrations.bedrock import wrap_bedrock_fn


def fake_bedrock_invoke(modelId: str, input: str):
    class Resp:
        def __init__(self):
            self.output = "Fake bedrock output"
            self.usage = type("U", (), {"prompt_tokens": 8, "completion_tokens": 6})()

    return Resp()


wrapped = wrap_bedrock_fn(fake_bedrock_invoke, max_tokens=1000, on_breach="warn")


if __name__ == "__main__":
    r = wrapped(modelId="amazon.titan", input="Hello Bedrock")
    print(r.output)
