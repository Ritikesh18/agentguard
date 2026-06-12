# AgentGuard

### Pre-execution budget enforcement for agentic AI.

A team's 4-agent LangGraph pipeline ran in an infinite loop for 11 days.

**$47,000 bill.** They had monitoring. They didn't have enforcement.

**AgentGuard is the missing piece.**

---

## Install

```bash
pip install agentguard
```

For LangGraph/LangChain integration:
```bash
pip install agentguard[langgraph]
```

---

## 3 Lines of Protection

```python
from agentguard import guard

@guard(max_usd=5.00, on_breach='kill')
def run_agent(query):
    # your agent code here — all LLM calls are now budget-enforced
    response = openai.chat.completions.create(...)
    return response.choices[0].message.content
```

That's it. All LLM calls inside `run_agent` are now budget-enforced.

---

## How It Works

AgentGuard intercepts every outbound LLM call **BEFORE** the HTTP request.

It checks your remaining budget. If exceeded, it fires your breach policy.

```
┌─────────────────────────────────────────────────────┐
│  Agent Code Calls LLM                               │
│  (e.g., openai.chat.completions.create(...))        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  ✋ AgentGuard Intercepts                            │
│  - Count tokens using tiktoken                      │
│  - Estimate cost from pricing table                 │
│  - Check: spent + estimated > budget?               │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼ PASS                ▼ BREACH
   Call proceeds      Fire policy (kill/warn/pause)
        │                     │
        ▼                     ▼
  Response saved      ✗ No API call made
  Tokens recorded     ✗ No tokens spent
  Cost logged         ✗ No $ wasted
```

**No call is made. No token is spent. No dollar is wasted.**

---

## Key Differences from Other Tools

| Tool | What It Does | When It Acts |
|------|--------------|--------------|
| **Langfuse / Phoenix** | Observability, trace storage, dashboards | *After* calls complete (post-call logging) |
| **LangSmith** | Monitoring, evaluation, debugging | *After* calls complete (post-call analysis) |
| **AgentGuard** | Budget enforcement | *Before* calls are made (pre-call blocking) |

Those tools tell you what happened. **AgentGuard prevents what should not happen.**

---

## API Reference

### 1. The `@guard` Decorator (Simplest)

```python
from agentguard import guard

@guard(
    max_tokens=50_000,        # hard token limit
    max_usd=5.00,             # hard USD limit
    on_breach='kill',         # 'kill' | 'warn' | 'pause' | callable
    model='gpt-4o',           # optional: used for pricing lookup
    agent_id='orchestrator',  # optional: for multi-agent tracking
)
def run_research_agent(query: str) -> str:
    # All LLM calls inside are budget-enforced
    ...

# Works with async too
@guard(max_usd=3.00, on_breach='kill')
async def run_async_agent(query: str) -> str:
    ...
```

### 2. Budget Context Manager (More Control)

```python
from agentguard import Budget

with Budget(max_usd=2.00, on_breach='warn') as b:
    result = agent.invoke({'query': query})
    
    # Check budget mid-run
    print(b.tokens_used)       # int: total tokens so far
    print(b.usd_spent)         # float: total USD so far
    print(b.usd_remaining)     # float: remaining budget
    print(b.breach_count)      # int: breach policy fires
    print(b.is_breached)       # bool: budget exceeded?

# After context exits
summary = b.summary()          # BudgetSummary dataclass
print(summary.to_dict())       # serializable for logging
```

### 3. Raw OpenAI Wrapper (No Refactoring)

```python
from agentguard.integrations.openai import patch_openai
import openai

# Call once at startup — patches globally
patch_openai(max_usd=10.00, on_breach='kill')

# All subsequent openai calls are budget-enforced
# No changes needed to existing code
response = openai.chat.completions.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### 4. Handling Breaches

```python
from agentguard import guard
from agentguard.exceptions import BudgetExceededError

@guard(max_usd=5.00, on_breach='kill')
def run_agent(query):
    ...

try:
    result = run_agent(query)
except BudgetExceededError as e:
    print(f"Tokens: {e.tokens_used}")
    print(f"Spent: ${e.usd_spent:.2f}")
    print(f"Limit: ${e.budget_limit:.2f}")
    print(f"Type: {e.breach_type}")        # 'token_limit' | 'usd_limit'
    print(f"Agent: {e.agent_id}")
    print(f"Call #: {e.call_number}")
```

---

## Supported Models & Pricing

AgentGuard includes hardcoded pricing for 25+ models:

**OpenAI — GPT-4o:**
- `gpt-4o` ($0.0025 / 1k input, $0.01 / 1k output)
- `gpt-4o-mini` ($0.00015 / 1k input, $0.0006 / 1k output)

**OpenAI — GPT-4.1 (2025):**
- `gpt-4.1` ($0.002 / 1k input, $0.008 / 1k output)
- `gpt-4.1-mini` ($0.0004 / 1k input, $0.0016 / 1k output)
- `gpt-4.1-nano` ($0.0001 / 1k input, $0.0004 / 1k output)

**OpenAI — Reasoning:**
- `o1` ($0.015 / 1k input, $0.06 / 1k output)
- `o3` ($0.01 / 1k input, $0.04 / 1k output)
- `o3-mini` / `o4-mini` ($0.0011 / 1k input, $0.0044 / 1k output)

**OpenAI — Legacy:**
- `gpt-4-turbo` ($0.01 / 1k input, $0.03 / 1k output)
- `gpt-3.5-turbo` ($0.0005 / 1k input, $0.0015 / 1k output)

**Anthropic — Claude 4 (2025/2026):**
- `claude-opus-4` ($0.015 / 1k input, $0.075 / 1k output)
- `claude-sonnet-4` ($0.003 / 1k input, $0.015 / 1k output)

**Anthropic — Claude 3.7 / 3.5:**
- `claude-3-7-sonnet` ($0.003 / 1k input, $0.015 / 1k output)
- `claude-3-5-sonnet` ($0.003 / 1k input, $0.015 / 1k output)
- `claude-3-5-haiku` ($0.0008 / 1k input, $0.004 / 1k output)
- `claude-3-opus` ($0.015 / 1k input, $0.075 / 1k output)

**Azure & AWS Bedrock:**
- `azure/gpt-4o`, `azure/gpt-4o-mini`, `azure/gpt-4.1`
- `bedrock/claude-3-5-sonnet`, `bedrock/claude-3-7-sonnet`
- `bedrock/llama3-70b`, `bedrock/llama3-8b`

**Fuzzy matching:** Model names with date suffixes (e.g., `gpt-4o-2024-11-20`, `claude-sonnet-4-20250514`) automatically match their base model.

---

## Breach Policies

### 'kill' (Default)
Raises `BudgetExceededError` and stops execution. **Safest.**

```python
@guard(max_usd=5.00, on_breach='kill')
def agent():
    # Breach → BudgetExceededError raised, execution stops
```

### 'warn'
Logs warning to stderr and continues execution.

```python
@guard(max_usd=5.00, on_breach='warn')
def agent():
    # Breach → warning logged, execution continues
```

### 'pause'
Blocks thread/task, waits for resume signal.

```python
@guard(max_usd=5.00, on_breach='pause')
def agent():
    # Breach → execution paused, waiting for external resume
```

### Custom Callable
Supply your own function:

```python
def my_breach_handler(breach_result, tracker):
    send_slack_alert(f'Budget breach: ${tracker.usd_spent:.2f}')
    if tracker.usd_spent > 20.00:
        raise BudgetExceededError(...)  # escalate
    # else: just alert and continue

@guard(max_usd=5.00, on_breach=my_breach_handler)
def agent():
    ...
```

---

## Examples

See `examples/` folder:

1. **`raw_openai.py`** — Minimal OpenAI example with budget breach
2. **`langgraph_basic.py`** — Single-agent LangGraph with GuardCallback
3. **`langgraph_multiagent.py`** — 3-agent graph with per-agent budgets

### Provider integrations

Anthropic and AWS Bedrock helper wrappers are provided as lightweight adapters:

```python
from agentguard.integrations.anthropic import wrap_completion_fn
wrapped = wrap_completion_fn(client.completions.create, max_usd=5.0)
resp = wrapped(prompt="Hello")
```

```python
from agentguard.integrations.bedrock import wrap_bedrock_fn
wrapped = wrap_bedrock_fn(client.invoke_model, max_usd=2.0)
resp = wrapped(modelId='amazon.titan', input='Hello')
```

See `examples/anthropic_example.py` and `examples/bedrock_example.py` for runnable examples.

---

## Why AgentGuard?

### Problems It Solves

**Agent Loops** — Two agents call each other without termination condition. Infinite back-and-forth.

**Token Bloat** — Context windows grow unbounded across recursive tool calls. Each iteration adds more tokens.

**Runaway Parallelism** — Multiple agent branches each spawn sub-agents, all consuming full token budgets.

**Silent Cost Accumulation** — No alert fires until the billing cycle closes. By then, $47K in damage.

### Why Pre-Execution Matters

Other observability tools react *after* damage. AgentGuard acts *before*:

- **Pre-execution:** Block the call before it happens ✓ Zero tokens spent
- **Post-execution:** Log what happened after payment ✗ $47K already charged

---

## Design Principles

1. **3 lines of code** — Simplest integration should be a decorator
2. **Framework-agnostic** — Works with LangGraph, CrewAI, raw SDK, anything
3. **Pre-execution enforcement** — Block before the HTTP request, not after
4. **Zero telemetry** — Runs entirely locally, no data leaves your machine
5. **MIT licensed** — Fully open source

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_budget.py -v

# Coverage
pytest tests/ --cov=agentguard
```

---

## License

MIT License — See LICENSE file

---

## Contributing

Contributions welcome! Areas for help:

- [ ] More model pricing updates (o1, o3, gpt-4.1, Claude 4 series)
- [ ] Async OpenAI support (`AsyncOpenAI` client patching)
- [ ] Dashboard/UI (nice-to-have)
- [ ] More LangGraph examples with real graph execution

---

## Contact

Built by **Ritikesh Choube**

Questions? Open an issue on [GitHub](https://github.com/ritikesh18/agentguard)

---

### The $47K Story

A Thoughtworks team deployed a 4-agent LangGraph pipeline to production.

One of the agents entered a loop, calling the other three agents recursively.

They had LangSmith monitoring. They saw the traces. By the time they saw what was happening, the agents had made 10,000+ calls.

**Bill: $47,000**

**Root cause:** No way to *enforce* a budget before the call happens.

AgentGuard prevents this.
