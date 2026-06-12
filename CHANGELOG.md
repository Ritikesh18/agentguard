# Changelog

## 0.1.0 - 2026-06-12

- Initial working core: BudgetTracker, Enforcer, Policies, Pricing
- `@guard` decorator (thread-safe, per-call budget) and `Budget` context manager
- OpenAI integration via `patch_openai`; `@guard` now automatically enforces when OpenAI is patched
- Anthropic integration via `wrap_completion_fn`
- AWS Bedrock integration via `wrap_bedrock_fn`
- LangGraph `GuardCallback` integration with global and per-agent budgets
- Pause policy with `PauseManager` (block/resume thread)
- 41 tests passing (budget, enforcer, pricing, pause, LangGraph)
- GitHub Actions CI workflow (pytest on 3.10/3.11/3.12, black, mypy) and PyPI publish workflow
