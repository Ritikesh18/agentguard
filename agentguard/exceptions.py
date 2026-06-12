"""Exception classes for AgentGuard budget enforcement."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BudgetExceededError(Exception):
    """Raised when a budget limit is breached in kill mode."""

    tokens_used: int
    """Total tokens consumed before breach."""

    usd_spent: float
    """Total USD spent before breach."""

    budget_limit: float | int | None
    """The configured budget limit that was exceeded."""

    breach_type: Optional[Literal["token_limit", "usd_limit"]]
    """Whether token or USD limit was exceeded."""

    agent_id: str | None = None
    """Which agent triggered the breach (if multi-agent)."""

    call_number: int | None = None
    """Which LLM call number triggered the breach."""

    def __str__(self) -> str:
        """Human-readable summary of the breach."""
        limit_name = "token" if self.breach_type == "token_limit" else "USD"
        agent_msg = f" (agent: {self.agent_id})" if self.agent_id else ""
        return (
            f"BudgetExceededError{agent_msg}: {limit_name} limit breached. "
            f"Spent: {self.usd_spent if self.breach_type == 'usd_limit' else self.tokens_used}, "
            f"Limit: {self.budget_limit}"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging."""
        return {
            "tokens_used": self.tokens_used,
            "usd_spent": self.usd_spent,
            "budget_limit": self.budget_limit,
            "breach_type": self.breach_type,
            "agent_id": self.agent_id,
            "call_number": self.call_number,
        }


class BudgetBreachWarning(UserWarning):
    """Warning issued when a budget is breached in warn mode."""

    pass
