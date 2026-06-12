"""Budget context manager for fine-grained control."""

from typing import Optional, Callable
from agentguard.core.budget import BudgetTracker, BudgetSummary
from agentguard.core.enforcer import Enforcer


class Budget:
    """
    Context manager for budget enforcement.

    Provides more control than @guard decorator:
    - Inspect budget mid-run
    - Share budget across functions
    - Attach multiple policies

    Example:
        with Budget(max_usd=2.00, on_breach='warn') as b:
            result = agent.invoke({'query': query})
            print(f"Used ${b.usd_spent:.2f} so far")
            print(f"Remaining: ${b.usd_remaining:.2f}")

        summary = b.summary()
        print(summary.to_json())
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_usd: Optional[float] = None,
        on_breach: Callable | str = "kill",
        session_id: Optional[str] = None,
    ):
        """
        Initialize budget context.

        Args:
            max_tokens: Hard token limit
            max_usd: Hard USD limit
            on_breach: Policy for handling breaches
            session_id: Optional session identifier
        """
        self.tracker = BudgetTracker(
            max_tokens=max_tokens,
            max_usd=max_usd,
            session_id=session_id,
        )
        self.enforcer = Enforcer(self.tracker, on_breach)

    def __enter__(self) -> "Budget":
        """Enter context."""
        # Could set up hooks here (e.g., register enforcer globally)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        # Cleanup if needed
        pass

    def summary(self) -> BudgetSummary:
        """Get full budget summary."""
        return self.tracker.summary()

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed so far."""
        return self.tracker.tokens_used

    @property
    def usd_spent(self) -> float:
        """Total USD spent so far."""
        return self.tracker.usd_spent

    @property
    def usd_remaining(self) -> float:
        """Budget remaining in USD."""
        return self.tracker.usd_remaining

    @property
    def breach_count(self) -> int:
        """Number of times breach policy fired."""
        return self.tracker._breach_count

    @property
    def is_breached(self) -> bool:
        """Whether budget was exceeded."""
        return self.tracker.is_breached
