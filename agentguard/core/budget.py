"""Budget tracking and accounting for agent LLM calls."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Union, Literal
import json

from agentguard.core.pricing import estimate_cost


@dataclass
class CallRecord:
    """Record of a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    model: str
    agent_id: Optional[str] = None
    cost_usd: float = 0.0

    def __post_init__(self):
        """Calculate cost automatically."""
        self.cost_usd = estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class BudgetSummary:
    """Summary of budget usage."""

    total_tokens: int
    total_usd: float
    max_tokens: Optional[int]
    max_usd: Optional[float]
    calls_made: int
    breaches: int
    per_agent_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "total_usd": self.total_usd,
            "max_tokens": self.max_tokens,
            "max_usd": self.max_usd,
            "calls_made": self.calls_made,
            "breaches": self.breaches,
            "per_agent_breakdown": self.per_agent_breakdown,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


@dataclass
class BreachResult:
    """Result of a pre-call budget check."""

    is_breach: bool
    breach_type: Optional[Literal["token_limit", "usd_limit"]] = (
        None  # 'token_limit' or 'usd_limit'
    )
    overage_usd: float = 0.0
    overage_tokens: int = 0


class BudgetTracker:
    """
    Central budget tracking and enforcement object.
    Thread-safe. Supports per-agent sub-budgets.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_usd: Optional[float] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize budget tracker.

        Args:
            max_tokens: Hard token limit across all calls
            max_usd: Hard USD limit across all calls
            session_id: Optional session identifier for grouping runs
        """
        if max_tokens is None and max_usd is None:
            raise ValueError("Must specify either max_tokens or max_usd")

        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.session_id = session_id

        self._lock = Lock()
        self._total_tokens = 0
        self._total_usd = 0.0
        self._calls: list[CallRecord] = []
        self._breach_count = 0
        self._per_agent_tokens: dict[str, int] = {}
        self._per_agent_usd: dict[str, float] = {}

    def record_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> CallRecord:
        """
        Record an LLM call after it completes.

        Args:
            prompt_tokens: Input tokens consumed
            completion_tokens: Output tokens generated
            model: Model identifier
            agent_id: Optional agent identifier for multi-agent tracking

        Returns:
            CallRecord with accumulated totals
        """
        record = CallRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            agent_id=agent_id,
        )

        with self._lock:
            self._calls.append(record)
            self._total_tokens += prompt_tokens + completion_tokens
            self._total_usd += record.cost_usd

            if agent_id:
                self._per_agent_tokens[agent_id] = (
                    self._per_agent_tokens.get(agent_id, 0) + prompt_tokens + completion_tokens
                )
                self._per_agent_usd[agent_id] = (
                    self._per_agent_usd.get(agent_id, 0.0) + record.cost_usd
                )

        return record

    def check_pre_call(
        self,
        estimated_prompt_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> BreachResult:
        """
        Check if the next call would breach budget (before making the call).

        Args:
            estimated_prompt_tokens: Estimated input tokens for next call
            model: Model identifier
            agent_id: Optional agent identifier for multi-agent tracking

        Returns:
            BreachResult indicating if breach would occur
        """
        # Estimate cost with 0 completion tokens (we don't know yet)
        estimated_cost = estimate_cost(model, estimated_prompt_tokens, 0)

        with self._lock:
            # Check token limit
            if self.max_tokens is not None:
                projected_tokens = self._total_tokens + estimated_prompt_tokens
                if projected_tokens > self.max_tokens:
                    overage_tokens = projected_tokens - self.max_tokens
                    self._breach_count += 1
                    return BreachResult(
                        is_breach=True,
                        breach_type="token_limit",
                        overage_tokens=overage_tokens,
                    )

            # Check USD limit
            if self.max_usd is not None:
                projected_usd = self._total_usd + estimated_cost
                if projected_usd > self.max_usd:
                    overage_usd = projected_usd - self.max_usd
                    self._breach_count += 1
                    return BreachResult(
                        is_breach=True,
                        breach_type="usd_limit",
                        overage_usd=overage_usd,
                    )

        return BreachResult(is_breach=False)

    def summary(self) -> BudgetSummary:
        """Get full budget summary."""
        with self._lock:
            return BudgetSummary(
                total_tokens=self._total_tokens,
                total_usd=self._total_usd,
                max_tokens=self.max_tokens,
                max_usd=self.max_usd,
                calls_made=len(self._calls),
                breaches=self._breach_count,
                per_agent_breakdown={
                    agent_id: {
                        "tokens": self._per_agent_tokens.get(agent_id, 0),
                        "usd": self._per_agent_usd.get(agent_id, 0.0),
                    }
                    for agent_id in set(self._per_agent_tokens.keys())
                    | set(self._per_agent_usd.keys())
                },
            )

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed so far."""
        with self._lock:
            return self._total_tokens

    @property
    def usd_spent(self) -> float:
        """Total USD spent so far."""
        with self._lock:
            return self._total_usd

    @property
    def usd_remaining(self) -> float:
        """Budget remaining in USD."""
        if self.max_usd is None:
            return float("inf")
        return max(0, self.max_usd - self.usd_spent)

    @property
    def is_breached(self) -> bool:
        """Whether budget was exceeded."""
        return self._breach_count > 0
