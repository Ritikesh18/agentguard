"""Policy handlers for budget breach responses."""

import logging
import asyncio
import warnings
import threading
import uuid
from typing import Callable, Optional, Union, Dict

from agentguard.core.budget import BreachResult, BudgetTracker
from agentguard.exceptions import BudgetExceededError, BudgetBreachWarning

logger = logging.getLogger(__name__)


class PolicyHandler:
    """
    Executes the configured breach response policy.

    Supports built-in policies ('kill', 'warn', 'pause') and custom callables.
    """

    def __init__(self, policy: Union[str, Callable]):
        """
        Initialize policy handler.

        Args:
            policy: 'kill' | 'warn' | 'pause' | callable
                - 'kill': Raise BudgetExceededError
                - 'warn': Log warning, continue execution
                - 'pause': Block thread, wait for resume signal
                - callable: Custom function (breach_result, tracker) -> None
        """
        self.policy = policy

    def handle(
        self,
        breach_result: BreachResult,
        tracker: BudgetTracker,
        estimated_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Execute the breach policy.

        Args:
            breach_result: Result from pre-call check
            tracker: BudgetTracker instance
            estimated_tokens: Estimated tokens for this call
            model: Model identifier
            agent_id: Optional agent identifier

        Raises:
            BudgetExceededError: If policy is 'kill' or custom callable raises
        """
        if not breach_result.is_breach:
            return

        if self.policy == "kill":
            self._handle_kill(breach_result, tracker, estimated_tokens, model, agent_id)
        elif self.policy == "warn":
            self._handle_warn(breach_result, tracker, estimated_tokens, model, agent_id)
        elif self.policy == "pause":
            self._handle_pause(breach_result, tracker, estimated_tokens, model, agent_id)
        elif callable(self.policy):
            self._handle_custom(
                self.policy, breach_result, tracker, estimated_tokens, model, agent_id
            )
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    @staticmethod
    def _handle_kill(
        breach_result: BreachResult,
        tracker: BudgetTracker,
        estimated_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Kill policy: raise BudgetExceededError."""
        error = BudgetExceededError(
            tokens_used=tracker.tokens_used,
            usd_spent=tracker.usd_spent,
            budget_limit=(
                tracker.max_usd if breach_result.breach_type == "usd_limit" else tracker.max_tokens
            ),
            breach_type=breach_result.breach_type,
            agent_id=agent_id,
            call_number=len(tracker._calls) + 1,
        )
        logger.error(f"Budget breach (kill): {error}")
        raise error

    @staticmethod
    def _handle_warn(
        breach_result: BreachResult,
        tracker: BudgetTracker,
        estimated_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Warn policy: log warning and continue."""
        agent_msg = f" (agent: {agent_id})" if agent_id else ""
        message = (
            f"Budget breach warning{agent_msg}: "
            f"{breach_result.breach_type} exceeded. "
            f"Spent: {tracker.usd_spent:.4f} USD / {tracker.tokens_used} tokens"
        )
        warnings.warn(message, BudgetBreachWarning, stacklevel=3)
        logger.warning(message)

    @staticmethod
    def _handle_pause(
        breach_result: BreachResult,
        tracker: BudgetTracker,
        estimated_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Pause policy: block execution and wait for resume signal."""
        agent_msg = f" (agent: {agent_id})" if agent_id else ""
        logger.warning(
            f"Budget breach (paused){agent_msg}: "
            f"{breach_result.breach_type} limit reached. "
            f"Execution paused. Send resume signal to continue."
        )
        # Create a pause token and block synchronously until resumed.
        pause_id = PauseManager.create_pause()
        # Block the current thread until resumed
        PauseManager.wait(pause_id)

    @staticmethod
    def _handle_custom(
        custom_fn: Callable,
        breach_result: BreachResult,
        tracker: BudgetTracker,
        estimated_tokens: int,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Custom policy: call user-provided function."""
        try:
            custom_fn(breach_result, tracker)
        except Exception as e:
            logger.error(f"Custom policy raised exception: {e}")
            raise


class PauseManager:
    """Manage pause tokens and blocking/resume behaviour.

    This uses threading.Event to block the current thread until a resume is issued.
    For convenience tests and tools can call `PauseManager.resume_all()` to
    release all paused callers.
    """

    _pauses: Dict[str, threading.Event] = {}
    _lock = threading.Lock()

    @classmethod
    def create_pause(cls) -> str:
        pause_id = str(uuid.uuid4())
        ev = threading.Event()
        with cls._lock:
            cls._pauses[pause_id] = ev
        logger.debug(f"PauseManager: created pause {pause_id}")
        return pause_id

    @classmethod
    def wait(cls, pause_id: str, timeout: Optional[float] = None) -> bool:
        ev = None
        with cls._lock:
            ev = cls._pauses.get(pause_id)
        if ev is None:
            return True
        # Block until set (or timeout)
        logger.debug(f"PauseManager: waiting on {pause_id}")
        result = ev.wait(timeout=timeout)
        logger.debug(f"PauseManager: wait returned {result} for {pause_id}")
        # Cleanup
        with cls._lock:
            cls._pauses.pop(pause_id, None)
        return result

    @classmethod
    def resume(cls, pause_id: str) -> None:
        with cls._lock:
            ev = cls._pauses.get(pause_id)
        if ev:
            ev.set()

    @classmethod
    def resume_all(cls) -> None:
        with cls._lock:
            items = list(cls._pauses.items())
        logger.debug(f"PauseManager: resuming all {len(items)} pauses")
        for pid, ev in items:
            logger.debug(f"PauseManager: resuming {pid}")
            ev.set()
