"""Pre-call enforcement gatekeeper for budget limits."""

import logging
from typing import Optional, Callable, Any
import tiktoken

from agentguard.core.budget import BudgetTracker, CallRecord
from agentguard.core.policies import PolicyHandler
from agentguard.core.pricing import estimate_cost

logger = logging.getLogger(__name__)


class Enforcer:
    """
    Pre-call enforcement gatekeeper.

    Checks budget before each LLM API call. If breach detected, fires policy
    (which may raise an exception, log a warning, or execute custom logic).
    """

    def __init__(self, tracker: BudgetTracker, policy: str | Callable):
        """
        Initialize enforcer.

        Args:
            tracker: BudgetTracker instance
            policy: Breach policy ('kill', 'warn', 'pause', or callable)
        """
        self.tracker = tracker
        self.policy_handler = PolicyHandler(policy)

    def pre_call_check(
        self,
        messages: list,
        model: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Check budget before making an LLM call.

        This is the main enforcement point. Counts tokens, estimates cost,
        and fires the policy if budget would be exceeded.

        Args:
            messages: LLM messages (list of dicts with 'role', 'content')
            model: Model identifier
            agent_id: Optional agent identifier for multi-agent tracking

        Raises:
            BudgetExceededError: If policy is 'kill' and breach occurs
        """
        # Estimate tokens for this call
        estimated_prompt_tokens = self._count_tokens(messages, model)

        # Check against budget
        breach_result = self.tracker.check_pre_call(estimated_prompt_tokens, model, agent_id)

        # If breach, fire policy
        if breach_result.is_breach:
            self.policy_handler.handle(
                breach_result, self.tracker, estimated_prompt_tokens, model, agent_id
            )

    def post_call_record(
        self,
        response: Any,
        model: str,
        agent_id: Optional[str] = None,
    ) -> CallRecord:
        """
        Record actual LLM call after it completes.

        Args:
            response: LLM response object with .usage.prompt_tokens and .usage.completion_tokens
            model: Model identifier
            agent_id: Optional agent identifier

        Returns:
            CallRecord with accumulated totals
        """
        # Extract token counts from response
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Record in tracker
        record = self.tracker.record_call(prompt_tokens, completion_tokens, model, agent_id)

        logger.debug(
            f"Recorded LLM call: {prompt_tokens} prompt + "
            f"{completion_tokens} completion tokens, "
            f"cost: ${record.cost_usd:.4f}"
        )

        return record

    @staticmethod
    def _count_tokens(messages: list, model: str) -> int:
        """
        Count tokens in messages using appropriate tokenizer.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier

        Returns:
            Estimated token count
        """
        # Use tiktoken for OpenAI/Azure models
        if "gpt" in model or "azure" in model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback for unknown OpenAI models
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

            # Count tokens in messages
            tokens = 0
            for message in messages:
                # Each message starts with a token overhead
                tokens += 4  # message start marker
                for key, value in message.items():
                    tokens += len(enc.encode(str(value)))

            # Add final completion token overhead
            tokens += 3

            return tokens

        # Claude models: approximate (1 token ~ 4 chars for English)
        elif "claude" in model:
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return max(1, total_chars // 4)

        # Bedrock models: approximate
        elif "bedrock" in model:
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return max(1, total_chars // 4)

        # Fallback: character-based approximation
        else:
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return max(1, total_chars // 4)
