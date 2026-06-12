"""Tests for pricing module."""

import pytest
from agentguard.core.pricing import get_price, estimate_cost, PRICING


class TestPricing:
    """Test pricing table and calculations."""

    def test_hardcoded_models_exist(self):
        """All 12+ models are in pricing table."""
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "gpt-4-turbo" in PRICING
        assert "gpt-3.5-turbo" in PRICING
        assert "claude-3-5-sonnet" in PRICING
        assert "claude-3-5-haiku" in PRICING
        assert "claude-3-opus" in PRICING
        assert "azure/gpt-4o" in PRICING
        assert "bedrock/claude-3-5-sonnet" in PRICING

    def test_exact_model_match(self):
        """Exact model names return correct pricing."""
        price = get_price("gpt-4o")

        assert "input" in price
        assert "output" in price
        assert price["input"] == 0.0025
        assert price["output"] == 0.01

    def test_fuzzy_model_match_version_suffix(self):
        """Fuzzy matching strips version suffixes."""
        # gpt-4o-2024-11-20 should match gpt-4o
        price = get_price("gpt-4o-2024-11-20")

        assert "input" in price
        assert price["input"] == 0.0025

    def test_fuzzy_model_match_date_variant(self):
        """Fuzzy matching works with any date format."""
        # Test multiple date formats
        for variant in [
            "gpt-4o-2024-05-13",
            "gpt-4o-2025-01-01",
            "gpt-4-turbo-2024-04-09",
        ]:
            price = get_price(variant)
            assert "input" in price

    def test_unknown_model_falls_back_to_default(self):
        """Unknown models use __default__ pricing."""
        price = get_price("unknown-model-xyz-123")

        assert "input" in price
        assert price["input"] == PRICING["__default__"]["input"]

    def test_estimate_cost_input_only(self):
        """Cost estimation for input tokens only."""
        cost = estimate_cost("gpt-4o", prompt_tokens=1000)

        # gpt-4o input: 0.0025 per 1k tokens
        assert cost == pytest.approx(0.0025, abs=0.0001)

    def test_estimate_cost_with_completion(self):
        """Cost estimation for prompt + completion."""
        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)

        # gpt-4o: (1000/1000)*0.0025 + (500/1000)*0.01 = 0.0025 + 0.005 = 0.0075
        assert cost == pytest.approx(0.0075, abs=0.0001)

    def test_estimate_cost_different_models(self):
        """Different models have different costs."""
        cost_4o = estimate_cost("gpt-4o", prompt_tokens=1000)
        cost_mini = estimate_cost("gpt-4o-mini", prompt_tokens=1000)
        cost_3_5 = estimate_cost("gpt-3.5-turbo", prompt_tokens=1000)

        # 4o should be expensive
        assert cost_4o > cost_mini
        assert cost_4o > cost_3_5

    def test_estimate_cost_zero_tokens(self):
        """Zero tokens costs zero."""
        cost = estimate_cost("gpt-4o", prompt_tokens=0)
        assert cost == 0.0

    def test_azure_pricing_same_as_openai(self):
        """Azure models use same pricing as OpenAI."""
        price_openai = get_price("gpt-4o")
        price_azure = get_price("azure/gpt-4o")

        assert price_openai == price_azure

    def test_bedrock_pricing(self):
        """Bedrock models have pricing."""
        price = get_price("bedrock/claude-3-5-sonnet")

        assert "input" in price
        assert price["input"] > 0
        assert price["output"] > 0

    def test_anthropic_pricing(self):
        """Anthropic models have pricing."""
        for model in ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"]:
            price = get_price(model)
            assert "input" in price
            assert price["input"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
