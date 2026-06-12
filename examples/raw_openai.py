"""Simple example using raw OpenAI SDK with AgentGuard."""

import os
from agentguard.integrations.openai import patch_openai
from agentguard.exceptions import BudgetExceededError

# Note: This is a demonstration example. To run it:
# 1. pip install openai
# 2. Set OPENAI_API_KEY environment variable
# 3. Uncomment the main() call below


def example_small_budget_breach():
    """
    Demonstrates budget enforcement with a small budget that breaches.
    
    This example intentionally sets a budget that will be exceeded,
    showing how AgentGuard prevents the call from being made.
    """
    import openai
    
    # Patch OpenAI with a tiny budget ($0.01)
    patch_openai(
        max_usd=0.01,
        on_breach='kill',
    )
    
    try:
        # Try to make an API call
        # This will likely exceed our $0.01 budget during token estimation
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    'role': 'user',
                    'content': 'Write a long essay about the history of AI and machine learning. Include details about transformers, attention mechanisms, and recent advances.'
                }
            ]
        )
        
        print("Call succeeded (budget wasn't breached)")
        
    except BudgetExceededError as e:
        print(f"\n✅ Budget breach prevented!")
        print(f"   Breach Type: {e.breach_type}")
        print(f"   Tokens Used: {e.tokens_used}")
        print(f"   USD Spent: ${e.usd_spent:.6f}")
        print(f"   Budget Limit: ${e.budget_limit:.4f}")
        print(f"\n   This demonstrates AgentGuard BLOCKING the call before it happens.")


def example_reasonable_budget():
    """
    Demonstrates budget enforcement with a reasonable budget that succeeds.
    """
    import openai
    from agentguard.integrations.openai import unpatch_openai, get_openai_enforcer
    
    # Unpatch from previous example
    unpatch_openai()
    
    # Patch OpenAI with a reasonable budget ($10)
    patch_openai(
        max_usd=10.00,
        on_breach='warn',
    )
    
    try:
        response = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'user',
                    'content': 'Say hello'
                }
            ]
        )
        
        print("\n✅ Call succeeded within budget!")
        print(f"   Response: {response.choices[0].message.content}")
        
        # Get summary from enforcer
        enforcer = get_openai_enforcer()
        if enforcer:
            summary = enforcer.tracker.summary()
            print(f"\n   Budget Summary:")
            print(f"   - Tokens: {summary.total_tokens} / {summary.max_tokens or 'unlimited'}")
            print(f"   - Cost: ${summary.total_usd:.6f} / ${summary.max_usd or 'unlimited'}")
        
    except BudgetExceededError as e:
        print(f"Budget breach prevented: {e}")


def example_multi_call_accumulation():
    """
    Demonstrates budget accumulation across multiple calls.
    
    Shows how AgentGuard tracks cumulative spending and enforces limits.
    """
    import openai
    from agentguard.integrations.openai import unpatch_openai
    
    unpatch_openai()
    
    # Set a $0.05 budget for multiple calls
    patch_openai(
        max_usd=0.05,
        on_breach='warn',
    )
    
    print("\n📊 Making multiple calls within shared budget...")
    
    for i in range(3):
        try:
            response = openai.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'user',
                        'content': f'Question {i+1}: Explain a simple concept in one sentence.'
                    }
                ]
            )
            print(f"   Call {i+1}: ✓ Success")
            
        except BudgetExceededError as e:
            print(f"   Call {i+1}: ✗ Budget exceeded - {e.breach_type}")
            break


if __name__ == '__main__':
    print("=" * 60)
    print("AgentGuard OpenAI Integration Examples")
    print("=" * 60)
    
    print("\n⚠️  Note: These examples require OPENAI_API_KEY to be set")
    print("     and the openai package to be installed.")
    print("\n     To run: pip install openai")
    print("     Then set: export OPENAI_API_KEY='your-key'")
    print("\n" + "=" * 60)
    
    # Uncomment below to run examples with real API key:
    # example_small_budget_breach()
    # example_reasonable_budget()
    # example_multi_call_accumulation()
