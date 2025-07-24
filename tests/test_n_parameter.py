#!/usr/bin/env python3
"""
Test script to verify n parameter works correctly with optillm
"""

import os
import sys
from openai import OpenAI
import json

def test_n_parameter(model="gpt-4o-mini", n_values=[1, 2, 3]):
    """
    Test the n parameter with different values
    """
    # Initialize OpenAI client with optillm proxy
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="http://localhost:8000/v1"
    )
    
    test_prompt = "Write a haiku about coding"
    
    for n in n_values:
        print(f"\nTesting n={n} with model {model}")
        print("-" * 50)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a creative poet."},
                    {"role": "user", "content": test_prompt}
                ],
                n=n,
                temperature=0.8,
                max_tokens=100
            )
            
            # Check response structure
            print(f"Response type: {type(response)}")
            print(f"Number of choices: {len(response.choices)}")
            
            # Print all generated responses
            for i, choice in enumerate(response.choices):
                print(f"\nChoice {i+1}:")
                print(choice.message.content)
            
            # Verify we got the expected number of responses
            if len(response.choices) == n:
                print(f"\n✅ SUCCESS: Got {n} responses as expected")
            else:
                print(f"\n❌ FAIL: Expected {n} responses, got {len(response.choices)}")
                
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")

def main():
    """
    Main test function
    """
    print("Testing n parameter support in optillm")
    print("=" * 50)
    
    # Test with different models if available
    models_to_test = []
    
    # Check for available models
    if os.environ.get("OPENAI_API_KEY"):
        models_to_test.append("gpt-4o-mini")
    
    # Check for MLX models
    if os.environ.get("OPTILLM_API_KEY") == "optillm":
        # Add MLX model if running with local inference
        models_to_test.append("Qwen/Qwen2.5-1.5B-Instruct")
    
    if not models_to_test:
        print("No models available to test. Set OPENAI_API_KEY or OPTILLM_API_KEY=optillm")
        return
    
    for model in models_to_test:
        print(f"\n\nTesting model: {model}")
        print("=" * 50)
        test_n_parameter(model)

if __name__ == "__main__":
    main()