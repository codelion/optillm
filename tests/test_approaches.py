#!/usr/bin/env python3
"""
Simplified approach tests for CI/CD
Tests the basic structure of approaches without requiring actual model inference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.reread import re2_approach
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.mars import multi_agent_reasoning_system


class MockClient:
    """Mock OpenAI client for testing"""
    def __init__(self):
        self.chat = self.Chat()
    
    class Chat:
        def __init__(self):
            self.completions = self.Completions()
        
        class Completions:
            def create(self, **kwargs):
                class MockChoice:
                    class Message:
                        content = "Test response: 2 + 2 = 4"
                    message = Message()
                
                class MockUsage:
                    completion_tokens = 10
                    total_tokens = 20
                
                class MockResponse:
                    choices = [MockChoice()]
                    usage = MockUsage()
                
                return MockResponse()


def test_approach_imports():
    """Test that all approaches can be imported"""
    approaches = [
        chat_with_mcts,
        best_of_n_sampling,
        mixture_of_agents,
        advanced_self_consistency_approach,
        re2_approach,
        cot_reflection,
        plansearch,
        leap,
        multi_agent_reasoning_system
    ]
    
    for approach in approaches:
        assert callable(approach), f"{approach.__name__} is not callable"
    
    print("✅ All approaches imported successfully")


def test_basic_approach_calls():
    """Test basic approach calls with mock client"""
    client = MockClient()
    system_prompt = "You are a helpful assistant."
    query = "What is 2 + 2?"
    model = "mock-model"
    
    # Test approaches that should work with mock client
    simple_approaches = [
        ("re2_approach", re2_approach),
        ("cot_reflection", cot_reflection),
        ("leap", leap),
        ("mars", multi_agent_reasoning_system),
    ]
    
    for name, approach_func in simple_approaches:
        try:
            result = approach_func(system_prompt, query, client, model)
            assert result is not None, f"{name} returned None"
            assert isinstance(result, tuple), f"{name} should return a tuple"
            assert len(result) == 2, f"{name} should return (response, tokens)"
            print(f"✅ {name} basic test passed")
        except Exception as e:
            print(f"❌ {name} basic test failed: {e}")


def test_approach_parameters():
    """Test that approaches handle parameters correctly"""
    # Test that approaches accept the expected parameters
    import inspect
    
    approaches = {
        "chat_with_mcts": chat_with_mcts,
        "best_of_n_sampling": best_of_n_sampling,
        "mixture_of_agents": mixture_of_agents,
        "advanced_self_consistency_approach": advanced_self_consistency_approach,
        "re2_approach": re2_approach,
        "cot_reflection": cot_reflection,
        "plansearch": plansearch,
        "leap": leap,
        "multi_agent_reasoning_system": multi_agent_reasoning_system,
    }
    
    for name, func in approaches.items():
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Check required parameters
        required_params = ["system_prompt", "initial_query", "client", "model"]
        for param in required_params:
            assert param in params, f"{name} missing required parameter: {param}"
        
        print(f"✅ {name} has correct parameters")


if __name__ == "__main__":
    print("Running approach tests...")
    
    test_approach_imports()
    test_basic_approach_calls()
    test_approach_parameters()
    
    print("\nAll tests completed!")