#!/usr/bin/env python3
"""
Simple tests for reasoning token functionality
Focuses on unit tests that don't require complex mocking
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
from optillm import count_reasoning_tokens as optillm_count
from optillm.inference import count_reasoning_tokens as inference_count


class TestReasoningTokensCore(unittest.TestCase):
    """Test core reasoning token functionality"""
    
    def test_count_reasoning_tokens_with_think_tags(self):
        """Test counting tokens in think tags"""
        text = "<think>Let me think about this problem step by step</think>The answer is 42"
        
        result1 = optillm_count(text)
        result2 = inference_count(text)
        
        self.assertGreater(result1, 0, "Should count tokens in think tags")
        self.assertEqual(result1, result2, "Both functions should return same result")
    
    def test_count_reasoning_tokens_without_think_tags(self):
        """Test with text that has no think tags"""
        text = "This is just a regular response without any thinking"
        
        result1 = optillm_count(text)
        result2 = inference_count(text)
        
        self.assertEqual(result1, 0, "Should return 0 for text without think tags")
        self.assertEqual(result2, 0, "Should return 0 for text without think tags")
    
    def test_count_reasoning_tokens_multiple_blocks(self):
        """Test with multiple think tag blocks"""
        text = """
        <think>First block of reasoning</think>
        Some output here
        <think>Second block with more reasoning</think>
        Final answer
        """
        
        result = optillm_count(text)
        self.assertGreater(result, 0, "Should count tokens from multiple blocks")
    
    def test_count_reasoning_tokens_empty_cases(self):
        """Test edge cases with empty or invalid input"""
        test_cases = ["", None, 123, "<think></think>"]
        
        for case in test_cases:
            result1 = optillm_count(case)
            result2 = inference_count(case)
            
            self.assertGreaterEqual(result1, 0, f"Should handle {case} gracefully")
            self.assertGreaterEqual(result2, 0, f"Should handle {case} gracefully")
    
    def test_count_reasoning_tokens_with_mock_tokenizer(self):
        """Test with a simple mock tokenizer"""
        class MockTokenizer:
            def encode(self, text):
                return text.split()  # Simple word-based tokenization
        
        tokenizer = MockTokenizer()
        text = "<think>hello world test</think>answer"
        
        result = optillm_count(text, tokenizer)
        self.assertEqual(result, 3, "Should use tokenizer when provided")
    
    def test_reasoning_tokens_fallback_estimation(self):
        """Test fallback estimation when tokenizer fails"""
        class FailingTokenizer:
            def encode(self, text):
                raise Exception("Tokenizer failed")
        
        tokenizer = FailingTokenizer()
        text = "<think>some reasoning content here</think>answer"
        
        result = optillm_count(text, tokenizer)
        self.assertGreater(result, 0, "Should fallback to character estimation")


class TestInferenceStructures(unittest.TestCase):
    """Test that inference structures support reasoning tokens"""
    
    def test_chat_completion_usage_with_reasoning_tokens(self):
        """Test ChatCompletionUsage supports reasoning_tokens"""
        from optillm.inference import ChatCompletionUsage
        
        # Test with reasoning tokens
        usage = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            reasoning_tokens=5
        )
        
        self.assertEqual(usage.reasoning_tokens, 5)
        
        # Test default value
        usage_default = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        self.assertEqual(usage_default.reasoning_tokens, 0)
    
    def test_chat_completion_model_dump_structure(self):
        """Test ChatCompletion model_dump includes reasoning_tokens"""
        from optillm.inference import ChatCompletion
        
        response_dict = {
            "id": "test-123",
            "object": "chat.completion", 
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
                "reasoning_tokens": 3
            }
        }
        
        completion = ChatCompletion(response_dict)
        result = completion.model_dump()
        
        # Check structure
        self.assertIn("usage", result)
        self.assertIn("completion_tokens_details", result["usage"])
        self.assertIn("reasoning_tokens", result["usage"]["completion_tokens_details"])
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 3)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)