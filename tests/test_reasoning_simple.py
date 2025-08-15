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
    
    def test_count_reasoning_tokens_truncated_response(self):
        """Test counting tokens when response is truncated (no closing </think> tag)"""
        # Test truncated think tag
        truncated_text = "<think>This reasoning was cut off due to max tokens"
        
        result1 = optillm_count(truncated_text)
        result2 = inference_count(truncated_text)
        
        self.assertGreater(result1, 0, "Should count tokens from truncated think block")
        self.assertEqual(result1, result2, "Both functions should return same result")
    
    def test_count_reasoning_tokens_mixed_complete_and_truncated(self):
        """Test with both complete and truncated think blocks"""
        mixed_text = """
        <think>First complete reasoning block</think>
        Some output here
        <think>This second block was truncated and never closed
        """
        
        result = optillm_count(mixed_text)
        self.assertGreater(result, 0, "Should count tokens from both complete and truncated blocks")
        
        # Should be more than just the first block alone
        first_block_only = "<think>First complete reasoning block</think>"
        first_result = optillm_count(first_block_only)
        self.assertGreater(result, first_result, "Should include truncated content")
    
    def test_count_reasoning_tokens_no_false_positives(self):
        """Test that we don't count think-like content that isn't actually truncated"""
        # This should NOT be counted as truncated since there's a </think> later
        text_with_complete_blocks = "<think>First block</think>Output<think>Second complete block</think>"
        
        result = optillm_count(text_with_complete_blocks)
        
        # Count manually - should only be the content inside the two complete blocks
        manual_count = optillm_count("<think>First blockSecond complete block</think>")
        self.assertEqual(result, manual_count, "Should only count complete blocks, not detect false truncation")
    
    def test_count_reasoning_tokens_edge_cases_truncated(self):
        """Test edge cases with truncated responses"""
        test_cases = [
            ("<think>", 0),  # Just opening tag, no content
            ("<think>a", 1),  # Minimal content
            ("Some output <think>reasoning here", None),  # Truncated at end
            ("<think>multi\nline\ntruncated", None),  # Multiline truncated
        ]
        
        for text, expected_min in test_cases:
            result = optillm_count(text)
            if expected_min is not None:
                if expected_min == 0:
                    self.assertEqual(result, expected_min, f"Should return {expected_min} for: {text}")
                else:
                    self.assertGreaterEqual(result, expected_min, f"Should be at least {expected_min} for: {text}")
            else:
                self.assertGreater(result, 0, f"Should count truncated content for: {text}")


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