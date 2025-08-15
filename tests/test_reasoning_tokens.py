#!/usr/bin/env python3
"""
Comprehensive tests for reasoning token functionality in OptILLM
"""

import pytest
import sys
import os
import re
from unittest.mock import Mock, patch

# Add parent directory to path to import optillm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import count_reasoning_tokens
from optillm.inference import count_reasoning_tokens as inference_count_reasoning_tokens


class TestCountReasoningTokensFunction:
    """Test the count_reasoning_tokens function with various inputs"""
    
    def test_empty_or_none_input(self):
        """Test handling of empty or None inputs"""
        assert count_reasoning_tokens(None) == 0
        assert count_reasoning_tokens("") == 0
        assert count_reasoning_tokens("   ") == 0
        assert count_reasoning_tokens(123) == 0  # Non-string input
        assert count_reasoning_tokens([]) == 0  # Non-string input
    
    def test_no_think_tags(self):
        """Test text without any think tags"""
        text = "This is a normal response without any thinking tags."
        assert count_reasoning_tokens(text) == 0
        
        text_with_similar = "I think this is good, but <thonk>not quite</thonk>"
        assert count_reasoning_tokens(text_with_similar) == 0
    
    def test_single_think_block(self):
        """Test text with a single think block"""
        text = "Here is my answer: <think>Let me work this out step by step. First, I need to consider...</think> The result is 42."
        tokens = count_reasoning_tokens(text)
        assert tokens > 0
        # Should count roughly the content inside think tags
        thinking_content = "Let me work this out step by step. First, I need to consider..."
        expected_rough = len(thinking_content) // 4  # Rough estimation
        assert tokens >= expected_rough - 5  # Allow some variance
    
    def test_multiple_think_blocks(self):
        """Test text with multiple think blocks"""
        text = """
        <think>First, let me analyze the problem. This seems complex.</think>
        
        The initial answer is A, but let me reconsider.
        
        <think>Actually, wait. I need to think about this differently. Maybe B is correct?</think>
        
        My final answer is B.
        """
        tokens = count_reasoning_tokens(text)
        assert tokens > 0
        
        # Should count content from both blocks
        content1 = "First, let me analyze the problem. This seems complex."
        content2 = "Actually, wait. I need to think about this differently. Maybe B is correct?"
        combined_content = content1 + content2
        expected_rough = len(combined_content) // 4
        assert tokens >= expected_rough - 10  # Allow variance for combined content
    
    def test_multiline_think_block(self):
        """Test think blocks that span multiple lines"""
        text = """<think>
        This is a multi-line thinking process.
        
        Step 1: Analyze the problem
        Step 2: Consider alternatives
        Step 3: Make a decision
        
        I need to be very careful here.
        </think>"""
        tokens = count_reasoning_tokens(text)
        assert tokens > 0
        # Should handle newlines and whitespace properly
    
    def test_malformed_think_tags(self):
        """Test handling of malformed think tags"""
        # Unclosed tag
        text1 = "Let me think: <think>This is unclosed thinking..."
        assert count_reasoning_tokens(text1) == 0
        
        # Unopened tag
        text2 = "Some thinking content here</think> and regular text."
        assert count_reasoning_tokens(text2) == 0
        
        # Nested tags - should extract outer content
        text3 = "<think>Outer thinking <think>inner</think> more outer</think>"
        tokens = count_reasoning_tokens(text3)
        assert tokens > 0  # Should extract the outer content including "inner" text
    
    def test_think_tags_with_attributes(self):
        """Test think tags with XML attributes (should not match)"""
        text = '<think id="1">This should not be counted</think>'
        # Our regex looks for exact <think> tags, not ones with attributes
        assert count_reasoning_tokens(text) == 0
    
    def test_case_sensitivity(self):
        """Test that think tags are case sensitive"""
        text1 = "<Think>This should not match</Think>"
        text2 = "<THINK>This should not match</THINK>"
        assert count_reasoning_tokens(text1) == 0
        assert count_reasoning_tokens(text2) == 0
    
    def test_with_tokenizer_mock(self):
        """Test using a mock tokenizer for precise counting"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = ['token1', 'token2', 'token3', 'token4', 'token5']
        
        text = "<think>Test content</think>"
        tokens = count_reasoning_tokens(text, tokenizer=mock_tokenizer)
        
        # Should use tokenizer when available
        assert tokens == 5
        mock_tokenizer.encode.assert_called_once_with("Test content")
    
    def test_tokenizer_error_fallback(self):
        """Test fallback when tokenizer fails"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        text = "<think>Test content for fallback</think>"
        tokens = count_reasoning_tokens(text, tokenizer=mock_tokenizer)
        
        # Should fall back to character-based estimation
        content = "Test content for fallback"
        expected = len(content) // 4
        assert tokens == expected
    
    def test_large_content_performance(self):
        """Test performance with large thinking content"""
        # Generate large thinking content
        large_content = "This is a long thinking process. " * 1000
        text = f"<think>{large_content}</think>"
        
        import time
        start = time.time()
        tokens = count_reasoning_tokens(text)
        end = time.time()
        
        assert tokens > 0
        assert end - start < 1.0  # Should complete within 1 second
    
    def test_special_characters_and_unicode(self):
        """Test handling of special characters and unicode"""
        text = "<think>Let's think about emojis 🤔 and special chars: @#$%^&*()</think>"
        tokens = count_reasoning_tokens(text)
        assert tokens > 0
        
        # Test unicode
        text_unicode = "<think>数学问题需要仔细思考</think>"
        tokens_unicode = count_reasoning_tokens(text_unicode)
        assert tokens_unicode > 0
    
    def test_inference_module_consistency(self):
        """Test that both implementations (optillm and inference) give same results"""
        test_cases = [
            "",
            "No thinking here",
            "<think>Simple thinking</think>",
            "<think>First thought</think> and <think>second thought</think>",
            "<think>Multi-line\nthinking\nprocess</think>"
        ]
        
        for text in test_cases:
            tokens1 = count_reasoning_tokens(text)
            tokens2 = inference_count_reasoning_tokens(text)
            assert tokens1 == tokens2, f"Inconsistent results for: {text}"


class TestReasoningTokensEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extremely_long_single_line(self):
        """Test with extremely long single line of thinking"""
        long_line = "A" * 10000
        text = f"<think>{long_line}</think>"
        tokens = count_reasoning_tokens(text)
        expected = len(long_line) // 4
        assert tokens == expected
    
    def test_many_small_think_blocks(self):
        """Test with many small think blocks"""
        blocks = ["<think>Short</think>"] * 100
        text = " ".join(blocks)
        tokens = count_reasoning_tokens(text)
        # Should count all blocks
        expected = (len("Short") * 100) // 4
        assert tokens == expected
    
    def test_mixed_content_structure(self):
        """Test complex mixed content"""
        text = """
        This is the introduction.
        
        <think>
        I need to solve this step by step:
        1. Parse the problem
        2. Apply the formula
        3. Check the result
        </think>
        
        Here's my first attempt: x = 5
        
        <think>
        Wait, that doesn't look right. Let me recalculate:
        - Original equation: 2x + 3 = 13  
        - Subtract 3: 2x = 10
        - Divide by 2: x = 5
        
        Actually, that is correct.
        </think>
        
        Therefore, the answer is x = 5.
        """
        tokens = count_reasoning_tokens(text)
        assert tokens > 0
        
        # Verify it extracts both thinking blocks
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, text, re.DOTALL)
        assert len(matches) == 2
    
    def test_boundary_whitespace_handling(self):
        """Test whitespace at boundaries of think tags"""
        text1 = "<think>  content with spaces  </think>"
        text2 = "<think>content without spaces</think>"
        text3 = "<think>\n  content with newlines  \n</think>"
        
        tokens1 = count_reasoning_tokens(text1)
        tokens2 = count_reasoning_tokens(text2)
        tokens3 = count_reasoning_tokens(text3)
        
        # All should return positive token counts
        assert tokens1 > 0
        assert tokens2 > 0
        assert tokens3 > 0


if __name__ == "__main__":
    # Run tests if pytest not available
    import traceback
    
    test_classes = [TestCountReasoningTokensFunction, TestReasoningTokensEdgeCases]
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"Running {method_name}...", end=' ')
                    getattr(instance, method_name)()
                    print("✅ PASSED")
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    traceback.print_exc()
    
    print("\n=== Test Summary Complete ===")