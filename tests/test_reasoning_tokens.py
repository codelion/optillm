#!/usr/bin/env python3
"""
Tests for reasoning token functionality in OptILLM
Covers count_reasoning_tokens function and API response format
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the count_reasoning_tokens function from both modules
from optillm import count_reasoning_tokens as optillm_count_reasoning_tokens
from optillm.inference import count_reasoning_tokens as inference_count_reasoning_tokens


class TestCountReasoningTokens(unittest.TestCase):
    """Test the count_reasoning_tokens function"""
    
    def test_count_reasoning_tokens_basic(self):
        """Test basic functionality of count_reasoning_tokens"""
        # Test with think tags
        text_with_think = "<think>This is reasoning content</think>This is output"
        
        # Test both implementations should work the same
        result1 = optillm_count_reasoning_tokens(text_with_think)
        result2 = inference_count_reasoning_tokens(text_with_think)
        
        self.assertGreater(result1, 0)
        self.assertEqual(result1, result2)
    
    def test_count_reasoning_tokens_no_think_tags(self):
        """Test with text that has no think tags"""
        text_without_think = "This is just regular output text"
        
        result1 = optillm_count_reasoning_tokens(text_without_think)
        result2 = inference_count_reasoning_tokens(text_without_think)
        
        self.assertEqual(result1, 0)
        self.assertEqual(result2, 0)
    
    def test_count_reasoning_tokens_multiple_think_blocks(self):
        """Test with multiple think tag blocks"""
        text_multiple = """
        <think>First reasoning block</think>
        Some output here
        <think>Second reasoning block with more content</think>
        Final output
        """
        
        result = optillm_count_reasoning_tokens(text_multiple)
        self.assertGreater(result, 0)
        
        # Should count tokens from both blocks
        single_block = "<think>First reasoning blockSecond reasoning block with more content</think>"
        single_result = optillm_count_reasoning_tokens(single_block)
        self.assertAlmostEqual(result, single_result, delta=2)  # Allow small variance due to formatting
    
    def test_count_reasoning_tokens_empty_input(self):
        """Test with empty or None input"""
        self.assertEqual(optillm_count_reasoning_tokens(""), 0)
        self.assertEqual(optillm_count_reasoning_tokens(None), 0)
        self.assertEqual(optillm_count_reasoning_tokens(123), 0)  # Non-string input
    
    def test_count_reasoning_tokens_malformed_tags(self):
        """Test with malformed think tags"""
        malformed_cases = [
            "<think>Unclosed think tag",
            "Unopened think tag</think>",
            "<think><think>Nested tags</think></think>",
            "<THINK>Wrong case</THINK>",
            "<think></think>",  # Empty think block
        ]
        
        for case in malformed_cases:
            result = optillm_count_reasoning_tokens(case)
            # Should handle gracefully, either 0 or some reasonable count
            self.assertGreaterEqual(result, 0)
    
    def test_count_reasoning_tokens_with_tokenizer(self):
        """Test with a mock tokenizer for precise counting"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        text = "<think>Some reasoning text</think>Output"
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        
        self.assertEqual(result, 5)
        mock_tokenizer.encode.assert_called_once_with("Some reasoning text")
    
    def test_count_reasoning_tokens_tokenizer_error(self):
        """Test fallback when tokenizer fails"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        text = "<think>Some reasoning text</think>Output"
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        
        # Should fallback to character-based estimation
        self.assertGreater(result, 0)
        mock_tokenizer.encode.assert_called_once()
    
    def test_count_reasoning_tokens_multiline(self):
        """Test with multiline think blocks"""
        multiline_text = """<think>
        This is a multi-line reasoning block
        with several lines of content
        that spans multiple lines
        </think>
        This is the final output"""
        
        result = optillm_count_reasoning_tokens(multiline_text)
        self.assertGreater(result, 10)  # Should be substantial content
    
    def test_count_reasoning_tokens_special_characters(self):
        """Test with special characters in think blocks"""
        special_text = "<think>Content with émojis 🤔 and symbols @#$%^&*()</think>Output"
        result = optillm_count_reasoning_tokens(special_text)
        self.assertGreater(result, 0)


class TestAPIResponseFormat(unittest.TestCase):
    """Test that API responses include reasoning token information"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import after setting up path
        import optillm
        self.app = optillm.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    @patch('optillm.get_config')
    def test_response_includes_completion_tokens_details(self, mock_get_config):
        """Test that API responses include completion_tokens_details"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<think>Some reasoning</think>Final answer: 42"
        mock_response.usage.completion_tokens = 20
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_config.return_value = (mock_client, "test-key")
        
        # Make request to the API
        response = self.client.post('/v1/chat/completions', 
                                  json={
                                      "model": "gpt-4o-mini",
                                      "messages": [{"role": "user", "content": "What is 2+2?"}]
                                  },
                                  headers={"Authorization": "Bearer test-key"})
        
        self.assertEqual(response.status_code, 200)
        
        # Check response format
        data = response.get_json()
        self.assertIn('usage', data)
        self.assertIn('completion_tokens_details', data['usage'])
        self.assertIn('reasoning_tokens', data['usage']['completion_tokens_details'])
        self.assertGreater(data['usage']['completion_tokens_details']['reasoning_tokens'], 0)
    
    @patch('optillm.get_config')
    def test_response_no_reasoning_tokens(self, mock_get_config):
        """Test API response when there are no reasoning tokens"""
        # Mock the OpenAI client with no think tags
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Final answer: 42"  # No think tags
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_config.return_value = (mock_client, "test-key")
        
        # Make request to the API
        response = self.client.post('/v1/chat/completions', 
                                  json={
                                      "model": "gpt-4o-mini",
                                      "messages": [{"role": "user", "content": "What is 2+2?"}]
                                  },
                                  headers={"Authorization": "Bearer test-key"})
        
        self.assertEqual(response.status_code, 200)
        
        # Check response format
        data = response.get_json()
        self.assertIn('usage', data)
        self.assertIn('completion_tokens_details', data['usage'])
        self.assertEqual(data['usage']['completion_tokens_details']['reasoning_tokens'], 0)
    
    @patch('optillm.get_config')
    def test_multiple_responses_reasoning_tokens(self, mock_get_config):
        """Test reasoning tokens with multiple responses (n > 1)"""
        # Mock the OpenAI client with multiple responses
        mock_client = Mock()
        mock_response = Mock()
        
        # Create multiple choices with different reasoning content
        choice1 = Mock()
        choice1.message.content = "<think>First reasoning</think>Answer 1"
        choice2 = Mock()
        choice2.message.content = "<think>Second longer reasoning content</think>Answer 2"
        
        mock_response.choices = [choice1, choice2]
        mock_response.usage.completion_tokens = 30
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 40
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_config.return_value = (mock_client, "test-key")
        
        # Make request with n=2
        response = self.client.post('/v1/chat/completions', 
                                  json={
                                      "model": "gpt-4o-mini",
                                      "messages": [{"role": "user", "content": "What is 2+2?"}],
                                      "n": 2
                                  },
                                  headers={"Authorization": "Bearer test-key"})
        
        self.assertEqual(response.status_code, 200)
        
        # Check response format
        data = response.get_json()
        self.assertIn('usage', data)
        self.assertIn('completion_tokens_details', data['usage'])
        self.assertGreater(data['usage']['completion_tokens_details']['reasoning_tokens'], 0)
        
        # Should have 2 choices
        self.assertEqual(len(data['choices']), 2)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing functionality"""
    
    def test_existing_approaches_still_work(self):
        """Test that existing approaches work without reasoning token changes"""
        # Import approaches that don't use reasoning
        from optillm.bon import best_of_n_sampling
        
        # Create mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Regular response"
        mock_response.usage.completion_tokens = 10
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test that approach still works
        try:
            result, tokens = best_of_n_sampling(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                client=mock_client,
                n=3
            )
            self.assertIsInstance(result, str)
            self.assertIsInstance(tokens, int)
        except Exception as e:
            self.fail(f"Existing approach failed: {e}")
    
    def test_api_without_auth_header(self):
        """Test API still returns proper errors without auth"""
        import optillm
        app = optillm.app
        app.config['TESTING'] = True
        client = app.test_client()
        
        response = client.post('/v1/chat/completions', 
                             json={"model": "test", "messages": []})
        
        # Should still return 401 for missing auth
        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)