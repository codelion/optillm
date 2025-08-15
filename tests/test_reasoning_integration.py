#!/usr/bin/env python3
"""
Integration tests for reasoning token functionality with OptILLM API
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Add parent directory to path to import optillm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import app, count_reasoning_tokens


class MockOpenAIClient:
    """Enhanced mock client that can generate responses with think tags"""
    
    def __init__(self, include_thinking=True):
        self.include_thinking = include_thinking
        self.chat = self.Chat(include_thinking)
        
    class Chat:
        def __init__(self, include_thinking):
            self.completions = self.Completions(include_thinking)
            self.include_thinking = include_thinking
        
        class Completions:
            def __init__(self, include_thinking):
                self.include_thinking = include_thinking
                
            def create(self, **kwargs):
                messages = kwargs.get('messages', [])
                n = kwargs.get('n', 1)
                
                # Generate response based on the query content
                if self.include_thinking and any('think' in str(msg).lower() for msg in messages):
                    # Generate response with thinking
                    content = "<think>Let me work through this step by step. First, I need to understand what's being asked. This requires careful analysis.</think>\n\nBased on my analysis, the answer is 42."
                else:
                    # Simple response without thinking
                    content = "The answer is 42."
                
                class MockChoice:
                    def __init__(self, content, index=0):
                        self.message = type('Message', (), {'content': content})()
                        self.index = index
                        self.finish_reason = 'stop'
                
                class MockUsage:
                    completion_tokens = 50
                    total_tokens = 75
                    prompt_tokens = 25
                    
                class MockResponse:
                    def __init__(self, choices, usage):
                        self.choices = choices
                        self.usage = usage
                        
                    def model_dump(self):
                        return {
                            'choices': [
                                {
                                    'index': choice.index,
                                    'message': {'content': choice.message.content},
                                    'finish_reason': choice.finish_reason
                                } for choice in self.choices
                            ],
                            'usage': {
                                'completion_tokens': self.usage.completion_tokens,
                                'total_tokens': self.usage.total_tokens,
                                'prompt_tokens': self.usage.prompt_tokens
                            }
                        }
                
                # Create multiple choices if n > 1
                choices = []
                for i in range(n):
                    if self.include_thinking:
                        varied_content = f"<think>Thinking process {i+1}: Let me analyze this carefully...</think>\n\nAnswer {i+1}: The result is {42 + i}."
                    else:
                        varied_content = f"Answer {i+1}: The result is {42 + i}."
                    choices.append(MockChoice(varied_content, i))
                
                return MockResponse(choices, MockUsage())


class TestReasoningTokensAPIIntegration:
    """Test reasoning tokens in API responses"""
    
    def setup_method(self):
        """Setup test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Mock the get_config function to return our mock client
        self.mock_client = MockOpenAIClient(include_thinking=True)
    
    @patch('optillm.get_config')
    def test_api_response_includes_reasoning_tokens(self, mock_get_config):
        """Test that API responses include reasoning_tokens in completion_tokens_details"""
        mock_get_config.return_value = (self.mock_client, "test-key")
        
        # Test request with none approach (direct proxy)
        response = self.client.post('/v1/chat/completions', 
                                  json={
                                      'model': 'none-gpt-4o-mini',
                                      'messages': [
                                          {'role': 'user', 'content': 'Please think about this problem step by step.'}
                                      ]
                                  },
                                  headers={'Authorization': 'Bearer test-key'})
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Check response structure
        assert 'usage' in data
        assert 'completion_tokens_details' in data['usage']
        assert 'reasoning_tokens' in data['usage']['completion_tokens_details']
        
        # Should have reasoning tokens since mock returns thinking content
        reasoning_tokens = data['usage']['completion_tokens_details']['reasoning_tokens']
        assert reasoning_tokens > 0
    
    @patch('optillm.get_config')
    def test_api_response_no_reasoning_tokens(self, mock_get_config):
        """Test that responses without think tags have 0 reasoning tokens"""
        mock_client_no_thinking = MockOpenAIClient(include_thinking=False)
        mock_get_config.return_value = (mock_client_no_thinking, "test-key")
        
        response = self.client.post('/v1/chat/completions',
                                  json={
                                      'model': 'none-gpt-4o-mini',
                                      'messages': [
                                          {'role': 'user', 'content': 'What is 2+2?'}
                                      ]
                                  },
                                  headers={'Authorization': 'Bearer test-key'})
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should have 0 reasoning tokens
        reasoning_tokens = data['usage']['completion_tokens_details']['reasoning_tokens']
        assert reasoning_tokens == 0
    
    @patch('optillm.get_config')
    def test_multiple_responses_reasoning_tokens(self, mock_get_config):
        """Test reasoning tokens calculation with n > 1"""
        mock_get_config.return_value = (self.mock_client, "test-key")
        
        response = self.client.post('/v1/chat/completions',
                                  json={
                                      'model': 'none-gpt-4o-mini',
                                      'messages': [
                                          {'role': 'user', 'content': 'Think through this problem.'}
                                      ],
                                      'n': 3
                                  },
                                  headers={'Authorization': 'Bearer test-key'})
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should have 3 choices
        assert len(data['choices']) == 3
        
        # Should sum reasoning tokens from all responses
        reasoning_tokens = data['usage']['completion_tokens_details']['reasoning_tokens']
        assert reasoning_tokens > 0
        
        # Each response should have thinking content, so total should be > individual
        # (This is a rough check since we're mocking)
        assert reasoning_tokens >= 10  # Reasonable minimum
    
    def test_reasoning_tokens_calculation_accuracy(self):
        """Test that reasoning token calculation is accurate"""
        # Test direct function with known content
        test_content = "<think>This is a test thinking block with exactly twenty words to verify the token counting accuracy works properly.</think>Result: 42"
        
        expected_thinking = "This is a test thinking block with exactly twenty words to verify the token counting accuracy works properly."
        tokens = count_reasoning_tokens(test_content)
        
        # With fallback estimation (4 chars per token)
        expected_tokens = len(expected_thinking) // 4
        assert tokens == expected_tokens
    
    @patch('optillm.get_config')
    def test_error_handling_invalid_response(self, mock_get_config):
        """Test error handling when response processing fails"""
        # Mock client that returns malformed response
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_config.return_value = (mock_client, "test-key")
        
        response = self.client.post('/v1/chat/completions',
                                  json={
                                      'model': 'none-gpt-4o-mini',
                                      'messages': [{'role': 'user', 'content': 'test'}]
                                  },
                                  headers={'Authorization': 'Bearer test-key'})
        
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data


class TestApproachIntegration:
    """Test reasoning tokens with different OptILLM approaches"""
    
    def setup_method(self):
        """Setup test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_reasoning_tokens_with_mock_approach(self):
        """Test reasoning tokens with a mock approach that generates thinking"""
        
        # Create a simple test that doesn't require external API calls
        test_text_with_thinking = """
        <think>
        I need to analyze this problem step by step:
        1. First, understand the requirements
        2. Then, consider the constraints  
        3. Finally, provide a solution
        
        This seems straightforward but requires careful thought.
        </think>
        
        Based on my analysis, the answer is: 42
        """
        
        # Test the reasoning token extraction directly
        reasoning_tokens = count_reasoning_tokens(test_text_with_thinking)
        assert reasoning_tokens > 0
        
        # The thinking content should be properly extracted
        thinking_content = """
        I need to analyze this problem step by step:
        1. First, understand the requirements
        2. Then, consider the constraints  
        3. Finally, provide a solution
        
        This seems straightforward but requires careful thought.
        """
        
        # Rough token estimate (fallback method)
        expected_tokens = len(thinking_content.strip()) // 4
        assert abs(reasoning_tokens - expected_tokens) <= 5  # Allow small variance
    
    def test_complex_thinking_patterns(self):
        """Test various thinking patterns that approaches might generate"""
        
        test_cases = [
            # Single block
            "<think>Simple thinking</think>Answer: Yes",
            
            # Multiple blocks  
            "<think>First thought</think>Intermediate result<think>Second thought</think>Final answer",
            
            # Nested structure (should extract outer)
            "<think>Outer<think>inner</think>more outer</think>Result",
            
            # With code blocks inside thinking
            "<think>Let me write some code:\n```python\nx = 1\n```\nThat should work.</think>Code solution provided",
            
            # With mathematical notation
            "<think>If x = 2, then x² = 4, so the equation becomes 4 + 3 = 7</think>The result is 7"
        ]
        
        for i, test_case in enumerate(test_cases):
            tokens = count_reasoning_tokens(test_case)
            assert tokens > 0, f"Test case {i+1} should have reasoning tokens: {test_case}"
    
    def test_backward_compatibility(self):
        """Test that non-thinking responses work normally"""
        normal_responses = [
            "This is a normal response without any thinking.",
            "The answer is 42.",
            "I can help you with that. Here's the solution: x = 5",
            "",  # Empty response
        ]
        
        for response in normal_responses:
            tokens = count_reasoning_tokens(response)
            assert tokens == 0, f"Normal response should have 0 reasoning tokens: {response}"


class TestStreamingIntegration:
    """Test reasoning tokens with streaming responses"""
    
    def setup_method(self):
        """Setup test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    @patch('optillm.get_config')
    def test_streaming_response_format(self, mock_get_config):
        """Test that streaming responses don't break with reasoning tokens"""
        mock_client = MockOpenAIClient(include_thinking=True)
        mock_get_config.return_value = (mock_client, "test-key")
        
        # Note: Streaming responses in OptILLM don't include reasoning token details
        # in the same way as non-streaming, but we test that it doesn't break
        response = self.client.post('/v1/chat/completions',
                                  json={
                                      'model': 'none-gpt-4o-mini',
                                      'messages': [
                                          {'role': 'user', 'content': 'Think about this'}
                                      ],
                                      'stream': True
                                  },
                                  headers={'Authorization': 'Bearer test-key'})
        
        # Streaming should work without errors
        assert response.status_code == 200
        assert response.content_type == 'text/event-stream; charset=utf-8'


if __name__ == "__main__":
    # Run tests if pytest not available
    import traceback
    
    test_classes = [
        TestReasoningTokensAPIIntegration,
        TestApproachIntegration, 
        TestStreamingIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        instance = test_class()
        instance.setup_method()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"Running {method_name}...", end=' ')
                    getattr(instance, method_name)()
                    print("✅ PASSED")
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    traceback.print_exc()
    
    print("\n=== Integration Tests Complete ===")