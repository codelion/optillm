"""Test the JSON plugin for compatibility with outlines>=1.1.0"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any

# Mock the dependencies before importing the plugin
import sys
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['outlines'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

# Import after mocking
from optillm.plugins.json_plugin import JSONGenerator, extract_schema_from_response_format, run


class TestJSONPlugin(unittest.TestCase):
    """Test cases for the JSON plugin with new outlines API."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample JSON schemas for testing
        self.simple_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age"]
        })
        
        self.complex_schema = json.dumps({
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "email": {"type": "string"},
                "score": {"type": "number"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"}
            },
            "required": ["id", "email"]
        })
    
    @patch('optillm.plugins.json_plugin.outlines.from_transformers')
    @patch('optillm.plugins.json_plugin.AutoTokenizer.from_pretrained')
    def test_json_generator_init(self, mock_tokenizer, mock_from_transformers):
        """Test JSONGenerator initialization with new API."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_from_transformers.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Initialize JSONGenerator
        generator = JSONGenerator()
        
        # Verify initialization
        mock_from_transformers.assert_called_once()
        mock_tokenizer.assert_called_once()
        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.tokenizer)
    
    @patch('optillm.plugins.json_plugin.create_model')
    def test_parse_json_schema_to_pydantic(self, mock_create_model):
        """Test JSON schema to Pydantic model conversion."""
        # Mock Pydantic model creation
        mock_model_class = Mock()
        mock_create_model.return_value = mock_model_class
        
        # Create generator with mocked dependencies
        generator = JSONGenerator.__new__(JSONGenerator)
        
        # Test simple schema parsing
        result = generator.parse_json_schema_to_pydantic(self.simple_schema)
        
        # Verify create_model was called with correct fields
        mock_create_model.assert_called_once()
        call_args = mock_create_model.call_args
        self.assertEqual(call_args[0][0], 'DynamicModel')
        
        # Check fields
        fields = call_args[1]
        self.assertIn('name', fields)
        self.assertIn('age', fields)
        self.assertIn('active', fields)
    
    @patch('optillm.plugins.json_plugin.outlines.from_transformers')
    @patch('optillm.plugins.json_plugin.AutoTokenizer.from_pretrained')
    def test_generate_json_new_api(self, mock_tokenizer, mock_from_transformers):
        """Test JSON generation with new outlines API."""
        # Create mock Pydantic instance with model_dump method
        mock_result = Mock()
        mock_result.model_dump.return_value = {"name": "Test", "age": 25}
        
        # Mock the model to return our result
        mock_model = Mock()
        mock_model.return_value = mock_result
        mock_from_transformers.return_value = mock_model
        
        # Initialize generator
        generator = JSONGenerator()
        
        # Test generation
        prompt = "Create a person named Test who is 25 years old"
        result = generator.generate_json(prompt, self.simple_schema)
        
        # Verify the result
        self.assertEqual(result, {"name": "Test", "age": 25})
        mock_model.assert_called_once()
    
    def test_extract_schema_from_response_format(self):
        """Test schema extraction from OpenAI response format."""
        # Test with OpenAI format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string"}
                    }
                }
            }
        }
        
        result = extract_schema_from_response_format(response_format)
        self.assertIsNotNone(result)
        
        # Verify it's valid JSON
        schema = json.loads(result)
        self.assertEqual(schema["type"], "object")
        self.assertIn("test", schema["properties"])
    
    @patch('optillm.plugins.json_plugin.JSONGenerator')
    def test_run_function_with_schema(self, mock_json_generator_class):
        """Test the main run function with a valid schema."""
        # Mock JSONGenerator instance
        mock_generator = Mock()
        mock_generator.generate_json.return_value = {"result": "test"}
        mock_generator.count_tokens.return_value = 10
        mock_json_generator_class.return_value = mock_generator
        
        # Mock client
        mock_client = Mock()
        
        # Test configuration
        request_config = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        # Run the plugin
        result, tokens = run(
            "System prompt",
            "Generate a test result",
            mock_client,
            "test-model",
            request_config
        )
        
        # Verify results
        self.assertIn("result", result)
        self.assertEqual(tokens, 10)
        mock_generator.generate_json.assert_called_once()
    
    def test_run_function_without_schema(self):
        """Test the main run function without a schema (fallback)."""
        # Mock client and response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Regular response"))]
        mock_response.usage.completion_tokens = 5
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        # Run without schema
        result, tokens = run(
            "System prompt",
            "Test query",
            mock_client,
            "test-model",
            {}
        )
        
        # Verify fallback behavior
        self.assertEqual(result, "Regular response")
        self.assertEqual(tokens, 5)
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('optillm.plugins.json_plugin.JSONGenerator')
    def test_error_handling(self, mock_json_generator_class):
        """Test error handling and fallback."""
        # Mock generator that raises an error
        mock_generator = Mock()
        mock_generator.generate_json.side_effect = Exception("Test error")
        mock_json_generator_class.return_value = mock_generator
        
        # Mock client for fallback
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Fallback response"))]
        mock_response.usage.completion_tokens = 8
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test configuration with schema
        request_config = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object"}
                }
            }
        }
        
        # Run and expect fallback
        result, tokens = run(
            "System prompt",
            "Test query",
            mock_client,
            "test-model",
            request_config
        )
        
        # Verify fallback was used
        self.assertEqual(result, "Fallback response")
        self.assertEqual(tokens, 8)
        mock_client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()