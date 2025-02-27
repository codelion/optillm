from typing import Tuple, Dict, Any, Optional
import logging
from outlines import models, generate
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Plugin identifier
SLUG = "json"

# Setup logging
logger = logging.getLogger(__name__)

class JSONGenerator:
    def get_device(self):
        """Get the appropriate device (mps, cuda, or cpu)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """Initialize the JSON generator with a specific model."""
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        try:
            llm = AutoModelForCausalLM.from_pretrained(model_name)
            llm.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = models.Transformers(llm, tokenizer)
            self.tokenizer = tokenizer
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    def generate_json(self, prompt: str, schema: str) -> Dict[str, Any]:
        """Generate JSON based on the provided schema and prompt."""
        try:
            # Create JSON generator with the schema
            generator = generate.json(self.model, schema)
            logger.info("Created JSON generator with schema")

            # Generate JSON response
            result = generator(prompt)
            logger.info("Successfully generated JSON response")
            return result

        except Exception as e:
            logger.error(f"Error generating JSON: {str(e)}")
            raise

def extract_schema_from_response_format(response_format: Dict[str, Any]) -> Optional[str]:
    """Extract schema from response_format field."""
    try:
        if not response_format:
            return None
            
        # Check if it's the OpenAI format
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                schema_data = response_format.get("json_schema", {})
                if isinstance(schema_data, dict) and "schema" in schema_data:
                    return json.dumps(schema_data["schema"])
                return json.dumps(schema_data)
                
        logger.warning(f"Could not extract valid schema from response_format")
        return None
    except Exception as e:
        logger.error(f"Error extracting schema from response_format: {str(e)}")
        return None

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None) -> Tuple[str, int]:
    """Main plugin execution function."""
    logger.info("Starting JSON plugin execution")
    completion_tokens = 0

    try:
        # Extract schema from response_format in request_config
        response_format = request_config.get("response_format") if request_config else None
        schema = extract_schema_from_response_format(response_format)
        
        if not schema:
            logger.warning("No valid schema found in response_format")
            # Fall back to regular completion if no schema is specified
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ]
            )
            return response.choices[0].message.content, response.usage.completion_tokens

        # Initialize JSON generator
        json_generator = JSONGenerator()
        
        # Generate JSON response
        result = json_generator.generate_json(initial_query, schema)
        
        # Convert result to string if it's not already
        json_response = json.dumps(result) if isinstance(result, dict) else str(result)
        completion_tokens = json_generator.count_tokens(json_response)
        
        logger.info(f"Successfully generated JSON response: {json_response}")
        return json_response, completion_tokens

    except Exception as e:
        logger.error(f"Error in JSON plugin: {str(e)}")
        # Fall back to regular completion on error
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ]
        )
        return response.choices[0].message.content, response.usage.completion_tokens