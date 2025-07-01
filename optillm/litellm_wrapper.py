import os
import time
import litellm
from litellm import completion
from litellm.utils import get_valid_models
from typing import List, Dict, Any, Optional

# Configure litellm to drop unsupported parameters
litellm.drop_params = True

SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
]

class LiteLLMWrapper:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = self.Chat(self.api_key, self.base_url)
        # litellm.set_verbose=True

    class Chat:
        def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
            self.api_key = api_key
            self.base_url = base_url
            self.completions = self.Completions(self.api_key, self.base_url)

        class Completions:
            def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
                self.api_key = api_key
                self.base_url = base_url
                
            def create(self, model: str, messages: List[Dict[str, str]], **kwargs):
                if model.startswith("gemini"):
                    response = completion(model=model, messages=messages, api_key=self.api_key, base_url=self.base_url, **kwargs, safety_settings=SAFETY_SETTINGS)
                else:
                    response = completion(model=model, messages=messages, api_key=self.api_key, base_url=self.base_url, **kwargs)
                # Convert LiteLLM response to match OpenAI response structure
                return response

    class Models:
        @staticmethod
        def list():
            try:
                # Get all valid models from LiteLLM
                valid_models = get_valid_models()
                
                # Format the response to match OpenAI's API format
                model_list = []
                for model in valid_models:
                    model_list.append({
                        "id": model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "litellm"
                    })
                
                return {
                    "object": "list",
                    "data": model_list
                }
            except Exception as e:
                # Fallback to a basic list if there's an error
                print(f"Error fetching LiteLLM models: {str(e)}")
                return {
                    "object": "list",
                    "data": [
                        {"id": "gpt-4o-mini", "object": "model", "created": int(time.time()), "owned_by": "litellm"},
                        {"id": "gpt-4o", "object": "model", "created": int(time.time()), "owned_by": "litellm"},
                        {"id": "command-nightly", "object": "model", "created": int(time.time()), "owned_by": "litellm"},
                        {"id": "claude-3-opus-20240229", "object": "model", "created": int(time.time()), "owned_by": "litellm"},
                        {"id": "claude-3-sonnet-20240229", "object": "model", "created": int(time.time()), "owned_by": "litellm"},
                        {"id": "gemini-1.5-pro-latest", "object": "model", "created": int(time.time()), "owned_by": "litellm"}
                    ]
                }
    models = Models()
