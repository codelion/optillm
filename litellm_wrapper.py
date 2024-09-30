import os
import litellm
from litellm import completion
from typing import List, Dict, Any, Optional

class LiteLLMWrapper:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = self.Chat()
        # litellm.set_verbose=True

    class Chat:
        class Completions:
            @staticmethod
            def create(model: str, messages: List[Dict[str, str]], **kwargs):
                response = completion(model=model, messages=messages, **kwargs)
                # Convert LiteLLM response to match OpenAI response structure
                return response

        completions = Completions()

    class Models:
        @staticmethod
        def list():
            # Since LiteLLM doesn't have a direct method to list models,
            # we'll return a predefined list of supported models.
            # This list can be expanded as needed.
            return {
                "data": [
                    {"id": "gpt-3.5-turbo"},
                    {"id": "gpt-4"},
                    {"id": "command-nightly"},
                    # Add more models as needed
                ]
            }

    models = Models()
