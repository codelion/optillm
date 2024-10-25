import os
import litellm
from litellm import completion
from typing import List, Dict, Any, Optional

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
        self.chat = self.Chat()
        # litellm.set_verbose=True

    class Chat:
        class Completions:
            @staticmethod
            def create(model: str, messages: List[Dict[str, str]], **kwargs):
                if model.startswith("gemini"):
                    response = completion(model=model, messages=messages, **kwargs, safety_settings=SAFETY_SETTINGS)
                else:
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
                    {"id": "gpt-4o-mini"},
                    {"id": "gpt-4o"},
                    {"id": "command-nightly"},
                    # Add more models as needed
                ]
            }

    models = Models()
