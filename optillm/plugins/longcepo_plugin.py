"""The Long-Context Cerebras Planning and Optimization (LongCePO) Method

LongCePO is an inference-time computation method designed to provide LLMs with the capability to work with infinite context such as external knowledge bases that can run into millions of tokens. We achieve this goal through a combination of multiple strategies including planning (query decomposition) and divide-and-conquer long-context processing. This approach enables to use a limited context window (e.g. 8K) and outperform full-context processing with the same base model in many question-answering tasks.

If you have any questions or want to contribute, please reach out to us on [cerebras.ai/discord](https://cerebras.ai/discord).
"""

import os
import sys
import importlib.util
from typing import Tuple

SLUG = "longcepo"

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    # Get the directory where this plugin is located
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    longcepo_dir = os.path.join(plugin_dir, 'longcepo')
    main_file = os.path.join(longcepo_dir, 'main.py')
    
    # Load the main module dynamically
    spec = importlib.util.spec_from_file_location("longcepo_main", main_file)
    longcepo_main = importlib.util.module_from_spec(spec)
    
    # Add the longcepo directory to the Python path temporarily
    if longcepo_dir not in sys.path:
        sys.path.insert(0, longcepo_dir)
    
    try:
        spec.loader.exec_module(longcepo_main)
        return longcepo_main.run_longcepo(system_prompt, initial_query, client, model)
    finally:
        # Remove from path after use
        if longcepo_dir in sys.path:
            sys.path.remove(longcepo_dir)
