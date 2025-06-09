"""
System Prompt Learning (SPL) Plugin for OptiLLM

This plugin implements Andrej Karpathy's proposed system prompt learning paradigm,
allowing LLMs to improve their problem-solving capabilities by:
1. Identifying problem types
2. Generating and refining strategies for solving different problems
3. Building a knowledge base of problem-solving techniques
4. Applying these techniques to new instances of similar problems
5. Tracking the success of different strategies to prioritize effective ones

The plugin maintains a database of strategies that evolves over time, making the
LLM incrementally better at solving problems by learning from its experiences.
"""

import os
import sys
import importlib.util
from typing import Tuple

# Plugin identifier
SLUG = "spl"

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None) -> Tuple[str, int]:
    """
    Plugin entry point for System Prompt Learning.
    
    Args:
        system_prompt: The system prompt
        initial_query: The user's query
        client: The LLM client
        model: The model identifier
        request_config: Optional request configuration
                       Can include {'spl_learning': True} to enable learning mode
    
    Returns:
        Tuple[str, int]: The LLM response and token count
    """
    # Get the directory where this plugin is located
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    spl_dir = os.path.join(plugin_dir, 'spl')
    main_file = os.path.join(spl_dir, 'main.py')
    
    # Load the main module dynamically
    spec = importlib.util.spec_from_file_location("spl_main", main_file)
    spl_main = importlib.util.module_from_spec(spec)
    
    # Add the spl directory to the Python path temporarily
    if spl_dir not in sys.path:
        sys.path.insert(0, spl_dir)
    
    try:
        spec.loader.exec_module(spl_main)
        return spl_main.run_spl(system_prompt, initial_query, client, model, request_config)
    finally:
        # Remove from path after use
        if spl_dir in sys.path:
            sys.path.remove(spl_dir)
