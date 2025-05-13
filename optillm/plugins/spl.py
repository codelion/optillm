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

from typing import Tuple
from optillm.plugins.spl.main import run_spl

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
                       Can include {'spl_inference_only': True} to run in inference-only mode
    
    Returns:
        Tuple[str, int]: The LLM response and token count
    """
    return run_spl(system_prompt, initial_query, client, model, request_config)
