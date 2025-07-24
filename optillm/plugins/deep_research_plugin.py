"""
Deep Research Plugin - OptILLM Plugin Interface

This plugin implements the Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
from the paper "Deep Researcher with Test-Time Diffusion".

Paper: https://arxiv.org/abs/2507.16075v1

The plugin orchestrates web search, URL fetching, and memory synthesis to provide
comprehensive research responses using an iterative refinement approach.
"""

from typing import Tuple, Dict, Optional
from optillm.plugins.deep_research import DeepResearcher

SLUG = "deep_research"


def run(system_prompt: str, initial_query: str, client, model: str, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Deep Research plugin implementing TTD-DR (Test-Time Diffusion Deep Researcher)
    
    This plugin orchestrates web search, URL fetching, and memory synthesis to provide
    comprehensive research responses using an iterative refinement approach.
    
    Based on: "Deep Researcher with Test-Time Diffusion" 
    https://arxiv.org/abs/2507.16075v1
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's research query
        client: OpenAI client for LLM calls
        model: Model name to use for synthesis
        request_config: Optional configuration dict with keys:
            - max_iterations: Maximum research iterations (default: 5)
            - max_sources: Maximum web sources per search (default: 10)
    
    Returns:
        Tuple of (comprehensive_research_response, total_completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    max_iterations = config.get("max_iterations", 5)
    max_sources = config.get("max_sources", 10)
    
    # Validate inputs
    if not initial_query.strip():
        return "Error: No research query provided", 0
    
    if not client:
        return "Error: No LLM client provided for research synthesis", 0
    
    # Initialize researcher
    researcher = DeepResearcher(
        client=client,
        model=model,
        max_iterations=max_iterations,
        max_sources=max_sources
    )
    
    try:
        # Perform deep research
        result, total_tokens = researcher.research(system_prompt, initial_query)
        return result, total_tokens
        
    except Exception as e:
        error_response = f"Deep research failed: {str(e)}\n\nFalling back to basic response..."
        
        # Fallback: provide basic response using just the model
        try:
            fallback_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ]
            )
            
            result = fallback_response.choices[0].message.content.strip()
            tokens = fallback_response.usage.completion_tokens
            
            return f"{error_response}\n\n{result}", tokens
            
        except Exception as fallback_error:
            return f"Deep research and fallback both failed: {str(e)} | {str(fallback_error)}", 0