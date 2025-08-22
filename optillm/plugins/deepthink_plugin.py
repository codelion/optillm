"""
Deep Think Plugin for OptILM

Combines SELF-DISCOVER framework with uncertainty-routed chain-of-thought
for enhanced reasoning in large language models.
"""

import logging
from typing import Tuple, Dict, Any
from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT

# Plugin identifier for optillm
SLUG = "deepthink"

logger = logging.getLogger(__name__)

def run(
    system_prompt: str, 
    initial_query: str, 
    client, 
    model: str, 
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    """
    Main entry point for the Deep Think plugin.
    
    Combines SELF-DISCOVER reasoning structure discovery with 
    uncertainty-routed chain-of-thought generation.
    
    Args:
        system_prompt: System prompt for the model
        initial_query: User's initial query/problem
        client: OpenAI-compatible client instance
        model: Model identifier
        request_config: Additional configuration parameters
        
    Returns:
        Tuple of (response_text, completion_tokens_used)
    """
    logger.info("Starting Deep Think reasoning process")
    
    # Extract configuration parameters
    config = _parse_config(request_config or {})
    
    # Initialize components
    self_discover = SelfDiscover(
        client=client,
        model=model,
        max_tokens=config["max_tokens"]
    )
    
    uncertainty_cot = UncertaintyRoutedCoT(
        client=client,
        model=model,
        max_tokens=config["max_tokens"]
    )
    
    total_tokens = 0
    
    # Stage 1: SELF-DISCOVER reasoning structure (if enabled)
    reasoning_structure = None
    if config["enable_self_discover"]:
        logger.info("Discovering task-specific reasoning structure")
        
        discovery_result = self_discover.discover_reasoning_structure(
            task_description=_extract_task_description(initial_query, system_prompt),
            task_examples=None  # Could be enhanced to extract examples
        )
        
        reasoning_structure = discovery_result["reasoning_structure"]
        total_tokens += discovery_result["completion_tokens"]
        
        logger.info(f"Discovered reasoning structure with {len(reasoning_structure)} components")
    
    # Prepare enhanced prompt
    enhanced_prompt = _create_enhanced_prompt(
        system_prompt=system_prompt,
        initial_query=initial_query,
        reasoning_structure=reasoning_structure,
        config=config
    )
    
    # Stage 2: Uncertainty-routed generation
    logger.info("Generating response with uncertainty routing")
    
    generation_result = uncertainty_cot.generate_with_uncertainty_routing(
        prompt=enhanced_prompt,
        num_samples=config["deepthink_samples"],
        confidence_threshold=config["confidence_threshold"],
        temperature=config["temperature"],
        top_p=config["top_p"]
    )
    
    total_tokens += generation_result["completion_tokens"]
    
    # Log routing decision
    logger.info(f"Routing decision: {generation_result['routing_decision']} "
               f"(confidence: {generation_result['confidence_score']:.3f})")
    
    final_response = generation_result["final_response"]
    
    # Clean up the response if needed
    final_response = _clean_response(final_response)
    
    logger.info(f"Deep Think completed successfully. Total tokens: {total_tokens}")
    
    return final_response, total_tokens

def _parse_config(request_config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate configuration parameters."""
    
    default_config = {
        "deepthink_samples": 3,
        "confidence_threshold": 0.7,
        "max_tokens": 16382,
        "temperature": 0.7,
        "top_p": 0.95,
        "enable_self_discover": True,
        "reasoning_modules_limit": 7
    }
    
    # Override with request config values
    for key, value in request_config.items():
        if key in default_config:
            default_config[key] = value
    
    # Validate ranges
    default_config["deepthink_samples"] = max(1, min(10, default_config["deepthink_samples"]))
    default_config["confidence_threshold"] = max(0.0, min(1.0, default_config["confidence_threshold"]))
    default_config["temperature"] = max(0.0, min(2.0, default_config["temperature"]))
    default_config["top_p"] = max(0.0, min(1.0, default_config["top_p"]))
    default_config["reasoning_modules_limit"] = max(3, min(15, default_config["reasoning_modules_limit"]))
    
    return default_config

def _extract_task_description(initial_query: str, system_prompt: str) -> str:
    """Extract a task description for SELF-DISCOVER from the query and system prompt."""
    
    # Combine system prompt and query to understand the task
    combined_text = f"{system_prompt}\n\n{initial_query}"
    
    # Try to identify the type of task based on keywords and patterns
    task_keywords = {
        "mathematical": ["solve", "calculate", "equation", "math", "number", "formula"],
        "analytical": ["analyze", "evaluate", "assess", "examine", "compare"],
        "creative": ["create", "design", "generate", "brainstorm", "invent"],
        "logical": ["reason", "logic", "prove", "deduce", "conclude"],
        "planning": ["plan", "strategy", "approach", "method", "steps"],
        "problem_solving": ["problem", "solution", "solve", "fix", "resolve"]
    }
    
    detected_types = []
    combined_lower = combined_text.lower()
    
    for task_type, keywords in task_keywords.items():
        if any(keyword in combined_lower for keyword in keywords):
            detected_types.append(task_type)
    
    if detected_types:
        primary_type = detected_types[0]
        task_description = f"This is primarily a {primary_type} task that requires {', '.join(detected_types)} thinking."
    else:
        task_description = "This is a general reasoning task that requires systematic thinking and analysis."
    
    # Add context from the query
    if len(initial_query) > 50:
        task_description += f" The specific task involves: {initial_query[:200]}..."
    else:
        task_description += f" The specific task is: {initial_query}"
    
    return task_description

def _create_enhanced_prompt(
    system_prompt: str,
    initial_query: str, 
    reasoning_structure: Dict[str, Any] = None,
    config: Dict[str, Any] = None
) -> str:
    """Create an enhanced prompt that incorporates the reasoning structure."""
    
    base_prompt = f"""System: {system_prompt}

Task: {initial_query}"""
    
    if reasoning_structure:
        import json
        structure_text = json.dumps(reasoning_structure, indent=2)
        
        enhanced_prompt = f"""{base_prompt}

REASONING STRUCTURE:
Please follow this discovered reasoning structure to solve the problem systematically:

{structure_text}

INSTRUCTIONS:
1. Use the reasoning structure above to guide your thinking process
2. Work through each component of the structure systematically  
3. Wrap your detailed reasoning process in <think> tags
4. After your reasoning, provide a clear and concise final answer
5. Be thorough in your analysis but also aim for clarity and accuracy

<think>
[Follow the reasoning structure step-by-step to analyze and solve the problem]
</think>

Based on my systematic analysis, the answer is:"""
    else:
        enhanced_prompt = f"""{base_prompt}

INSTRUCTIONS:
Please solve this problem using careful step-by-step reasoning.

1. Wrap your detailed reasoning process in <think> tags
2. Consider the problem from multiple angles
3. Work through the solution systematically
4. Provide a clear and well-supported final answer

<think>
[Provide your detailed step-by-step reasoning here]
</think>

Based on my analysis, the answer is:"""
    
    return enhanced_prompt

def _clean_response(response: str) -> str:
    """Clean up the final response."""
    
    # Remove any trailing whitespace
    response = response.strip()
    
    # Ensure the response doesn't end abruptly
    if response and not response.endswith(('.', '!', '?', ':', '"', "'")):
        # Don't add punctuation if it's a number or simple phrase
        if not (response.replace(' ', '').replace(',', '').replace('.', '').isdigit() or len(response.split()) <= 3):
            response += "."
    
    return response
