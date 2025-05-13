"""
Functions for evaluating strategies in the System Prompt Learning (SPL) plugin.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from optillm.plugins.spl.strategy import Strategy
from optillm.plugins.spl.utils import extract_thinking
from optillm.plugins.spl.prompts import (
    STRATEGY_EVALUATION_PROMPT,
    STRATEGY_REFINEMENT_PROMPT
)
from optillm.plugins.spl.config import (
    DEFAULT_MAX_TOKENS,
    MAX_STRATEGIES_FOR_INFERENCE
)

# Setup logging
logger = logging.getLogger(__name__)

def select_relevant_strategies(query: str, problem_type: str, db: Any, max_strategies: int = MAX_STRATEGIES_FOR_INFERENCE) -> List[Strategy]:
    """
    Select the most relevant strategies for a given problem to be used during inference.
    This controls how many strategies are included in the system prompt augmentation.
    
    Args:
        query: The problem/query text
        problem_type: The type of problem
        db: Strategy database
        max_strategies: Maximum number of strategies to return
    
    Returns:
        List[Strategy]: The selected strategies
    """
    # First, get strategies specifically for this problem type
    type_specific = db.get_strategies_for_problem(problem_type)
    
    # If we have more type-specific strategies than needed, sort and select the best ones
    if len(type_specific) > max_strategies:
        # Score each strategy based on success rate and recency
        scored_strategies = []
        for strategy in type_specific:
            recency_score = 0
            if strategy.last_used:
                # Calculate days since last use
                last_used = datetime.fromisoformat(strategy.last_used)
                days_since = (datetime.now() - last_used).days
                recency_score = max(0, 1.0 - min(1.0, days_since / 30.0))  # Higher for more recent
            
            # Combined score with success rate weighing more
            score = (0.7 * strategy.success_rate) + (0.3 * recency_score)
            scored_strategies.append((strategy, score))
        
        # Sort by score (descending) and take top strategies
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_strategies[:max_strategies]]
    
    # If we don't have enough type-specific strategies, get similar strategies from other types
    if len(type_specific) < max_strategies:
        # Calculate how many more strategies we need
        needed = max_strategies - len(type_specific)
        
        # Get similar strategies from other problem types
        type_specific_ids = {s.strategy_id for s in type_specific}
        similar_strategies = []
        
        for s, score in db.get_similar_strategies(query, n=max_strategies*2):  # Get more than needed to filter
            # Only include strategies from other problem types and not already selected
            if s.strategy_id not in type_specific_ids and s.problem_type != problem_type:
                similar_strategies.append(s)
                if len(similar_strategies) >= needed:
                    break
        
        # Combine type-specific strategies with similar strategies
        combined = type_specific + similar_strategies[:needed]  # Only add as many as needed
        
        # Log which strategies we're using
        for i, strategy in enumerate(combined, 1):
            problem_type_str = "(same type)" if strategy.problem_type == problem_type else f"(from {strategy.problem_type})"
            logger.info(f"Selected strategy {i}/{max_strategies} for inference: {strategy.strategy_id} {problem_type_str} (success rate: {strategy.success_rate:.2f})")
            
        return combined
    
    # If we have exactly the right number, just return them
    # Log which strategies we're using
    for i, strategy in enumerate(type_specific, 1):
        logger.info(f"Selected strategy {i}/{max_strategies} for inference: {strategy.strategy_id} (same type) (success rate: {strategy.success_rate:.2f})")
        
    return type_specific[:max_strategies]

def evaluate_strategy_effectiveness(response: str, thinking: Optional[str], selected_strategies: List[Strategy], client, model: str) -> Dict[str, bool]:
    """
    Evaluate how effective each strategy was in generating the response.
    
    Args:
        response: The LLM's final response to the query
        thinking: The LLM's reasoning process (if any)
        selected_strategies: The strategies that were used
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Dict[str, bool]: Mapping from strategy ID to effectiveness (True/False)
    """
    if not selected_strategies:
        return {}
    
    results = {}
    
    try:
        for strategy in selected_strategies:
            # Include thinking in the evaluation if available
            full_response = thinking + "\n\n" + response if thinking else response
            
            messages = [
                {
                    "role": "system", 
                    "content": STRATEGY_EVALUATION_PROMPT
                },
                {
                    "role": "user", 
                    "content": (
                        f"Strategy:\n{strategy.strategy_text}\n\n"
                        f"Response (including reasoning):\n{full_response}\n\n"
                        f"Does the response show clear evidence that the strategy was effectively applied? "
                        f"Answer with ONLY YES or NO."
                    )
                }
            ]
            
            eval_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
            )
            
            # Get the response and extract final answer (remove thinking blocks)
            result_text = eval_response.choices[0].message.content
            final_result, eval_thinking = extract_thinking(result_text)
            
            # Clean up and normalize the result
            final_result = final_result.strip().upper()
            
            logger.debug(f"Strategy evaluation - raw response: '{result_text}'")
            logger.debug(f"Strategy evaluation - final result after removing thinking: '{final_result}'")
            
            # Check for YES in the final answer (not in thinking blocks)
            is_effective = "YES" in final_result
            
            results[strategy.strategy_id] = is_effective
            logger.info(f"Strategy {strategy.strategy_id} evaluation: {final_result} -> {is_effective}")
    
    except Exception as e:
        logger.error(f"Error evaluating strategy effectiveness: {str(e)}")
        # Default to neutral results if evaluation fails
        for strategy in selected_strategies:
            results[strategy.strategy_id] = False
    
    return results

def refine_strategy(strategy: Strategy, problem: str, response: str, thinking: Optional[str], client, model: str) -> Strategy:
    """
    Refine a strategy based on its application to a specific problem.
    
    Args:
        strategy: The strategy to refine
        problem: The problem that was solved
        response: The LLM's final response to the problem
        thinking: The LLM's reasoning process (if any)
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Strategy: The refined strategy
    """
    try:
        # Include thinking in refinement if available
        full_response = thinking + "\n\n" + response if thinking else response
        
        messages = [
            {
                "role": "system", 
                "content": STRATEGY_REFINEMENT_PROMPT
            },
            {
                "role": "user", 
                "content": (
                    f"Original strategy for {strategy.problem_type} problems:\n{strategy.strategy_text}\n\n"
                    f"New problem:\n{problem}\n\n"
                    f"Solution process (including reasoning):\n{full_response}\n\n"
                    f"Provide a refined version of the original strategy that incorporates "
                    f"any insights from this new example."
                )
            }
        ]
        
        refine_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        response_text = refine_response.choices[0].message.content
        
        # Extract refined strategy and thinking
        refined_text, refinement_thinking = extract_thinking(response_text)
        if not refined_text.strip():
            refined_text = response_text  # Use full response if extraction failed
        
        logger.debug(f"Strategy refinement - raw response: '{response_text}'")
        logger.debug(f"Strategy refinement - final text after removing thinking: '{refined_text}'")
        
        # Create a copy of the strategy with the refined text
        refined_strategy = Strategy(
            strategy_id=strategy.strategy_id,
            problem_type=strategy.problem_type,
            strategy_text=refined_text.strip(),
            examples=strategy.examples + [problem],
            success_count=strategy.success_count,
            total_attempts=strategy.total_attempts,
            created_at=strategy.created_at,
            last_used=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            confidence=strategy.confidence,
            tags=strategy.tags,
            reasoning_examples=strategy.reasoning_examples.copy()
        )
        
        # Add the refinement thinking if available
        if refinement_thinking:
            refined_strategy.add_reasoning_example(refinement_thinking)
        
        return refined_strategy
    
    except Exception as e:
        logger.error(f"Error refining strategy: {str(e)}")
        # Return the original strategy if refinement fails
        return strategy
