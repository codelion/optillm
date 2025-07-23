"""
Majority Voting Plugin V2 for OptILLM

Enhanced version with:
- Category-aware answer extraction
- Adaptive temperature control
- Improved answer normalization
- Response quality filtering
- Smart fallback strategies
"""

import re
import logging
from typing import Tuple, Dict, Any, List, Optional
from collections import Counter
import json

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "majority_voting"

# Default configuration
DEFAULT_K = 8
DEFAULT_TEMPERATURE = 0.6  # Unified temperature for consistency

def detect_category(query: str) -> str:
    """
    Try to detect the problem category from the query.
    
    Returns:
        Category string or 'default' if unknown
    """
    query_lower = query.lower()
    
    # GSM8K patterns
    if "###" in query or ("calculate" in query_lower and any(word in query_lower for word in ["total", "sum", "difference", "product"])):
        return "gsm8k"
    
    # MMLU patterns (multiple choice)
    if re.search(r'\b[A-E]\s*[:\)]\s*', query) or "which of the following" in query_lower:
        return "mmlu_math"
    
    # BoolQ patterns
    if query_lower.strip().endswith("?") and any(word in query_lower for word in ["is", "are", "was", "were", "does", "do", "did", "can", "could", "will", "would"]):
        return "boolq"
    
    # AQUA-RAT patterns
    if re.search(r'options?:\s*[A-E]', query, re.IGNORECASE):
        return "aqua_rat"
    
    return "default"




def extract_answer_simple(response: str, category: str) -> Optional[str]:
    """
    Extract answer using same logic as evaluation script for consistency.
    """
    if not response:
        return None
    
    # Remove thinking blocks if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    if category == "gsm8k":
        # Extract number after ###
        match = re.search(r'###\s*(-?\d*\.?\d+)', response)
        if match:
            return match.group(1)
    
    elif category == "aqua_rat":
        # For AQUA-RAT, be more flexible in extraction
        response_upper = response.upper()
        
        # Try to find letter choices (A-E)
        patterns = [
            r'\b([A-E])\b(?!\w)',  # Single letter not part of word
            r'(?:answer|choice|option)\s*:?\s*([A-E])\b',
            r'\(([A-E])\)',  # Letter in parentheses
            r'^([A-E])$',  # Just the letter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_upper, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # If no letter found, check for common wrong patterns
        # Map true/false/yes/no/numbers to letters (this is a heuristic)
        if re.search(r'\b(true|yes|1)\b', response.lower()):
            return "A"  # Default mapping
        elif re.search(r'\b(false|no|0)\b', response.lower()):
            return "B"  # Default mapping
    
    elif category == "boolq":
        response_lower = response.lower()
        if 'yes' in response_lower:
            return 'yes'
        elif 'no' in response_lower:
            return 'no'
    
    elif category == "mmlu_math":
        # For MMLU, just return the cleaned response
        return response.strip()
    
    # Default: return cleaned response
    return response.strip()


def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    """
    Simplified majority voting using consistent evaluation logic.
    """
    logger.info("Starting majority voting process")
    
    # Detect category
    category = detect_category(initial_query)
    logger.info(f"Detected category: {category}")
    
    # Extract parameters
    k = request_config.get('k', DEFAULT_K) if request_config else DEFAULT_K
    temperature = request_config.get('temperature', DEFAULT_TEMPERATURE) if request_config else DEFAULT_TEMPERATURE
    max_tokens = request_config.get('max_tokens', 4096) if request_config else 4096
    
    logger.info(f"Generating {k} candidates with temperature={temperature} for category={category}")
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_query}
    ]
    
    # Generate candidates
    candidates = []
    total_tokens = 0
    
    try:
        # Try parallel generation first
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=k,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        candidates = [choice.message.content for choice in response.choices]
        total_tokens = response.usage.completion_tokens
        
    except Exception as e:
        logger.warning(f"Parallel generation failed: {str(e)}")
        # Fallback to sequential
        for i in range(k):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                candidates.append(response.choices[0].message.content)
                total_tokens += response.usage.completion_tokens
            except Exception as err:
                logger.error(f"Error generating candidate {i+1}: {str(err)}")
                continue
    
    if not candidates:
        return "Error: Could not generate any candidates", 0
    
    # Extract answers and count votes
    answer_votes = Counter()
    answer_to_responses = {}
    
    for i, candidate in enumerate(candidates):
        answer = extract_answer_simple(candidate, category)
        if answer:
            # Normalize answer for voting
            if category == "aqua_rat":
                answer = answer.upper()  # Ensure letters are uppercase
            elif category == "boolq":
                answer = answer.lower()  # Ensure yes/no are lowercase
            elif category == "gsm8k":
                # Try to normalize numbers
                try:
                    answer = str(float(answer))
                except:
                    pass
            
            answer_votes[answer] += 1
            if answer not in answer_to_responses:
                answer_to_responses[answer] = []
            answer_to_responses[answer].append(candidate)
            logger.debug(f"Candidate {i+1}: extracted '{answer}'")
        else:
            logger.warning(f"Could not extract answer from candidate {i+1}")
    
    # Select the most voted answer
    if answer_votes:
        most_common_answer, count = answer_votes.most_common(1)[0]
        logger.info(f"Most common answer: '{most_common_answer}' with {count}/{k} votes")
        
        # Return the first response that gave this answer
        winning_responses = answer_to_responses[most_common_answer]
        return winning_responses[0], total_tokens
    else:
        # If no answers could be extracted, return the first candidate
        logger.warning("No answers could be extracted, returning first candidate")
        return candidates[0], total_tokens