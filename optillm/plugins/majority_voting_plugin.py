"""
Majority Voting Plugin for OptILLM

This plugin implements a majority voting approach where k candidate solutions
are generated and the most frequent answer is selected. This is particularly
effective for problems with discrete answers (math, coding, multiple choice).

The plugin uses the OpenAI API's n parameter to generate multiple responses
efficiently in a single API call.
"""

import re
import logging
from typing import Tuple, Dict, Any, List, Optional
from collections import Counter
import json

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "majority_voting"

# Default number of candidates to generate
DEFAULT_K = 6

# Default temperature for candidate generation
DEFAULT_TEMPERATURE = 0.6

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from a response text.
    
    This function looks for common answer patterns in the response:
    1. Text after "Answer:" or "Final Answer:"
    2. Text within \\boxed{} (LaTeX format)
    3. Numbers at the end of the response
    4. The last line if it's short (likely the answer)
    
    Args:
        text: The response text to extract answer from
        
    Returns:
        The extracted answer or None if no clear answer found
    """
    # Remove any trailing whitespace
    text = text.strip()
    
    # Pattern 1: Look for LaTeX boxed format first (handle both \boxed and \\boxed)
    boxed_match = re.search(r'\\{1,2}boxed\{([^}]+)\}', text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        logger.debug(f"Extracted boxed answer: {answer}")
        return answer
    
    # Pattern 2: Look for "Answer:" or "Final Answer:" patterns
    answer_patterns = [
        r'(?:final\s+)?answer\s*[:=]\s*(.+?)(?:\n|$)',
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so)\s*,?\s*(.+?)(?:\n|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = answer.rstrip('.,;')
            if answer:
                logger.debug(f"Extracted answer using pattern: {answer}")
                return answer
    
    # Pattern 3: Look for standalone numbers (useful for math problems)
    # Check the last few lines for a number
    lines = text.split('\n')
    for line in reversed(lines[-3:]):  # Check last 3 lines
        line = line.strip()
        # Match numbers (including decimals, fractions, negative numbers)
        number_match = re.match(r'^-?\d+\.?\d*$|^-?\d+/\d+$', line)
        if number_match:
            logger.debug(f"Extracted number answer: {line}")
            return line
    
    # Pattern 4: For multiple choice, look for single letter answers
    # Check this before the generic last line check
    mc_patterns = [
        r'(?:the\s+)?(?:correct\s+)?(?:answer|option)\s+is\s+([A-E])(?:\b|$)',
        r'(?:choose|select|pick)\s+(?:option\s+)?([A-E])(?:\b|$)',
        r'\b([A-E])\s*\)\s*[A-Za-z]+.*is\s+(?:the\s+)?(?:correct|right)',
        r'^([A-E])$',  # Just a letter on its own line
    ]
    
    for pattern in mc_patterns:
        mc_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if mc_match:
            answer = mc_match.group(1).upper()
            logger.debug(f"Extracted multiple choice answer: {answer}")
            return answer
    
    # Pattern 5: If the last line is short (< 50 chars), it might be the answer
    if lines:
        last_line = lines[-1].strip()
        if last_line and len(last_line) < 50 and not last_line.endswith(':'):
            logger.debug(f"Using last line as answer: {last_line}")
            return last_line
    
    logger.warning("Could not extract a clear answer from the response")
    return None

def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    This helps ensure that equivalent answers are treated as the same:
    - Converts to lowercase
    - Removes extra whitespace
    - Removes quotes
    - Normalizes number formats
    
    Args:
        answer: The answer to normalize
        
    Returns:
        The normalized answer
    """
    # Convert to lowercase
    answer = answer.lower().strip()
    
    # Remove quotes
    answer = answer.strip('"\'')
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    # Try to normalize numbers
    try:
        # Check if it's a float
        if '.' in answer:
            num = float(answer)
            # Format to remove trailing zeros
            answer = f"{num:g}"
        else:
            # Try integer
            num = int(answer)
            answer = str(num)
    except ValueError:
        # Not a number, keep as is
        pass
    
    # Handle yes/no variations
    if answer in ['yes', 'yeah', 'yep', 'true', 'correct']:
        answer = 'yes'
    elif answer in ['no', 'nope', 'false', 'incorrect']:
        answer = 'no'
    
    return answer

def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    """
    Main entry point for the majority voting plugin.
    
    Generates k candidate solutions and returns the most frequent answer.
    
    Args:
        system_prompt: System prompt for the model
        initial_query: User's query
        client: OpenAI-compatible client instance
        model: Model identifier
        request_config: Additional configuration parameters
        
    Returns:
        Tuple of (response_text, completion_tokens_used)
    """
    logger.info("Starting majority voting process")
    
    # Extract parameters from request_config
    k = DEFAULT_K
    temperature = DEFAULT_TEMPERATURE
    
    if request_config:
        k = request_config.get('k', DEFAULT_K)
        # Allow overriding temperature if needed
        temperature = request_config.get('temperature', DEFAULT_TEMPERATURE)
        # Respect max_tokens if provided
        max_tokens = request_config.get('max_tokens', 4096)
    else:
        max_tokens = 4096
    
    logger.info(f"Generating {k} candidates with temperature={temperature}")
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_query}
    ]
    
    try:
        # Generate k candidates in a single API call using n parameter
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=k,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract all candidate responses
        candidates = [choice.message.content for choice in response.choices]
        total_tokens = response.usage.completion_tokens
        
        logger.info(f"Generated {len(candidates)} candidates. Tokens used: {total_tokens}")
        
        # Extract answers from each candidate
        answers = []
        answer_to_response = {}  # Map normalized answers to full responses
        
        for i, candidate in enumerate(candidates):
            answer = extract_answer(candidate)
            if answer:
                normalized = normalize_answer(answer)
                answers.append(normalized)
                # Keep the first full response for each unique answer
                if normalized not in answer_to_response:
                    answer_to_response[normalized] = candidate
                logger.debug(f"Candidate {i+1} answer: {answer} (normalized: {normalized})")
            else:
                logger.warning(f"Could not extract answer from candidate {i+1}")
        
        if not answers:
            logger.warning("No answers could be extracted from any candidate")
            # Return the first candidate as fallback
            return candidates[0] if candidates else "Error: No candidates generated", total_tokens
        
        # Count answer frequencies
        answer_counts = Counter(answers)
        logger.info(f"Answer distribution: {dict(answer_counts)}")
        
        # Get the most common answer
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / len(answers)
        
        logger.info(f"Most common answer: '{most_common_answer}' with {count}/{len(answers)} votes ({confidence:.1%} confidence)")
        
        # Get the full response corresponding to the most common answer
        winning_response = answer_to_response.get(most_common_answer, candidates[0])
        
        # Log voting summary to console instead of adding to response
        logger.info("Majority Voting Summary:")
        logger.info(f"  - Generated {k} candidates")
        logger.info(f"  - Most common answer: {most_common_answer}")
        logger.info(f"  - Votes: {count}/{len(answers)} ({confidence:.1%} confidence)")
        
        if len(answer_counts) > 1:
            other_answers = [f"{ans} ({cnt} votes)" for ans, cnt in answer_counts.items() if ans != most_common_answer]
            logger.info(f"  - Other answers: {', '.join(other_answers)}")
        
        # Return only the full response from the winning answer
        return winning_response, total_tokens
        
    except Exception as e:
        logger.error(f"Error in majority voting: {str(e)}")
        # Fall back to single response
        logger.info("Falling back to single response generation")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content, response.usage.completion_tokens