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
from fractions import Fraction

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "majority_voting"

# Default configuration
DEFAULT_K = 8
DEFAULT_TEMPERATURE = 0.3  # Lower for better consistency

# Category-specific temperatures
CATEGORY_TEMPERATURES = {
    "gsm8k": 0.2,      # Math needs precision
    "mmlu_math": 0.3,  # Multiple choice math
    "boolq": 0.3,      # Boolean questions
    "aqua_rat": 0.3,   # Reasoning with choices
    "default": 0.3     # General default
}

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


def extract_answer_by_category(text: str, category: str) -> Optional[str]:
    """
    Extract answer based on problem category.
    
    Args:
        text: Response text
        category: Problem category
        
    Returns:
        Extracted answer or None
    """
    text = text.strip()
    
    if category == "gsm8k":
        # Look for ### pattern specifically
        match = re.search(r'###\s*(-?\d*\.?\d+)', text)
        if match:
            return match.group(1)
        
        # Fallback: look for "answer is" pattern with number
        match = re.search(r'answer\s+is\s*:?\s*\$?(-?\d*\.?\d+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
            
    elif category == "mmlu_math":
        # Look for letter choices first
        patterns = [
            r'\b([A-E])\b(?:\s*\)|:|\.)?(?:\s|$)',  # Letter with optional punctuation
            r'(?:answer|choice|option)\s*(?:is\s*)?:?\s*([A-E])\b',
            r'^([A-E])$',  # Just a letter
            r'\b([0-3])\b(?:\s*\)|:|\.)?(?:\s|$)',  # Index (0-3)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
                
    elif category == "boolq":
        # Extract boolean answers
        text_lower = text.lower()
        
        # Direct true/false
        if re.search(r'\b(true|false)\b', text_lower):
            match = re.search(r'\b(true|false)\b', text_lower)
            return match.group(1)
        
        # Yes/no
        if re.search(r'\b(yes|no)\b', text_lower):
            match = re.search(r'\b(yes|no)\b', text_lower)
            return match.group(1)
            
    elif category == "aqua_rat":
        # Similar to MMLU but may have more complex patterns
        patterns = [
            r'(?:answer|option)\s*(?:is\s*)?:?\s*\(?([A-E])\)?',
            r'\b([A-E])\s*\)',
            r'^([A-E])$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
    
    # If category-specific extraction fails, fall back to generic
    return extract_answer(text)


def extract_answer(text: str) -> Optional[str]:
    """
    Generic answer extraction fallback.
    Enhanced from original version.
    """
    text = text.strip()
    
    # LaTeX boxed format
    boxed_match = re.search(r'\\{1,2}boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Answer patterns
    answer_patterns = [
        r'(?:final\s+)?answer\s*[:=]\s*(.+?)(?:\n|$)',
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so)\s*,?\s*(.+?)(?:\n|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().rstrip('.,;')
            if answer:
                return answer
    
    # Check last line if short
    lines = text.split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line and len(last_line) < 50 and not last_line.endswith(':'):
            return last_line
    
    return None


def normalize_answer_enhanced(answer: str, category: str = "default") -> str:
    """
    Enhanced answer normalization with category awareness.
    
    Args:
        answer: Raw answer text
        category: Problem category for specific normalization
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
        
    # Basic normalization
    answer = answer.lower().strip()
    answer = answer.strip('"\'')
    answer = ' '.join(answer.split())
    
    # Category-specific normalization
    if category in ["gsm8k", "mmlu_math"] and re.match(r'^-?\d*\.?\d+$', answer):
        # Numeric normalization
        try:
            # Handle different number formats
            answer = answer.replace(',', '')  # Remove commas
            
            # Convert to float for consistent representation
            num = float(answer)
            
            # Handle integers
            if num.is_integer():
                return str(int(num))
            else:
                # Format to remove trailing zeros
                return f"{num:g}"
                
        except ValueError:
            pass
    
    elif category == "mmlu_math":
        # Ensure single letter answers are uppercase
        if len(answer) == 1 and answer.isalpha():
            return answer.upper()
        
        # Extract letter from "option A", "choice B", etc.
        match = re.match(r'(?:option|choice|answer)\s*([a-e])', answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    elif category == "boolq":
        # Boolean normalization
        true_values = ['yes', 'true', 'correct', '1', 't', 'y']
        false_values = ['no', 'false', 'incorrect', '0', 'f', 'n']
        
        if answer in true_values:
            return 'true'
        elif answer in false_values:
            return 'false'
    
    # Handle mathematical expressions
    if category in ["gsm8k", "mmlu_math"]:
        # Try to evaluate simple fractions
        fraction_match = re.match(r'^(\d+)/(\d+)$', answer)
        if fraction_match:
            try:
                frac = Fraction(int(fraction_match.group(1)), int(fraction_match.group(2)))
                return str(float(frac))
            except:
                pass
        
        # Handle percentages
        percent_match = re.match(r'^(\d*\.?\d+)%$', answer)
        if percent_match:
            try:
                return str(float(percent_match.group(1)) / 100)
            except:
                pass
    
    return answer


def score_response_quality(response: str, category: str) -> float:
    """
    Score response quality for weighted voting.
    
    Returns:
        Quality score between 0 and 1
    """
    if not response:
        return 0.0
    
    score = 1.0
    
    # Check for completeness
    if len(response.strip()) < 10:
        score *= 0.5
    
    # Check for uncertainty markers
    uncertainty_words = ['maybe', 'probably', 'might', 'could be', 'not sure', 'guess']
    for word in uncertainty_words:
        if word in response.lower():
            score *= 0.7
    
    # Category-specific checks
    if category == "gsm8k":
        # Should have ### marker
        if "###" not in response:
            score *= 0.8
    elif category in ["mmlu_math", "aqua_rat"]:
        # Should have clear choice indication
        if not re.search(r'\b[A-E]\b', response):
            score *= 0.8
    
    # Check if response seems cut off
    if response.strip().endswith(('...', 'Therefore,', 'So,', 'Thus,')):
        score *= 0.5
    
    return score


def add_self_consistency_prompt(system_prompt: str, query: str, category: str) -> str:
    """
    Add format instructions to encourage consistency.
    """
    format_instructions = {
        "gsm8k": "\n\nIMPORTANT: After showing your work, provide your final numerical answer after ### on a new line.",
        "mmlu_math": "\n\nIMPORTANT: After your reasoning, clearly state your choice as a single letter (A, B, C, D, or E).",
        "boolq": "\n\nIMPORTANT: After your analysis, clearly state your answer as either 'true' or 'false'.",
        "aqua_rat": "\n\nIMPORTANT: After solving the problem, clearly indicate your choice with the letter (A, B, C, D, or E).",
        "default": "\n\nIMPORTANT: Clearly state your final answer at the end of your response."
    }
    
    instruction = format_instructions.get(category, format_instructions["default"])
    return system_prompt + instruction


def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    """
    Enhanced majority voting with category awareness and better extraction.
    """
    logger.info("Starting enhanced majority voting process")
    
    # Detect category
    category = detect_category(initial_query)
    logger.info(f"Detected category: {category}")
    
    # Extract parameters
    k = request_config.get('k', DEFAULT_K) if request_config else DEFAULT_K
    
    # Use category-specific temperature
    base_temperature = CATEGORY_TEMPERATURES.get(category, DEFAULT_TEMPERATURE)
    temperature = request_config.get('temperature', base_temperature) if request_config else base_temperature
    
    max_tokens = request_config.get('max_tokens', 4096) if request_config else 4096
    
    logger.info(f"Generating {k} candidates with temperature={temperature} for category={category}")
    
    # Add self-consistency prompt
    enhanced_system_prompt = add_self_consistency_prompt(system_prompt, initial_query, category)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
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
    
    # Extract and normalize answers with quality scores
    answer_data = []  # List of (normalized_answer, raw_answer, response, quality_score)
    
    for i, candidate in enumerate(candidates):
        # Extract answer using category-aware extraction
        answer = extract_answer_by_category(candidate, category)
        
        if answer:
            normalized = normalize_answer_enhanced(answer, category)
            quality = score_response_quality(candidate, category)
            answer_data.append((normalized, answer, candidate, quality))
            logger.debug(f"Candidate {i+1}: {answer} -> {normalized} (quality: {quality:.2f})")
        else:
            logger.warning(f"Could not extract answer from candidate {i+1}")
    
    if not answer_data:
        # Fallback: return highest quality response
        quality_scores = [(score_response_quality(c, category), c) for c in candidates]
        quality_scores.sort(reverse=True)
        return quality_scores[0][1], total_tokens
    
    # Count weighted votes
    weighted_votes = Counter()
    answer_to_response = {}
    
    for normalized, raw, response, quality in answer_data:
        weighted_votes[normalized] += quality
        # Keep the highest quality response for each answer
        if normalized not in answer_to_response or quality > answer_to_response[normalized][1]:
            answer_to_response[normalized] = (response, quality)
    
    # Get the answer with highest weighted votes
    most_common_answer, weighted_score = weighted_votes.most_common(1)[0]
    
    # Calculate confidence
    total_weight = sum(weighted_votes.values())
    confidence = weighted_score / total_weight if total_weight > 0 else 0
    
    logger.info(f"Most common answer: '{most_common_answer}' with weighted score {weighted_score:.2f} ({confidence:.1%} confidence)")
    
    # Return the best response for the winning answer
    winning_response = answer_to_response[most_common_answer][0]
    
    return winning_response, total_tokens