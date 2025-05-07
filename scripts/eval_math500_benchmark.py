import argparse
import json
import os
import logging
import re
from typing import Dict, Optional, Union, List, Tuple
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Enhanced logging configuration
logger = logging.getLogger(__name__)

# Initialize OpenAI client
# client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

# Define the three system prompts for our experiment
STANDARD_PROMPT = '''You are solving mathematics problems.

Important: Provide your final answer in this format:

\\[
\\boxed{your_answer_here}
\\]

The entire answer should be contained completely within the \\boxed{} command.'''

COT_PROMPT = '''You are solving mathematics problems.

Please think step by step.

Important: Always end your solution with the final answer in this format:

\\[
\\boxed{your_answer_here}
\\]

The entire answer should be contained completely within the \\boxed{} command.'''

GIBBERISH_PROMPT = '''You are solving mathematics problems.

First, write some meaningless text of similar length to what you would use for step-by-step reasoning. This text should be complete nonsense with no actual reasoning. Make sure it has a similar number of paragraphs and similar length to how you would normally explain the problem, but ensure the content is completely unrelated to the solution.

Important: Always end your solution with the final answer in this format:

\\[
\\boxed{your_answer_here}
\\]

The entire answer should be contained completely within the \\boxed{} command.'''

# Dictionary to map prompt types to their actual text
PROMPTS = {
    "standard": STANDARD_PROMPT,
    "cot": COT_PROMPT,
    "gibberish": GIBBERISH_PROMPT
}

def load_math500_dataset() -> list[dict]:
    """
    Load the MATH-500 dataset.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    dataset = dataset["test"]
    logging.debug(f"Dataset size: {len(dataset)}.")
    return dataset

def extract_answer(response: str) -> Optional[str]:
    """Extract the answer from a math solution response."""
    if not response:
        logger.debug("Empty response received")
        return None
    
    # Find the last \boxed{...} in the response
    start_idx = response.rfind('\\boxed{')
    if start_idx == -1:
        logger.debug("No \\boxed{} found in response")
        return None
        
    # Find the matching closing brace
    brace_count = 1
    pos = start_idx + 7  # length of '\boxed{'
    
    while pos < len(response) and brace_count > 0:
        if response[pos] == '{':
            brace_count += 1
        elif response[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        answer = response[start_idx + 7:pos - 1]
        logger.debug(f"Extracted answer: {answer}")
        return answer.strip()
    
    logger.debug("No matching closing brace found")
    return None

def normalize_number(num_str: str) -> str:
    """Helper function to normalize number representation."""
    try:
        # Remove commas, currency symbols, units, and whitespace
        cleaned = re.sub(r'[,\$\\]|\s*(?:cm|m|kg|ft|in|lb|oz|ml|L)$|\s*\\text{[^}]+}', '', num_str).strip()
        
        # Handle leading decimal point
        if cleaned.startswith('.'):
            cleaned = '0' + cleaned
            
        # Convert to float
        num = float(cleaned)
        
        # For small decimals, preserve exact representation
        if abs(num) < 1 and '.' in cleaned:
            # Count original decimal places
            decimal_places = len(cleaned.split('.')[1])
            format_str = f"{{:.{decimal_places}f}}"
            result = format_str.format(num)
        else:
            result = str(num)
        
        logger.debug(f"Normalized number result: {repr(result)}")
        return result
    except Exception as e:
        logger.debug(f"Failed to normalize number: {str(e)}")
        return num_str

def numerically_equal(str1: str, str2: str) -> bool:
    """Compare if two numeric strings represent the same value."""
    try:
        return abs(float(str1) - float(str2)) < 1e-10
    except:
        return False
    
def normalize_fraction(fraction_str: str) -> str:
    """Helper function to normalize fractions."""
    logger.debug(f"Normalizing fraction: {repr(fraction_str)}")
    try:
        # Convert \dfrac to \frac
        fraction_str = fraction_str.replace('\\dfrac', '\\frac')
        
        # Remove all whitespace
        fraction_str = ''.join(fraction_str.split())
        
        # Remove any trailing text
        fraction_str = re.sub(r'\s*\\text{[^}]+}', '', fraction_str)
        
        # Handle mixed brace format first (\frac9{19})
        mixed_brace = re.match(r'^\\frac(\d+)\{(\d+)\}$', fraction_str)
        if mixed_brace:
            num, den = mixed_brace.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Handle no braces format (\frac12)
        no_braces = re.match(r'^\\frac(\d+)(\d+)$', fraction_str)
        if no_braces:
            num, den = no_braces.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Handle a/b format
        if '/' in fraction_str and not any(c in fraction_str for c in '\\{}'):
            num, den = fraction_str.split('/')
            return f"\\frac{{{num.strip()}}}{{{den.strip()}}}"
        
        # Handle standard \frac{a}{b}
        standard = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', fraction_str)
        if standard:
            num, den = standard.groups()
            return f"\\frac{{{num}}}{{{den}}}"
            
    except Exception as e:
        logger.debug(f"Failed to normalize fraction: {str(e)}")
        logger.debug(f"Original fraction string: {repr(fraction_str)}")
    return fraction_str

def normalize_matrix_entry(entry: str) -> str:
    """Helper function to normalize a single matrix entry."""
    logger.debug(f"Normalizing matrix entry input: {repr(entry)}")
    
    # Remove all spaces first
    entry = ''.join(entry.split())
    
    # If it's already in simple a/b format, standardize spacing
    if '/' in entry and not any(c in entry for c in '\\{}'):
        if entry.startswith('-'):
            num, den = entry[1:].split('/')
            return f"-{num.strip()}/{den.strip()}"
        else:
            num, den = entry.split('/')
            return f"{num.strip()}/{den.strip()}"
            
    # Convert \dfrac to \frac
    entry = entry.replace('\\dfrac', '\\frac')
    
    # Handle LaTeX fractions
    frac_match = re.match(r'^(-)?\\frac\{(\d+)\}\{(\d+)\}$', entry)
    if frac_match:
        sign, num, den = frac_match.groups()
        sign = sign if sign else ''
        return f"{sign}{num}/{den}"
    
    return entry

def normalize_matrix(matrix_str: str) -> str:
    """Helper function to normalize matrices and vectors."""
    logger.debug(f"Normalizing matrix input: {repr(matrix_str)}")
    try:
        # Remove all whitespace
        matrix_str = ''.join(matrix_str.split())
        
        # Extract the matrix content
        match = re.match(r'^\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}$', matrix_str)
        if not match:
            return matrix_str
            
        content = match.group(1)
        rows = content.split('\\\\')
        
        # Normalize each entry in each row
        normalized_rows = []
        for row in rows:
            if '&' in row:
                entries = [normalize_matrix_entry(entry) for entry in row.split('&')]
            else:
                entries = [normalize_matrix_entry(row)]
            normalized_rows.append('&'.join(entries))
        
        # Reconstruct the matrix
        result = "\\begin{pmatrix}" + "\\\\".join(normalized_rows) + "\\end{pmatrix}"
        logger.debug(f"Normalized matrix result: {repr(result)}")
        return result
        
    except Exception as e:
        logger.debug(f"Failed to normalize matrix: {str(e)}")
        return matrix_str

def normalize_algebraic_expression(expr: str) -> str:
    """Helper function to normalize algebraic expressions."""
    logger.debug(f"Normalizing algebraic expression: {repr(expr)}")
    try:
        # Remove all whitespace
        expr = ''.join(expr.split())
        
        # Handle simple monomial with exponent (e.g., 5r^5)
        monomial_match = re.match(r'^(-?\d*\.?\d*)?([a-zA-Z])(?:\^(-?\d+))?$', expr)
        if monomial_match:
            coeff, var, exp = monomial_match.groups()
            coeff = coeff if coeff and coeff not in ['+', '-'] else ('1' if not coeff else '-1')
            exp = exp if exp else '1'
            if coeff == '1' and exp == '1':
                result = var
            elif coeff == '1':
                result = f"{var}^{exp}"
            elif coeff == '-1' and exp == '1':
                result = f"-{var}"
            elif coeff == '-1':
                result = f"-{var}^{exp}"
            elif exp == '1':
                result = f"{coeff}{var}"
            else:
                result = f"{coeff}{var}^{exp}"
            logger.debug(f"Matched as monomial with exponent: {repr(result)}")
            return result.lower()
            
        # Special case: If it's a single term with π
        pi_term_match = re.match(r'^(-?\d*\.?\d*)\\?pi$', expr)
        if pi_term_match:
            coeff = pi_term_match.group(1)
            if not coeff or coeff == '-':
                coeff = '-1' if coeff == '-' else '1'
            return f"{coeff}\\pi"
            
        # Handle fractions with π
        frac_pi_match = re.match(r'^\\frac{([^{}]+)}{([^{}]+)}\\?pi$', expr)
        if frac_pi_match:
            num, den = frac_pi_match.groups()
            return f"\\frac{{{num}}}{{{den}}}\\pi"
        
        # Handle basic fractions
        frac_match = re.match(r'^\\frac{([^{}]+)}{([^{}]+)}$', expr)
        if frac_match:
            num, den = frac_match.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Split into terms (handle both + and -)
        terms = []
        current_term = ""
        for i, char in enumerate(expr):
            if char in ['+', '-'] and i > 0:
                if current_term:
                    terms.append(current_term)
                current_term = char
            else:
                current_term += char
        if current_term:
            terms.append(current_term)
        
        # If it's just a single number, return normalized version
        if len(terms) == 1 and re.match(r'^-?[\d,]+$', terms[0]):
            return normalize_number(terms[0])
            
        # Process each term and sort
        processed_terms = []
        for term in terms:
            # Handle leading + if present
            if term.startswith('+'):
                term = term[1:]
                
            # Add implicit + for positive terms
            if not term.startswith('-'):
                term = '+' + term
                
            # Separate coefficient and variable parts
            match = re.match(r'^([+-])?\s*(\d*\.?\d*)?([a-zA-Z](?:\^\d+)?)?$', term)
            if match:
                sign, coeff, var = match.groups()
                # Handle default coefficients
                if not coeff and var:
                    coeff = '1'
                elif not coeff:
                    coeff = '0'
                # Create standardized term
                processed_terms.append((sign, float(coeff), var or ''))
        
        # Sort terms: variables first (in alphabetical order), then constants
        processed_terms.sort(key=lambda x: (not bool(x[2]), x[2], -x[1]))
        
        # Reconstruct the expression
        result = ""
        for sign, coeff, var in processed_terms:
            if coeff == 0:
                continue
            term = ""
            if coeff == 1 and var:
                term = var
            elif coeff == -1 and var:
                term = f"-{var}"
            elif var:
                term = f"{coeff}{var}"
            else:
                term = str(coeff)
            
            if result and term[0] != '-':
                result += '+'
            result += term
        
        logger.debug(f"Normalized algebraic expression result: {repr(result)}")
        return result.lower()
    except Exception as e:
        logger.debug(f"Failed to normalize algebraic expression: {str(e)}")
        return expr.lower()  # Return lowercased original if normalization fails
    
def normalize_interval_bound(bound: str) -> str:
    """Helper function to normalize interval bounds."""
    logger.debug(f"Normalizing interval bound: {repr(bound)}")
    
    # Handle infinity
    if '\\infty' in bound:
        sign = '-' if bound.startswith('-') else ''
        return f"{sign}\\infty"
        
    # For other bounds, use regular answer normalization
    return normalize_answer(bound) or bound

def normalize_interval(interval_str: str) -> str:
    """Helper function to normalize intervals."""
    logger.debug(f"Normalizing interval: {repr(interval_str)}")
    try:
        # Remove all whitespace first
        interval_str = ''.join(interval_str.split())
        
        # Extract the interval content, handling \left and \right
        # Fixed regex to avoid nested set warning by using explicit character classes
        match = re.match(r'^\\left?([\[\(])(.*?),(.*?)\\right?([\]\)])$', interval_str)
        if not match:
            # Try without \left and \right
            match = re.match(r'^([\[\(])(.*?),(.*?)([\]\)])$', interval_str)
            if not match:
                return interval_str
                
        left_bracket, left_bound, right_bound, right_bracket = match.groups()
        
        # Normalize each bound
        norm_left = normalize_interval_bound(left_bound)
        norm_right = normalize_interval_bound(right_bound)
        
        # Reconstruct the interval
        result = f"\\left{left_bracket}{norm_left},{norm_right}\\right{right_bracket}"
        logger.debug(f"Normalized interval result: {repr(result)}")
        return result
        
    except Exception as e:
        logger.debug(f"Failed to normalize interval: {str(e)}")
        return interval_str
    
def normalize_ordered_tuple(tuple_str: str) -> str:
    """Helper function to normalize ordered tuples/lists of numbers."""
    logger.debug(f"Normalizing tuple: {repr(tuple_str)}")
    try:
        # First standardize \dfrac to \frac
        tuple_str = tuple_str.replace('\\dfrac', '\\frac')
        
        # Remove \left and \right
        tuple_str = tuple_str.replace('\\left', '').replace('\\right', '')
        
        # Remove all spaces and backslash spaces
        tuple_str = re.sub(r'\\?\s+', '', tuple_str)
        
        # Remove outer parentheses and split by commas
        inner = tuple_str.strip('()')
        parts = inner.split(',')
        
        # Normalize each part
        normalized_parts = []
        for part in parts:
            norm_part = normalize_answer(part.strip())
            if not norm_part:  # If any part fails to normalize, return None
                logger.debug(f"Failed to normalize part: {part}")
                return None
            normalized_parts.append(norm_part)
            
        # Always reconstruct with standard format (using parentheses)
        result = f"({','.join(normalized_parts)})"
        logger.debug(f"Normalized tuple result: {repr(result)}")
        return result
    except Exception as e:
        logger.debug(f"Failed to normalize tuple: {str(e)}")
        return None

def normalize_answer(answer: str) -> str:
    """Normalize the answer string for comparison."""
    logger.debug(f"Normalizing answer: {repr(answer)}")
    
    if answer is None:
        logger.debug("Received None answer")
        return ""
    
    # Remove \text{} with units first
    answer = re.sub(r'\\text{[^}]+(?:inches|feet|meters|cm|m|kg|ft|in|lb|oz|ml|L|per|second|minute|hour)[^}]*}', '', answer)
    

    # Remove all whitespace first but preserve backslash space temporarily
    answer = re.sub(r'(?<!\\)\s+', '', answer)
    logger.debug(f"After initial whitespace removal: {repr(answer)}")
    
    # Then handle ordered pairs/tuples with potential \left, \right
    ordered_pair_match = re.match(r'^(?:\\left)?\((.*?)(?:\\right)?\)$', answer)
    if ordered_pair_match:
        content = ordered_pair_match.group(1)
        # Split by comma and normalize each part
        parts = content.split(',')
        normalized_parts = []
        for part in parts:
            # Remove any remaining backslash spaces
            part = re.sub(r'\\?\s+', '', part)
            norm_part = normalize_answer(part)
            if norm_part is None:
                return None
            normalized_parts.append(norm_part)
        return f"({','.join(normalized_parts)})"
    
    # Remove all whitespace
    answer = ''.join(answer.split())
    logger.debug(f"After whitespace removal: {repr(answer)}")
    
    if not answer:
        logger.debug("Answer became empty after whitespace removal")
        return None
    
    # Handle plus-minus expressions first
    # This will match both forms: "a \pm b" and "a - b"
    pm_match = re.match(r'^(.*?)(?:\\pm|-)(.*?)$', answer)
    if pm_match:
        left, right = pm_match.groups()
        # Normalize both sides
        norm_left = normalize_answer(left) if left else ""
        norm_right = normalize_answer(right) if right else ""
        if norm_left or norm_right:  # If either side normalized successfully
            # Always use \pm in the normalized form
            result = f"{norm_left}\\pm{norm_right}"
            logger.debug(f"Matched as plus-minus expression: {repr(result)}")
            return result
    
    # Handle trigonometric functions
    trig_match = re.match(r'^\\(?:sin|cos|tan|cot|sec|csc)\s*([a-zA-Z])$', answer)
    if trig_match:
        variable = trig_match.group(1)
        # Get the function name without the backslash
        func_name = re.match(r'^\\(.*?)(?:\s|$)', answer).group(1)
        result = f"\\{func_name}{variable}"
        logger.debug(f"Matched as trigonometric function: {repr(result)}")
        return result

    # Handle text-only answers first (including multiple choice)
    text_match = re.match(r'^(?:\\text{)?([A-Za-z]+)(?:})?$', answer)
    if text_match:
        result = text_match.group(1).lower()
        logger.debug(f"Matched as text answer: {repr(result)}")
        return result

    # Handle intervals first (with or without \left and \right)
    if (answer.startswith('\\left[') or answer.startswith('\\left(') or 
        answer.startswith('[') or answer.startswith('(')) and \
       (answer.endswith('\\right]') or answer.endswith('\\right)') or 
        answer.endswith(']') or answer.endswith(')')):
        result = normalize_interval(answer)
        if result:
            logger.debug(f"Matched as interval: {repr(result)}")
            return result
    
    # Handle matrices/vectors
    if answer.startswith('\\begin{pmatrix}') and answer.endswith('\\end{pmatrix}'):
        result = normalize_matrix(answer)
        if result:
            logger.debug(f"Matched as matrix: {repr(result)}")
            return result
    
    # Normalize all fraction commands to \frac first
    answer = answer.replace('\\dfrac', '\\frac')

    # Handle fractions (both \frac and \dfrac)
    if '\\frac' in answer or '\\dfrac' in answer or '/' in answer:
        result = normalize_fraction(answer)
        if result:
            logger.debug(f"Matched as fraction: {repr(result)}")
            return result

    # Handle negative square roots first (before other square root handling)
    neg_sqrt_match = re.match(r'^-\\sqrt\{?(\d+)\}?$', answer)
    if neg_sqrt_match:
        num = neg_sqrt_match.group(1)
        result = f"-\\sqrt{{{num}}}"
        logger.debug(f"Matched as negative square root: {repr(result)}")
        return result

    # Handle direct square root expressions first
    logger.debug("Checking for square root pattern...")
    sqrt_match = re.match(r'^(\d*)?\\sqrt\{?(\d+)\}?$', answer)
    if sqrt_match:
        coeff, num = sqrt_match.groups()
        coeff = coeff if coeff else '1'
        if coeff == '1':
            result = f"\\sqrt{{{num}}}"
        else:
            result = f"{coeff}\\sqrt{{{num}}}"
        logger.debug(f"Matched as pure square root: {repr(result)}")
        return result

    # Now handle coefficient with square root
    sqrt_with_coeff_match = re.match(r'^(\d+)\\sqrt\{?(\d+)\}?$', answer)
    if sqrt_with_coeff_match:
        coeff, num = sqrt_with_coeff_match.groups()
        result = f"{coeff}\\sqrt{{{num}}}"
        logger.debug(f"Matched as coefficient with square root: {repr(result)}")
        return result
    
    # Handle numbers with base subscripts
    base_match = re.match(r'^(\d+)(?:_\{?(\d+)\}?|_(\d+))$', answer)
    if base_match:
        number, base1, base2 = base_match.groups()
        base = base1 if base1 else base2
        result = f"{number}_{base}"
        logger.debug(f"Matched as base number: {repr(result)}")
        return result

    # Handle numbers with percentage sign first
    percent_match = re.match(r'^(\d+(?:\.\d*)?)\s*\\?%$', answer)
    if percent_match:
        number = percent_match.group(1)
        result = normalize_number(number)
        logger.debug(f"Matched as percentage: {repr(result)}")
        return result
    
    # Handle numbers with units (including LaTeX spaces and comma-separated units)
    unit_match = re.match(r'^(\d+(?:\.\d*)?)\s*(?:(?:\\[,\s])|,)?\s*(?:\\\\)?(?:\\text{(\w+)}|\\?(?:cm|m|kg|ft|in|lb|oz|ml|L))$', answer)
    if unit_match:
        number = unit_match.group(1)
        result = normalize_number(number)
        logger.debug(f"Matched as number with unit: {repr(result)}")
        return result
    
    # Try to handle currency values first
    currency_match = re.match(r'^\\?\$?([\d,]+\.?\d*)$', answer)
    if currency_match:
        result = normalize_number(currency_match.group(1))
        logger.debug(f"Matched as currency: {repr(result)}")
        return result
    
    # Try to handle pure numbers with commas first
    if re.match(r'^-?[\d,]+$', answer):
        result = normalize_number(answer)
        logger.debug(f"Matched as number: {repr(result)}")
        return result
    
    # Try to extract numeric value with optional units
    unit_match = re.match(r'^(-?[\d,]+(?:\.\d*)?)\s*(?:\\(?:mbox|text|hbox|displaystyle)\{[^}]+\})?(?:\^?\d)?$', answer)
    if unit_match:
        result = normalize_number(unit_match.group(1))
        logger.debug(f"Matched as number with units: {repr(result)}")
        return result
    
    # Handle multiple choice answers
    mc_match = re.match(r'^\\text{\(?([A-Za-z])\)?}$|^\(?([A-Za-z])\)?$', answer)
    if mc_match:
        result = (mc_match.group(1) or mc_match.group(2)).lower()
        logger.debug(f"Matched as multiple choice: {repr(result)}")
        return result
    
    # Handle degrees
    degree_match = re.match(r'^(-?[\d,]+(?:\.\d*)?)\s*(?:(?:\^?\\circ)|(?:{\\circ})|(?:°))?$', answer)
    if degree_match:
        result = normalize_number(degree_match.group(1))
        logger.debug(f"Matched as degrees: {repr(result)}")
        return result
    
    # Remove \text{} command without changing content FIRST
    answer = re.sub(r'\\text{([^{}]+)}', r'\1', answer)
    logger.debug(f"After \\text removal: {repr(answer)}")
    
    # Try to handle algebraic expressions
    try:
        result = normalize_algebraic_expression(answer)
        logger.debug(f"Normalized as algebraic expression: {repr(result)}")
        return result
    except:
        logger.debug("Failed to normalize as algebraic expression")
        pass
    
    # Remove \left and \right commands
    answer = answer.replace('\\left', '').replace('\\right', '')
    
    # Remove any remaining extra backslashes before common symbols
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\(', '(').replace('\\)', ')')
    answer = answer.replace('\\[', '[').replace('\\]', ']')
    answer = answer.replace('\\{', '{').replace('\\}', '}')
    
    # Normalize square roots consistently
    answer = re.sub(r'\\sqrt\{?(\d+)\}?', r'\\sqrt{\1}', answer)
    answer = re.sub(r'\\sqrt{([^{}]+)}', r'\\sqrt\1', answer)
    
    # Handle percentage notation
    if re.match(r'^\d+\\%$', answer) or re.match(r'^\d+$', answer):
        answer = re.sub(r'\\%$', '', answer)
    
    # Handle \text{} command again in case it was nested
    answer = re.sub(r'\\text{([^{}]+)}', r'\1', answer)
    
    # Strip unnecessary outer braces
    while len(answer) >= 2 and answer[0] == '{' and answer[-1] == '}':
        if '\\frac' in answer:
            break
        answer = answer[1:-1]
    
    result = answer.lower()
    logger.debug(f"Final normalized result: {repr(result)}")
    return result if result else None

def compare_answers(correct_answer: str, predicted_answer: Optional[str]) -> bool:
    """Compare the correct answer with the predicted answer."""
    logger.debug(f"Comparing answers - Correct: {repr(correct_answer)}, Predicted: {repr(predicted_answer)}")
    
    if predicted_answer is None:
        logger.debug("Predicted answer is None")
        return False
    
    # Try numerical comparison first
    if numerically_equal(correct_answer, predicted_answer):
        return True
        
    normalized_correct = normalize_answer(correct_answer)
    normalized_predicted = normalize_answer(predicted_answer)
    
    logger.debug(f"Normalized answers - Correct: {repr(normalized_correct)}, Predicted: {repr(normalized_predicted)}")
    
    # If either normalization returns None or empty string, answers don't match
    if not normalized_correct or not normalized_predicted:
        logger.debug("One or both normalized answers are None or empty")
        return False
        
    # If both answers became empty strings, they don't match
    if normalized_correct == "" and normalized_predicted == "":
        logger.debug("Both answers normalized to empty strings")
        return False
    
    # For intervals, they must match exactly (including brackets)
    if ('\\left[' in normalized_correct or '\\left(' in normalized_correct) and \
       ('\\left[' in normalized_predicted or '\\left(' in normalized_predicted):
        result = normalized_correct == normalized_predicted
        logger.debug(f"Interval comparison result: {result}")
        return result
    
    result = normalized_correct == normalized_predicted
    logger.debug(f"Comparison result: {result}")
    return result

def get_llm_response(problem: str, model: str, prompt_type: str = "cot") -> str:
    """
    Get response from the LLM for a given problem.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        prompt_type (str): Type of system prompt to use ('standard', 'cot', or 'gibberish')
        
    Returns:
        str: Model's response
    """
    try:
        selected_prompt = PROMPTS.get(prompt_type, COT_PROMPT)
        
        response = client.chat.completions.create(
            model=model,
            temperature=0.6,  # Lower temperature for more consistent answers
            messages=[
                {"role": "user", "content": selected_prompt + "\n" + problem}
            ],
            max_tokens=32768, # for thinking models, we need to use a lot more tokens
            # extra_body = {
            #     "decoding" : "thinkdeeper",
            # }
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return ""

def load_existing_results(filename: str) -> list[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def analyze_results(results: list[Dict], prompt_type: str = None):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (list[Dict]): List of evaluation results
        prompt_type (str, optional): The prompt type used for these results
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    prefix = f"[{prompt_type.upper()}] " if prompt_type else ""
    print(f"\n=== {prefix}Results Summary ===")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print(f"\n=== {prefix}Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print(f"Predicted: {r['predicted_answer']}")
            print("---")
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }

def estimate_reasoning_quality(response: str) -> float:
    """
    Estimate the quality of reasoning in a response.
    This is a simple heuristic to check if the text contains mathematical symbols,
    numbers, and relevant keywords.
    
    Args:
        response (str): The model's response
    
    Returns:
        float: A score between 0 and 1 indicating reasoning quality
    """
    # Check for common mathematical symbols and expressions
    math_symbols = ['=', '+', '-', '*', '/', '\\frac', '\\sqrt', '\\pi', '^', '<', '>']
    symbol_score = sum(1 for symbol in math_symbols if symbol in response) / len(math_symbols)
    
    # Check for numbers in the response
    number_count = len(re.findall(r'\d+', response))
    number_score = min(1.0, number_count / 20)  # Cap at 20 numbers
    
    # Check for reasoning keywords
    reasoning_keywords = ['therefore', 'thus', 'since', 'because', 'we have', 'we get', 'we find', 'we need to', 'let', 'first', 'second', 'finally']
    keyword_score = sum(1 for keyword in reasoning_keywords if keyword.lower() in response.lower()) / len(reasoning_keywords)
    
    # Combine scores with weights
    total_score = (0.4 * symbol_score) + (0.3 * number_score) + (0.3 * keyword_score)
    return total_score

def estimate_gibberish_level(response: str) -> float:
    """
    Estimate how gibberish-like a response is.
    This checks for coherent mathematical content vs random text.
    
    Args:
        response (str): The model's response
    
    Returns:
        float: A score between 0 and 1 indicating gibberish level (higher = more gibberish)
    """
    # Extract everything before the answer box
    reasoning_part = response.split('\\boxed{')[0] if '\\boxed{' in response else response
    
    # Check for mathematical coherence (inverse of reasoning quality)
    math_coherence = 1 - estimate_reasoning_quality(reasoning_part)
    
    # Check for unusual word patterns
    unusual_patterns = re.findall(r'\b\w{15,}\b', reasoning_part)  # Very long words
    unusual_pattern_score = min(1.0, len(unusual_patterns) / 5)  # Cap at 5 occurrences
    
    # Check for repetition
    words = re.findall(r'\b\w+\b', reasoning_part.lower())
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    repetition_score = 0
    if words:
        repetition_score = sum(1 for count in word_counts.values() if count > 3) / len(word_counts)
    
    # Combine scores
    total_score = (0.5 * math_coherence) + (0.3 * unusual_pattern_score) + (0.2 * repetition_score)
    return total_score

def analyze_response_length(results: list[Dict]) -> dict:
    """
    Analyze the length of responses.
    
    Args:
        results (list[Dict]): List of evaluation results
    
    Returns:
        dict: Statistics about response lengths
    """
    lengths = [len(r['response']) for r in results if 'response' in r]
    
    if not lengths:
        return {
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "median_length": 0
        }
    
    return {
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2]
    }

def generate_comparison_report(results_by_type: Dict[str, list]) -> str:
    """
    Generate a detailed comparison report.
    
    Args:
        results_by_type (Dict[str, list]): Dictionary mapping prompt types to results
    
    Returns:
        str: Formatted report text
    """
    # Analyze basic stats for each prompt type
    summary_stats = {}
    length_stats = {}
    quality_stats = {}
    gibberish_stats = {}
    
    for prompt_type, results in results_by_type.items():
        summary_stats[prompt_type] = analyze_results(results, prompt_type)
        length_stats[prompt_type] = analyze_response_length(results)
        
        # Calculate reasoning quality and gibberish estimates
        qualities = [estimate_reasoning_quality(r['response']) for r in results if 'response' in r]
        gibberish_levels = [estimate_gibberish_level(r['response']) for r in results if 'response' in r]
        
        quality_stats[prompt_type] = sum(qualities) / len(qualities) if qualities else 0
        gibberish_stats[prompt_type] = sum(gibberish_levels) / len(gibberish_levels) if gibberish_levels else 0
    
    # Build the report
    report = "# Chain of Thought (CoT) Experiment Report\n\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Experiment Overview\n\n"
    report += "This experiment tests whether Chain of Thought (CoT) reasoning provides value through coherent reasoning or simply by giving models more computation time/tokens to produce answers.\n\n"
    
    report += "Three different prompting strategies were compared:\n\n"
    report += "1. **Standard**: Direct answer with no reasoning steps\n"
    report += "2. **Chain of Thought (CoT)**: Structured reasoning before the answer\n"
    report += "3. **Gibberish CoT**: Meaningless text of similar length to CoT before the answer\n\n"
    
    report += "## Accuracy Results\n\n"
    report += "| Prompt Type | Problems | Correct | Accuracy |\n"
    report += "|-------------|----------|---------|----------|\n"
    
    for prompt_type, stats in summary_stats.items():
        report += f"| {prompt_type.capitalize()} | {stats['total']} | {stats['correct']} | {stats['accuracy']:.2%} |\n"
    
    report += "\n## Response Characteristics\n\n"
    report += "| Prompt Type | Avg Length | Reasoning Quality | Gibberish Level |\n"
    report += "|-------------|------------|-------------------|----------------|\n"
    
    for prompt_type in summary_stats.keys():
        report += (f"| {prompt_type.capitalize()} | {length_stats[prompt_type]['avg_length']:.1f} | "
                  f"{quality_stats[prompt_type]:.2f} | {gibberish_stats[prompt_type]:.2f} |\n")
    
    report += "\n## Analysis\n\n"
    
    # Add comparative analysis
    std_acc = summary_stats.get('standard', {}).get('accuracy', 0)
    cot_acc = summary_stats.get('cot', {}).get('accuracy', 0)
    gib_acc = summary_stats.get('gibberish', {}).get('accuracy', 0)
    
    # Compare CoT vs Standard
    cot_vs_std = (cot_acc - std_acc) / std_acc * 100 if std_acc > 0 else 0
    report += f"- CoT improved accuracy by {cot_vs_std:.1f}% compared to the standard prompt.\n"
    
    # Compare Gibberish vs Standard
    gib_vs_std = (gib_acc - std_acc) / std_acc * 100 if std_acc > 0 else 0
    report += f"- Gibberish CoT improved accuracy by {gib_vs_std:.1f}% compared to the standard prompt.\n"
    
    # Compare CoT vs Gibberish
    if gib_acc > 0:
        cot_vs_gib = (cot_acc - gib_acc) / gib_acc * 100
        report += f"- CoT was {cot_vs_gib:.1f}% more accurate than Gibberish CoT.\n\n"
    
    # Conclusions based on the results
    report += "## Conclusions\n\n"
    
    if cot_acc > gib_acc > std_acc:
        report += ("The results suggest that both structured reasoning (CoT) and additional computation time "
                  "(Gibberish CoT) improve performance compared to direct answers. However, since CoT outperforms "
                  "Gibberish CoT, there appears to be value in the structured reasoning process beyond just "
                  "the additional computation time.\n\n")
    elif gib_acc >= cot_acc > std_acc:
        report += ("The results suggest that the primary benefit of Chain of Thought may indeed be the additional "
                  "computation time rather than the structured reasoning itself, as Gibberish CoT performed "
                  "similarly to or better than standard CoT. This supports the hypothesis that CoT is primarily "
                  "providing value through extended computation.\n\n")
    elif cot_acc > std_acc >= gib_acc:
        report += ("The results strongly suggest that structured reasoning is crucial for improved performance. "
                  "Since Gibberish CoT did not improve over the standard prompt while CoT did, the benefit "
                  "appears to come from the reasoning process itself rather than just extra computation time.\n\n")
    else:
        report += ("The results show an interesting pattern that warrants further investigation. The relationship "
                  "between reasoning structure and performance appears more complex than initially hypothesized.\n\n")
    
    report += "## Future Work\n\n"
    report += ("- Expand testing to different problem domains beyond mathematics\n"
              "- Test with different model architectures to see if the pattern holds\n"
              "- Analyze intermediate activation patterns during reasoning vs gibberish generation\n"
              "- Investigate whether fine-tuning on gibberish CoT would yield similar benefits to CoT fine-tuning\n")
    
    return report

def create_comparison_plots(results_by_type: Dict[str, list], model_name: str) -> str:
    """
    Create visualization plots comparing the different prompt types.
    
    Args:
        results_by_type (Dict[str, list]): Dictionary mapping prompt types to results
        model_name (str): Name of the model used
        
    Returns:
        str: Filename of the saved plot
    """
    # Calculate accuracy for each prompt type
    accuracies = {}
    for prompt_type, results in results_by_type.items():
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracies[prompt_type] = (correct / total) if total > 0 else 0
    
    # Create the accuracy bar chart
    plt.figure(figsize=(10, 6))
    
    # Bar colors
    colors = {'standard': '#3498db', 'cot': '#2ecc71', 'gibberish': '#e74c3c'}
    
    # Plot bars
    bars = plt.bar(accuracies.keys(), [acc * 100 for acc in accuracies.values()], color=[colors.get(k, 'gray') for k in accuracies.keys()])
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add titles and labels
    plt.title(f'Accuracy Comparison of Different Prompting Strategies\nModel: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, max([acc * 100 for acc in accuracies.values()]) + 10)  # Add some margin at the top
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Custom x-axis labels
    x_labels = {'standard': 'Standard\n(Direct Answer)', 'cot': 'Chain of Thought\n(Reasoning)', 'gibberish': 'Gibberish CoT\n(Nonsense Text)'}
    plt.xticks(range(len(accuracies)), [x_labels.get(k, k) for k in accuracies.keys()], fontsize=10)
    
    # Save the plot
    plot_filename = f"cot_experiment_results_{model_name.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    return plot_filename

def evaluate_with_prompt_type(model: str, dataset, prompt_type: str, results_file: str, limit: int = None):
    """
    Evaluate model performance using a specific prompt type.
    
    Args:
        model (str): The model identifier
        dataset: The dataset to evaluate on
        prompt_type (str): Type of prompt to use
        results_file (str): File to save results to
        limit (int, optional): Maximum number of problems to evaluate
    
    Returns:
        list: List of evaluation results
    """
    existing_results = load_existing_results(results_file)
    
    # Create a set of already processed indexes for efficient lookup
    processed_indexes = {result['index'] for result in existing_results}
    
    # Instead of slicing the dataset, use a counter
    processed_count = 0
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating with {prompt_type}")):
        # Skip if this index has already been processed
        if idx in processed_indexes:
            continue
        
        # Break if we've hit the limit
        if limit is not None and processed_count >= limit:
            break
            
        problem_text = item['problem']
        correct_answer = item['answer']
        
        # Get model's response with the specified prompt type
        response = get_llm_response(problem_text, model, prompt_type)
        predicted_answer = extract_answer(response)

        # Compare answers
        is_correct = compare_answers(correct_answer, predicted_answer)
        
        result = {
            "index": idx,
            "problem": problem_text,
            "prompt_type": prompt_type,
            "response": response,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        }
        save_result(results_file, result)
        
        # Increment our counter
        processed_count += 1
    
    # Load all results for this prompt type
    all_results = load_existing_results(results_file)
    prompt_results = [r for r in all_results if r.get('prompt_type') == prompt_type]
    
    return prompt_results

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on MATH-500 problems with different prompting strategies")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--compare", action="store_true", help="Compare all three prompting strategies")
    parser.add_argument("--prompt", type=str, choices=["standard", "cot", "gibberish"], default="cot", 
                        help="Type of prompt to use if not comparing all")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of problems to evaluate")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('math500_debug.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    dataset = load_math500_dataset()
    
    if args.compare:
        print("Running comparative evaluation of all three prompting strategies...")
        
        # Results for each prompt type
        results_by_type = {}
        
        # Process each prompt type
        for prompt_type in ["standard", "cot", "gibberish"]:
            results_file = f"results/evaluation_results_math500_{args.model.replace('/', '_')}_{prompt_type}.json"
            prompt_results = evaluate_with_prompt_type(args.model, dataset, prompt_type, results_file, args.limit)
            results_by_type[prompt_type] = prompt_results
            
            # Print individual summary
            analyze_results(prompt_results, prompt_type)
        
        # Generate comprehensive comparison report
        report = generate_comparison_report(results_by_type)
        report_filename = f"cot_experiment_report_{args.model.replace('/', '_')}.md"
        
        with open(report_filename, "w") as f:
            f.write(report)
        
        # Generate visualization
        plot_filename = create_comparison_plots(results_by_type, args.model)
        
        print(f"\nComparison complete!")
        print(f"Report saved to: {report_filename}")
        print(f"Visualization saved to: {plot_filename}")
    else:
        # Single prompt type evaluation
        results_file = f"results/evaluation_results_math500_{args.model.replace('/', '_')}_{args.prompt}.json"
        prompt_results = evaluate_with_prompt_type(args.model, dataset, args.prompt, results_file, args.limit)
        
        # Analyze and print results
        analyze_results(prompt_results, args.prompt)

if __name__ == "__main__":
    main()