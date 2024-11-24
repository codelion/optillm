import re
import logging
from typing import Tuple, Dict, Any, List
import ast
import traceback
import math
import importlib
import json

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "coc"

# Maximum attempts to fix code
MAX_FIX_ATTEMPTS = 3

# List of allowed modules for execution
ALLOWED_MODULES = {
    'math': math,
}

# Initial code generation prompt
CHAIN_OF_CODE_PROMPT = '''
Write Python code to solve this problem. The code should:
1. Break down the problem into clear computational steps
2. Use standard Python features and math operations
3. Store the final result in a variable named 'answer'
4. Include error handling where appropriate
5. Be complete and executable

Format your response using:
```python
[Your complete Python program here]
```
'''

# Code fix prompt
CODE_FIX_PROMPT = '''
The following Python code failed to execute. Fix the code to make it work.
Original code:
```python
{code}
```

Error encountered:
{error}

Please provide a complete, fixed version of the code that:
1. Addresses the error message
2. Maintains the same logic and approach
3. Stores the final result in 'answer'
4. Is complete and executable

Return only the fixed code in a code block:
```python
[Your fixed code here]
```
'''

# Simulation prompt
SIMULATION_PROMPT = '''
The following Python code could not be executed after several attempts. 
Please simulate its execution and determine the final value that would be in the 'answer' variable.

Code to simulate:
```python
{code}
```

Last error encountered:
{error}

Important:
1. Follow the logic of the code exactly
2. Perform all calculations carefully
3. Return ONLY the final numeric or string value, no explanations
4. If the code contains semantic functions (like text analysis), use your judgment to simulate them
'''

def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    blocks = [m.strip() for m in matches]
    logger.info(f"Extracted {len(blocks)} code blocks")
    for i, block in enumerate(blocks):
        logger.info(f"Code block {i+1}:\n{block}")
    return blocks

def sanitize_code(code: str) -> str:
    """Prepare code for execution by adding necessary imports and safety checks."""
    # Add standard imports
    imports = "\n".join(f"import {mod}" for mod in ALLOWED_MODULES)
    
    # Add safety wrapper
    wrapper = f"""
{imports}

def safe_execute():
    {code.replace('\n', '\n    ')}
    return answer if 'answer' in locals() else None

result = safe_execute()
answer = result
"""
    return wrapper

def execute_code(code: str) -> Tuple[Any, str]:
    """Attempt to execute the code and return result or error."""
    logger.info("Attempting to execute code")
    logger.info(f"Code:\n{code}")
    
    execution_env = {}
    try:
        sanitized_code = sanitize_code(code)
        exec(sanitized_code, execution_env)
        answer = execution_env.get('answer')
        
        if answer is not None:
            logger.info(f"Execution successful. Answer: {answer}")
            return answer, None
        else:
            error = "Code executed but did not produce an answer"
            logger.warning(error)
            return None, error
            
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Execution failed: {error}")
        return None, error

def generate_fixed_code(original_code: str, error: str, client, model: str) -> Tuple[str, int]:
    """Ask LLM to fix the broken code."""
    logger.info("Requesting code fix from LLM")
    logger.info(f"Original error: {error}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CODE_FIX_PROMPT.format(
                code=original_code, error=error)},
            {"role": "user", "content": "Fix the code to make it work."}
        ],
        temperature=0.2
    )
    
    fixed_code = response.choices[0].message.content
    code_blocks = extract_code_blocks(fixed_code)
    
    if code_blocks:
        logger.info("Received fixed code from LLM")
        return code_blocks[0], response.usage.completion_tokens
    else:
        logger.warning("No code block found in LLM response")
        return None, response.usage.completion_tokens

def simulate_execution(code: str, error: str, client, model: str) -> Tuple[Any, int]:
    """Ask LLM to simulate code execution."""
    logger.info("Attempting code simulation with LLM")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SIMULATION_PROMPT.format(
                code=code, error=error)},
            {"role": "user", "content": "Simulate this code and return the final answer value."}
        ],
        temperature=0.2
    )
    
    try:
        result = response.choices[0].message.content.strip()
        # Try to convert to appropriate type
        try:
            answer = ast.literal_eval(result)
        except:
            answer = result
        logger.info(f"Simulation successful. Result: {answer}")
        return answer, response.usage.completion_tokens
    except Exception as e:
        logger.error(f"Failed to parse simulation result: {str(e)}")
        return None, response.usage.completion_tokens

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main Chain of Code execution function."""
    logger.info("Starting Chain of Code execution")
    logger.info(f"Query: {initial_query}")
    
    # Initial code generation
    messages = [
        {"role": "system", "content": system_prompt + "\n" + CHAIN_OF_CODE_PROMPT},
        {"role": "user", "content": initial_query}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    total_tokens = response.usage.completion_tokens
    
    # Extract initial code
    code_blocks = extract_code_blocks(response.choices[0].message.content)
    if not code_blocks:
        logger.warning("No code blocks found in response")
        return response.choices[0].message.content, total_tokens

    current_code = code_blocks[0]
    fix_attempts = 0
    last_error = None
    
    # Strategy 1: Direct execution and fix attempts
    while fix_attempts < MAX_FIX_ATTEMPTS:
        fix_attempts += 1
        logger.info(f"Execution attempt {fix_attempts}/{MAX_FIX_ATTEMPTS}")
        
        # Try to execute current code
        answer, error = execute_code(current_code)
        
        # If successful, return the answer
        if error is None:
            logger.info(f"Successful execution on attempt {fix_attempts}")
            return str(answer), total_tokens
            
        last_error = error
        
        # If we hit max attempts, break to try simulation
        if fix_attempts >= MAX_FIX_ATTEMPTS:
            logger.warning(f"Failed after {fix_attempts} fix attempts")
            break
            
        # Otherwise, try to get fixed code from LLM
        logger.info(f"Requesting code fix, attempt {fix_attempts}")
        fixed_code, fix_tokens = generate_fixed_code(current_code, error, client, model)
        total_tokens += fix_tokens
        
        if fixed_code:
            current_code = fixed_code
        else:
            logger.error("Failed to get fixed code from LLM")
            break
    
    # Strategy 2: If all execution attempts failed, try simulation
    logger.info("All execution attempts failed, trying simulation")
    simulated_answer, sim_tokens = simulate_execution(current_code, last_error, client, model)
    total_tokens += sim_tokens
    
    if simulated_answer is not None:
        logger.info("Successfully got answer from simulation")
        return str(simulated_answer), total_tokens
    
    # If we get here, everything failed
    logger.warning("All strategies failed")
    return f"Error: Could not solve problem after all attempts. Last error: {last_error}", total_tokens