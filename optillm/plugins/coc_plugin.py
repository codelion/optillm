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

# List of allowed modules for execution
ALLOWED_MODULES = {
    'math': math,
}

# Prompts
CHAIN_OF_CODE_PROMPT = '''
You are an AI assistant that uses Chain of Code (CoC) approach to solve problems. Follow these steps:

1. Write Python code that breaks down the problem into clear steps
2. Each step should either be:
   - Executable Python code that performs computations
   - Pseudocode that you will simulate with natural language understanding
3. Track final result in an 'answer' variable
4. Return the final answer within the <output> tags

Format your response using:
```python
[Your complete Python program here]
```

Finally provide output as:
<output>
[Your final answer]
</output>
'''

STATE_SIMULATION_PROMPT = '''You are simulating the execution of a Python program.
Given the code below, simulate its execution and return the final value that would be in the 'answer' variable.
Return ONLY the final value, no explanations or additional text.

Code to simulate:
{code}
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

def extract_output(text: str) -> str:
    """Extract content from output tags."""
    pattern = r'<output>(.*?)</output>'
    match = re.search(pattern, text, re.DOTALL)
    result = match.group(1).strip() if match else text.strip()
    logger.info(f"Extracted output: {result}")
    return result

def execute_code(code: str, client, model: str) -> Tuple[Any, int]:
    """Execute full code block either with Python or LM simulation."""
    logger.info("Attempting to execute complete code block")
    logger.info(f"Code:\n{code}")
    
    # Add imports
    execution_env = {}
    for mod_name, mod in ALLOWED_MODULES.items():
        execution_env[mod_name] = mod
    
    try:
        # Try executing the complete code block with Python
        logger.info("Attempting Python execution")
        exec(code, execution_env)
        answer = execution_env.get('answer')
        logger.info(f"Python execution successful. Answer: {answer}")
        return answer, 0
        
    except Exception as e:
        logger.info(f"Python execution failed: {str(e)}")
        logger.info("Falling back to LM simulation")
        
        # If Python execution fails, simulate with LM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STATE_SIMULATION_PROMPT.format(code=code)},
                {"role": "user", "content": "Simulate this code and return the final value of 'answer'."}
            ],
            temperature=0.2
        )
        
        try:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LM simulation successful. Answer: {answer}")
            
            # Try to convert to number if possible
            try:
                answer = ast.literal_eval(answer)
            except:
                pass
                
            return answer, response.usage.completion_tokens
            
        except Exception as e:
            logger.error(f"Could not parse LM simulation response: {str(e)}")
            return None, response.usage.completion_tokens

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main Chain of Code execution function."""
    logger.info("Starting Chain of Code execution")
    logger.info(f"Query: {initial_query}")
    
    messages = [
        {"role": "system", "content": system_prompt + "\n" + CHAIN_OF_CODE_PROMPT},
        {"role": "user", "content": initial_query}
    ]
    
    logger.info("Generating code solution")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    initial_response = response.choices[0].message.content
    total_tokens = response.usage.completion_tokens
    
    logger.info("Initial response from LM:")
    logger.info(initial_response)

    code_blocks = extract_code_blocks(initial_response)
    if not code_blocks:
        logger.warning("No code blocks found in response")
        return initial_response, total_tokens

    # Execute the complete code block
    code = code_blocks[0]
    answer, execution_tokens = execute_code(code, client, model)
    total_tokens += execution_tokens

    # If we got an answer from code execution, use it
    if answer is not None:
        final_answer = str(answer)
    else:
        # Fall back to output tags if code execution failed
        final_answer = extract_output(initial_response)
            
    logger.info(f"Chain of Code execution completed. Final answer: {final_answer}")
    return final_answer, total_tokens