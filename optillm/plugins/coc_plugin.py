import re
import logging
from typing import Tuple, Dict, Any, List
import ast
import traceback
import math
import importlib

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
3. Track program state after each line execution
4. Return the final answer within the <output> tags

Format your response using:
```python
[Your code here]
```

And track state after each line with:
delta_state: {...}

Finally provide output as:
<output>
[Your final answer]
</output>
'''

STATE_SIMULATION_PROMPT = '''You are simulating the execution of Python code. 
Given the current program state and a line of code, return ONLY a Python dictionary representing the new state variables.
For module imports and references, return string representations.

Return ONLY primitive types (numbers, strings, lists, dicts) - no module references or complex objects.

For example:
state = {'x': 5}
code = "y = x + 3"
Return: {'y': 8}

For modules:
code = "import math"
Return: {'math': 'module:math'}
'''

def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]

def extract_output(text: str) -> str:
    """Extract content from output tags."""
    pattern = r'<output>(.*?)</output>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def extract_state_updates(text: str) -> List[Dict[str, Any]]:
    """Extract state updates from delta_state markers."""
    pattern = r'delta_state:\s*({.*?})'
    matches = re.findall(pattern, text, re.DOTALL)
    states = []
    for m in matches:
        try:
            cleaned = re.sub(r'```python\s*|\s*```', '', m)
            state = parse_state_dict(cleaned)
            states.append(state)
        except:
            logger.warning(f"Could not parse state update: {m}")
    return states

def clean_state_response(response: str) -> str:
    """Clean up LM state response to get just the dictionary."""
    # Remove any code blocks
    response = re.sub(r'```python\s*|\s*```', '', response)
    # Remove any natural language before or after the dict
    response = re.sub(r'^[^{]*', '', response)
    response = re.sub(r'[^}]*$', '', response)
    return response.strip()

def parse_state_dict(state_str: str) -> Dict[str, Any]:
    """Safely parse state dictionary, handling module references."""
    try:
        # First try direct evaluation
        return ast.literal_eval(state_str)
    except:
        # If that fails, try to parse manually
        state_dict = {}
        try:
            # Use a custom safe eval that handles module references
            # Remove brackets and split by commas
            items = state_str.strip('{}').split(',')
            for item in items:
                if ':' not in item:
                    continue
                key, value = item.split(':', 1)
                key = key.strip().strip("'").strip('"')
                value = value.strip()
                
                # Handle module references
                if 'module' in value:
                    module_name = value.split("'")[1] if "'" in value else value.split(':')[1].strip()
                    if module_name in ALLOWED_MODULES:
                        state_dict[key] = ALLOWED_MODULES[module_name]
                # Handle normal values
                else:
                    try:
                        state_dict[key] = ast.literal_eval(value)
                    except:
                        state_dict[key] = value.strip("'").strip('"')
            return state_dict
        except Exception as e:
            logger.error(f"Failed to parse state dict: {state_str}")
            logger.error(f"Error: {str(e)}")
            return {}

def execute_line(line: str, state: Dict[str, Any], client, model: str) -> Tuple[Any, Dict[str, Any]]:
    """Execute a single line of code, either with Python or LM simulation."""
    try:
        # Handle imports specially
        if line.startswith('import '):
            module_name = line.split()[1]
            if module_name in ALLOWED_MODULES:
                return None, {module_name: ALLOWED_MODULES[module_name]}
            else:
                logger.warning(f"Skipping import of unauthorized module: {module_name}")
                return None, {}

        # Try executing with Python
        local_state = state.copy()
        # Add allowed modules to local state
        for mod_name, mod in ALLOWED_MODULES.items():
            if mod_name in state:
                local_state[mod_name] = mod
                
        exec(line, globals(), local_state)
        # Extract any new/modified variables
        new_state = {k:v for k,v in local_state.items() 
                    if k not in state or state[k] != v}
        return None, new_state
        
    except Exception as e:
        # If Python execution fails, simulate with LM
        context = f"Current program state: {state}\nExecute line: {line}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STATE_SIMULATION_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.2
        )
        try:
            cleaned_response = clean_state_response(response.choices[0].message.content)
            new_state = parse_state_dict(cleaned_response)
            return response.usage.completion_tokens, new_state
        except Exception as e:
            logger.error(f"Could not parse LM state response: {response.choices[0].message.content}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Cleaned response: {cleaned_response}")
            return response.usage.completion_tokens, {}

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main Chain of Code execution function."""
    messages = [
        {"role": "system", "content": system_prompt + "\n" + CHAIN_OF_CODE_PROMPT},
        {"role": "user", "content": initial_query}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    initial_response = response.choices[0].message.content
    total_tokens = response.usage.completion_tokens

    code_blocks = extract_code_blocks(initial_response)
    if not code_blocks:
        logger.warning("No code blocks found in response")
        return initial_response, total_tokens

    final_state = {}
    code = code_blocks[0]  # Take first code block
    
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    
    for line in lines:
        if not line or line.startswith('#'):
            continue
            
        tokens, new_state = execute_line(line, final_state, client, model)
        if tokens:
            total_tokens += tokens
        final_state.update(new_state)
        logger.debug(f"Executed line: {line}")
        logger.debug(f"New state: {new_state}")

    final_answer = extract_output(initial_response)
    if not final_answer and 'answer' in final_state:
        final_answer = str(final_state['answer'])

    return final_answer, total_tokens