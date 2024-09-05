import signal
from typing import Dict, Any
from z3 import *
import io
import re
import contextlib
import logging
import ast

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")

class Z3SolverSystem:
    def __init__(self, system_prompt: str, client, model: str, timeout: int = 30):
        self.system_prompt = system_prompt
        self.model = model
        self.client = client
        self.timeout = timeout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def process_query(self, query: str) -> str:
        try:
            analysis = self.analyze_query(query)
            # print("Analysis: "+ analysis)
            if "SOLVER_CAN_BE_APPLIED: True" not in analysis:
                return self.standard_llm_inference(query)
            
            formulation = self.extract_and_validate_expressions(analysis)
            # print("Formulation: "+ formulation)
            solver_result = self.solve_with_z3(formulation)
            # print(solver_result)
            
            return self.generate_response(query, analysis, solver_result)
        except Exception as e:
            return f"An error occurred while processing the query: {str(e)}"

    def analyze_query(self, query: str) -> str:
        analysis_prompt = f"""Analyze the given query and determine if it can be solved using Z3:

1. Identify variables, constraints, and objectives.
2. Determine the problem type (e.g., SAT, optimization).
3. Decide if Z3 is suitable.

If Z3 can be applied, provide Python code using Z3 to solve the problem.

Query: {query}

Respond with:
SOLVER_CAN_BE_APPLIED: [True/False]

SOLVER_FORMULATION:
```python
# Z3 code here
```

Analysis:
[Your step-by-step analysis]
"""
        
        analysis_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=1024,
            n=1,
            temperature=0.1
        )
        return analysis_response.choices[0].message.content

    def generate_response(self, query: str, analysis: str, solver_result: Dict[str, Any]) -> str:
        if solver_result.get("status") != "success":
            return self.standard_llm_inference(query)
        
        response_prompt = f"""Provide a clear answer to the query using the analysis and solver result:

Query: {query}

Analysis: {analysis}

Solver Result: {solver_result.get("output")}

Response:
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": response_prompt}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.1
        )
        return response.choices[0].message.content

    def standard_llm_inference(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.1
        )
        return response.choices[0].message.content

    def extract_and_validate_expressions(self, analysis: str) -> str:
        formulation = re.search(r"```python\n([\s\S]+?)```", analysis)
        if formulation:
            return formulation.group(1).strip()
        raise ValueError("No valid Z3 formulation found in the analysis.")

    def solve_with_z3(self, formulation: str, max_attempts: int = 3) -> Dict[str, Any]:
        for attempt in range(max_attempts):
            output = self.execute_solver_code(formulation)
            if "Error:" not in output:
                return {"status": "success", "output": output}
            
            error_prompt = f"""Fix the Z3 code that resulted in an error:

Code:
{formulation}

Error:
{output}

Provide corrected Z3 code:
```python
# Corrected Z3 code here
```
"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": error_prompt}
                ],
                max_tokens=1024,
                n=1,
                temperature=0.1
            )
            formulation = self.extract_and_validate_expressions(response.choices[0].message.content)

        return {"status": "failed", "output": "Failed to solve after multiple attempts."}

    def execute_solver_code(self, code: str) -> str:
        logging.info("Executing Z3 solver code")
        
        # Define a whitelist of allowed Z3 names
        z3_whitelist = set(dir(z3))
        
        # Parse the code into an AST
        try:
            parsed_ast = ast.parse(code)
        except SyntaxError as e:
            logging.error(f"Syntax error in provided code: {e}")
            return f"Error: Syntax error: {e}"

        # Check for any potentially unsafe operations
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != 'z3':
                        logging.warning(f"Unauthorized import: {alias.name}")
                        return f"Error: Unauthorized import: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module != 'z3':
                    logging.warning(f"Unauthorized import from: {node.module}")
                    return f"Error: Unauthorized import from: {node.module}"

        # Prepare a restricted global namespace
        safe_globals = {
            'z3': z3,
            'print': print,  # Allow print for output
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
                'abs': abs,
                'float': float,
                'int': int,
                'len': len,
                'max': max,
                'min': min,
                'round': round,
                'sum': sum,
            }
        }
        safe_globals.update({name: getattr(z3, name) for name in z3_whitelist})

        # Execute the code
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            try:
                exec(code, safe_globals, {})
            except Exception as e:
                logging.error(f"Execution error: {str(e)}")
                return f"Error: Execution error: {str(e)}"

        executed_output = output_buffer.getvalue()
        logging.info("Z3 solver code executed successfully")
        return executed_output