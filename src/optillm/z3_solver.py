from typing import Dict, Any
from z3 import *
import io
import re
import contextlib
import logging
import ast
import math
import multiprocessing
import traceback

class TimeoutException(Exception):
    pass

def prepare_safe_globals():
    safe_globals = {
        'print': print,
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
            'complex': complex,
        }
    }
    
    # Add common math functions
    safe_globals.update({
        'log': math.log,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e,
    })

    # Add complex number support
    safe_globals['I'] = complex(0, 1)
    safe_globals['Complex'] = complex

    return safe_globals

def execute_code_in_process(code: str):
    import z3
    import math
    import itertools
    from fractions import Fraction

    safe_globals = prepare_safe_globals()
    
    # Add Z3 specific functions
    z3_whitelist = set(dir(z3))
    safe_globals.update({name: getattr(z3, name) for name in z3_whitelist})

    # Ensure key Z3 components are available
    safe_globals.update({
        'z3': z3,
        'Solver': z3.Solver,
        'solver': z3.Solver,
        'Optimize': z3.Optimize,
        'sat': z3.sat,
        'unsat': z3.unsat,
        'unknown': z3.unknown,
        'Real': z3.Real,
        'Int': z3.Int,
        'Bool': z3.Bool,
        'And': z3.And,
        'Or': z3.Or,
        'Not': z3.Not,
        'Implies': z3.Implies,
        'If': z3.If,
        'Sum': z3.Sum,
        'ForAll': z3.ForAll,
        'Exists': z3.Exists,
        'model': z3.Model,
    })
    
    # Add custom functions
    def as_numerical(x):
        if z3.is_expr(x):
            if z3.is_int_value(x) or z3.is_rational_value(x):
                return float(x.as_decimal(20))
            elif z3.is_algebraic_value(x):
                return x.approx(20)
        return float(x)

    safe_globals['as_numerical'] = as_numerical

    def Mod(x, y):
        return x % y

    safe_globals['Mod'] = Mod

    def Rational(numerator, denominator=1):
        return z3.Real(str(Fraction(numerator, denominator)))

    safe_globals['Rational'] = Rational

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(code, safe_globals, {})
        except Exception:
            return ("error", traceback.format_exc())
    return ("success", output_buffer.getvalue())

class Z3SolverSystem:
    def __init__(self, system_prompt: str, client, model: str, timeout: int = 30):
        self.system_prompt = system_prompt
        self.model = model
        self.client = client
        self.timeout = timeout
        self.z3_completion_tokens = 0
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def process_query(self, query: str) -> str:
        try:
            analysis = self.analyze_query(query)
            # print("Analysis: "+ analysis)
            if "SOLVER_CAN_BE_APPLIED: True" not in analysis:
                return self.standard_llm_inference(query) , self.z3_completion_tokens
            
            formulation = self.extract_and_validate_expressions(analysis)
            # print("Formulation: "+ formulation)
            solver_result = self.solve_with_z3(formulation)
            # print(solver_result)
             
            return self.generate_response(query, analysis, solver_result), self.z3_completion_tokens
        except Exception as e:
            logging.error(f"An error occurred while processing the query with Z3, returning standard llm inference results: {str(e)}")
            return self.standard_llm_inference(query), self.z3_completion_tokens

    def analyze_query(self, query: str) -> str:
        analysis_prompt = f"""Analyze the given query and determine if it can be solved using Z3:

1. Identify variables, constraints, and objectives.
2. Determine the problem type (e.g., SAT, optimization).
3. Decide if Z3 is suitable.

If Z3 can be applied, provide Python code using Z3 to solve the problem. Make sure you define any additional methods you need for solving the problem.
The code will be executed in an environment with only Z3 available, so do not include any other libraries or modules.

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
        self.z3_completion_tokens  = analysis_response.usage.completion_tokens
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
        self.z3_completion_tokens  = response.usage.completion_tokens
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
        self.z3_completion_tokens  = response.usage.completion_tokens
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
        
            error_prompt = f"""Fix the Z3 code that resulted in an error. Follow these steps:

    1. Review the original code and the error message carefully.
    2. Analyze the error and identify its root cause.
    3. Think through the necessary changes to fix the error.
    4. Generate a corrected version of the Z3 code.

    Original Code:
    {formulation}

    Error Message:
    {output}

    Step-by-Step Analysis:
    [Provide your step-by-step analysis here]

    Corrected Z3 Code:
    ```python
    # Corrected Z3 code here
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
            self.z3_completion_tokens  = response.usage.completion_tokens
            formulation = self.extract_and_validate_expressions(response.choices[0].message.content)

        return {"status": "failed", "output": "Failed to solve after multiple attempts."}

    def execute_solver_code(self, code: str) -> str:
        logging.info("Executing Z3 solver code")
        logging.info(f"Code: {code}")
        
        # Parse the code into an AST
        try:
            _ = ast.parse(code)
        except SyntaxError as e:
            logging.error(f"Syntax error in provided code: {e}")
            return f"Error: Syntax error: {e}"

        # Execute the code in a separate process
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(1) as pool:
            async_result = pool.apply_async(execute_code_in_process, (code,))
            try:
                status, result = async_result.get(timeout=self.timeout)
            except multiprocessing.TimeoutError:
                pool.terminate()
                logging.error("Execution timed out")
                return "Error: Execution timed out"

        if status == "error":
            logging.error(f"Execution error: {result}")
            return f"Error: {result}"

        logging.info("Z3 solver code executed successfully")
        return result