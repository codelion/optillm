import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class PlanSearch:
    def __init__(self, system_prompt: str, client, model: str):
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.plansearch_completion_tokens = 0

    def generate_observations(self, problem: str, num_observations: int = 3) -> List[str]:
        prompt = f"""You are an expert Python programmer. You will be given a competitive programming question
(problem specification). You will return several useful, non-obvious, and correct observations
about the problem, like hints to solve the problem. You will NOT return any code. Be as
creative as possible, going beyond what you think is intuitively correct.

Here is the competitive programming problem:
{problem}

Please provide {num_observations} observations."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.plansearch_completion_tokens += response.usage.completion_tokens
        observations = response.choices[0].message.content.strip().split('\n')
        return [obs.strip() for obs in observations if obs.strip()]

    def generate_derived_observations(self, problem: str, observations: List[str], num_new_observations: int = 2) -> List[str]:
        prompt = f"""You are an expert Python programmer. You will be given a competitive programming question
(problem specification) and several correct observations about the problem.
You will brainstorm several new, useful, and correct observations about the problem, derived
from the given observations. You will NOT return any code. Be as creative as possible, going
beyond what you think is intuitively correct.

Here is the competitive programming problem:
{problem}

Here are the existing observations:
{chr(10).join(f"{i+1}. {obs}" for i, obs in enumerate(observations))}

Please provide {num_new_observations} new observations derived from the existing ones."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.plansearch_completion_tokens += response.usage.completion_tokens
        new_observations = response.choices[0].message.content.strip().split('\n')
        return [obs.strip() for obs in new_observations if obs.strip()]

    def generate_solution(self, problem: str, observations: List[str]) -> str:
        prompt = f"""Here is the competitive programming problem:
{problem}

Here are the intelligent observations to help solve the problem:
{chr(10).join(f"Observation {i+1}: {obs}" for i, obs in enumerate(observations))}

Use these observations above to brainstorm a natural language solution to the problem above.
Note that your intuition may lead you astray, so come up with simple, creative ideas that
go beyond what you would usually come up with and exceeds your narrow intuition.
Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING
IS CRUCIAL."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.plansearch_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def implement_solution(self, problem: str, solution: str) -> str:
        prompt = f"""You are an expert Python programmer. You will be given a question (problem specification)
and a natural language solution/tutorial that describes how to solve the problem. You will
generate a correct Python program that matches said specification and tutorial and passes
all tests. You will NOT return anything except for the program inside markdown codeblocks.

Problem:
{problem}

Solution:
{solution}

Please implement the solution in Python."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.plansearch_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def solve(self, problem: str, num_initial_observations: int = 3, num_derived_observations: int = 2) -> Tuple[str, str]:
        logger.info("Generating initial observations")
        initial_observations = self.generate_observations(problem, num_initial_observations)
        
        logger.info("Generating derived observations")
        derived_observations = self.generate_derived_observations(problem, initial_observations, num_derived_observations)
        
        all_observations = initial_observations + derived_observations
        
        logger.info("Generating solution based on observations")
        natural_language_solution = self.generate_solution(problem, all_observations)
        
        logger.info("Implementing solution in Python")
        python_implementation = self.implement_solution(problem, natural_language_solution)
        
        return natural_language_solution, python_implementation

    def solve_multiple(self, problem: str, n: int, num_initial_observations: int = 3, num_derived_observations: int = 2) -> List[str]:
        solutions = []
        for _ in range(n):
            _, python_implementation = self.solve(problem, num_initial_observations, num_derived_observations)
            solutions.append(python_implementation)
        return solutions

def plansearch(system_prompt: str, initial_query: str, client, model: str, n: int = 1) -> List[str]:
    planner = PlanSearch(system_prompt, client, model)
    return planner.solve_multiple(initial_query, n), planner.plansearch_completion_tokens
