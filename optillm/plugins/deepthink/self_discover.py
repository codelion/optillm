"""
SELF-DISCOVER Framework Implementation

This module implements the SELF-DISCOVER framework for automatically discovering
task-intrinsic reasoning structures.
"""

import json
import logging
import re
from typing import List, Dict, Any, Tuple
from .reasoning_modules import get_all_modules, get_module_descriptions

logger = logging.getLogger(__name__)

class SelfDiscover:
    """
    Implementation of the SELF-DISCOVER framework.
    
    The framework operates in two stages:
    1. Stage 1: Discover task-specific reasoning structure (SELECT, ADAPT, IMPLEMENT)
    2. Stage 2: Use discovered structure to solve problem instances
    """
    
    def __init__(self, client, model: str, max_tokens: int = 16382):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_modules = get_all_modules()
        self.completion_tokens = 0
        
    def discover_reasoning_structure(self, task_description: str, task_examples: List[str] = None) -> Dict[str, Any]:
        """
        Stage 1: Discover reasoning structure for the given task.
        
        Args:
            task_description: Description of the task type
            task_examples: Optional examples of the task (without labels)
            
        Returns:
            Dict containing the discovered reasoning structure
        """
        logger.info("Starting SELF-DISCOVER reasoning structure discovery")
        
        # Step 1: SELECT relevant reasoning modules
        selected_modules = self._select_modules(task_description, task_examples)
        logger.info(f"Selected {len(selected_modules)} reasoning modules")
        
        # Step 2: ADAPT modules to be task-specific
        adapted_modules = self._adapt_modules(selected_modules, task_description, task_examples)
        logger.info("Adapted modules to be task-specific")
        
        # Step 3: IMPLEMENT structured reasoning plan
        reasoning_structure = self._implement_structure(adapted_modules, task_description, task_examples)
        logger.info("Implemented reasoning structure")
        
        return {
            "selected_modules": selected_modules,
            "adapted_modules": adapted_modules,
            "reasoning_structure": reasoning_structure,
            "completion_tokens": self.completion_tokens
        }
    
    def _select_modules(self, task_description: str, task_examples: List[str] = None) -> List[Dict[str, Any]]:
        """SELECT: Choose relevant reasoning modules for the task."""
        
        module_descriptions = get_module_descriptions()
        modules_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(module_descriptions)])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        select_prompt = f"""You are an expert in problem-solving and reasoning. Given a task description and available reasoning modules, select the most relevant modules that would be useful for solving this type of task.

Task description: {task_description}{examples_text}

Available reasoning modules:
{modules_text}

Instructions:
1. Analyze the task and identify what types of reasoning would be most helpful
2. Select 3-7 reasoning modules that are most relevant for this task
3. Consider both the complexity of the task and the complementary nature of different modules
4. Avoid selecting too many similar modules

Respond with a JSON list containing the numbers of the selected modules. For example: [1, 5, 9, 15, 23]

Selected modules:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": select_prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        
        self.completion_tokens += response.usage.completion_tokens
        
        try:
            # Extract JSON from response
            response_text = response.choices[0].message.content.strip()
            # Look for JSON array in the response
            json_match = re.search(r'\[[\d,\s]+\]', response_text)
            if json_match:
                selected_indices = json.loads(json_match.group(0))
            else:
                # Fallback: extract numbers from response
                numbers = re.findall(r'\b(\d+)\b', response_text)
                selected_indices = [int(n) for n in numbers[:7]]  # Limit to 7 modules
            
            # Convert to module objects (1-indexed to 0-indexed)
            selected_modules = []
            for idx in selected_indices:
                if 1 <= idx <= len(self.reasoning_modules):
                    selected_modules.append(self.reasoning_modules[idx-1])
            
            return selected_modules[:7]  # Ensure we don't exceed reasonable limit
            
        except Exception as e:
            logger.warning(f"Error parsing selected modules: {e}")
            # Fallback to first few modules
            return self.reasoning_modules[:5]
    
    def _adapt_modules(self, selected_modules: List[Dict[str, Any]], task_description: str, task_examples: List[str] = None) -> List[str]:
        """ADAPT: Rephrase modules to be more task-specific."""
        
        modules_text = "\n".join([f"- {module['description']}" for module in selected_modules])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        adapt_prompt = f"""You are an expert in adapting general reasoning strategies to specific tasks. Given the selected reasoning modules and task description, rephrase each module to be more specific and tailored to this particular type of task.

Task description: {task_description}{examples_text}

Selected reasoning modules:
{modules_text}

Instructions:
1. For each module, rephrase the description to be more specific to this task
2. Keep the core reasoning approach but make it more actionable for this specific type of problem
3. Use terminology and concepts relevant to the task domain
4. Make the adapted descriptions more concrete and specific

Provide the adapted modules as a numbered list:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": adapt_prompt}],
            max_tokens=2048,
            temperature=0.3
        )
        
        self.completion_tokens += response.usage.completion_tokens
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract adapted modules from numbered list
        adapted_modules = []
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                # Remove the number prefix
                adapted_desc = re.sub(r'^\d+\.\s*', '', line)
                adapted_modules.append(adapted_desc)
        
        return adapted_modules
    
    def _implement_structure(self, adapted_modules: List[str], task_description: str, task_examples: List[str] = None) -> Dict[str, Any]:
        """IMPLEMENT: Create a structured reasoning plan in JSON format."""
        
        modules_text = "\n".join([f"{i+1}. {module}" for i, module in enumerate(adapted_modules)])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        # Provide a demonstration of a reasoning structure
        demo_structure = """{
    "problem_analysis": "Analyze the core components and requirements",
    "approach_selection": "Choose the most appropriate solution method",
    "step_by_step_solution": {
        "step_1": "First logical step with clear reasoning",
        "step_2": "Second step building on previous results", 
        "step_3": "Continue logical progression"
    },
    "verification": "Check the solution for accuracy and completeness",
    "final_answer": "Present the final result clearly"
}"""
        
        implement_prompt = f"""You are an expert in creating structured reasoning plans. Given the adapted reasoning modules for a specific task, create a detailed JSON reasoning structure that can be followed step-by-step to solve instances of this task.

Task description: {task_description}{examples_text}

Adapted reasoning modules:
{modules_text}

Example of a reasoning structure format:
{demo_structure}

Instructions:
1. Create a JSON structure that operationalizes the adapted reasoning modules
2. The structure should be specific enough to guide step-by-step reasoning
3. Include clear field names that indicate what should be filled in each step
4. Make it actionable - each field should represent a concrete reasoning step
5. Ensure the structure flows logically from problem understanding to final answer
6. The structure should be comprehensive enough to handle the complexity of the task

Create the reasoning structure in valid JSON format:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": implement_prompt}],
            max_tokens=2048,
            temperature=0.3
        )
        
        self.completion_tokens += response.usage.completion_tokens
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        try:
            # Look for JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                structure = json.loads(json_match.group(0))
            else:
                # Fallback structure
                structure = {
                    "problem_understanding": "Analyze and understand the problem requirements",
                    "solution_approach": "Determine the best approach based on problem characteristics",
                    "step_by_step_reasoning": "Work through the problem systematically",
                    "verification": "Verify the solution is correct and complete",
                    "final_answer": "State the final answer clearly"
                }
            
            return structure
            
        except Exception as e:
            logger.warning(f"Error parsing reasoning structure: {e}")
            # Return fallback structure
            return {
                "analysis": "Analyze the problem systematically",
                "approach": "Select appropriate solution method",
                "reasoning": "Apply step-by-step logical reasoning",
                "conclusion": "Draw final conclusion with supporting evidence"
            }
    
    def solve_with_structure(self, problem: str, reasoning_structure: Dict[str, Any]) -> str:
        """
        Stage 2: Use the discovered reasoning structure to solve a specific problem.
        """
        
        structure_text = json.dumps(reasoning_structure, indent=2)
        
        solve_prompt = f"""Follow the step-by-step reasoning structure below to solve the given problem. Fill in each field with your reasoning and analysis, then provide your final answer.

Reasoning Structure:
{structure_text}

Problem to solve: {problem}

Instructions:
1. Work through each field in the reasoning structure systematically
2. Provide detailed reasoning for each step
3. Use the structure to guide your thinking process
4. Ensure your reasoning is logical and well-supported
5. Wrap your internal reasoning in <think> tags
6. Provide a clear final answer after your reasoning

<think>
[Follow the reasoning structure step by step here]
</think>

Based on my systematic analysis using the reasoning structure, the answer is:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": solve_prompt}],
            max_tokens=self.max_tokens,
            temperature=0.7
        )
        
        self.completion_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content.strip()
