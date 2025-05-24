"""
Atomic Reasoning Modules for SELF-DISCOVER Framework

This module contains the 39 reasoning modules as described in the SELF-DISCOVER paper.
These modules represent high-level cognitive heuristics for problem-solving.
"""

# 39 Atomic Reasoning Modules from SELF-DISCOVER paper
REASONING_MODULES = [
    {
        "id": 1,
        "name": "experimental_design",
        "description": "How could I devise an experiment to help solve that problem?"
    },
    {
        "id": 2,
        "name": "iterative_problem_solving",
        "description": "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made."
    },
    {
        "id": 3,
        "name": "progress_measurement",
        "description": "How could I measure progress on this problem?"
    },
    {
        "id": 4,
        "name": "problem_simplification",
        "description": "How can I simplify the problem so that it is easier to solve?"
    },
    {
        "id": 5,
        "name": "assumption_analysis",
        "description": "What are the key assumptions underlying this problem?"
    },
    {
        "id": 6,
        "name": "risk_assessment",
        "description": "What are the potential risks and drawbacks of each solution?"
    },
    {
        "id": 7,
        "name": "perspective_analysis",
        "description": "What are the alternative perspectives or viewpoints on this problem?"
    },
    {
        "id": 8,
        "name": "long_term_implications",
        "description": "What are the long-term implications of this problem and its solutions?"
    },
    {
        "id": 9,
        "name": "problem_decomposition",
        "description": "How can I break down this problem into smaller, more manageable parts?"
    },
    {
        "id": 10,
        "name": "critical_thinking",
        "description": "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking."
    },
    {
        "id": 11,
        "name": "creative_thinking",
        "description": "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality."
    },
    {
        "id": 12,
        "name": "collaborative_thinking",
        "description": "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions."
    },
    {
        "id": 13,
        "name": "systems_thinking",
        "description": "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focus on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole."
    },
    {
        "id": 14,
        "name": "risk_analysis",
        "description": "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits."
    },
    {
        "id": 15,
        "name": "reflective_thinking",
        "description": "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches."
    },
    {
        "id": 16,
        "name": "core_issue_identification",
        "description": "What is the core issue or problem that needs to be addressed?"
    },
    {
        "id": 17,
        "name": "causal_analysis",
        "description": "What are the underlying causes or factors contributing to the problem?"
    },
    {
        "id": 18,
        "name": "historical_analysis",
        "description": "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?"
    },
    {
        "id": 19,
        "name": "obstacle_identification",
        "description": "What are the potential obstacles or challenges that might arise in solving this problem?"
    },
    {
        "id": 20,
        "name": "data_analysis",
        "description": "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?"
    },
    {
        "id": 21,
        "name": "stakeholder_analysis",
        "description": "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?"
    },
    {
        "id": 22,
        "name": "resource_analysis",
        "description": "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?"
    },
    {
        "id": 23,
        "name": "success_metrics",
        "description": "How can progress or success in solving the problem be measured or evaluated?"
    },
    {
        "id": 24,
        "name": "metric_identification",
        "description": "What indicators or metrics can be used?"
    },
    {
        "id": 25,
        "name": "problem_type_technical",
        "description": "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?"
    },
    {
        "id": 26,
        "name": "physical_constraints",
        "description": "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?"
    },
    {
        "id": 27,
        "name": "behavioral_aspects",
        "description": "Is the problem related to human behavior, such as a social, cultural, or psychological issue?"
    },
    {
        "id": 28,
        "name": "decision_making",
        "description": "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?"
    },
    {
        "id": 29,
        "name": "analytical_problem",
        "description": "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?"
    },
    {
        "id": 30,
        "name": "design_challenge",
        "description": "Is the problem a design challenge that requires creative solutions and innovation?"
    },
    {
        "id": 31,
        "name": "systemic_issues",
        "description": "Does the problem require addressing systemic or structural issues rather than just individual instances?"
    },
    {
        "id": 32,
        "name": "time_sensitivity",
        "description": "Is the problem time-sensitive or urgent, requiring immediate attention and action?"
    },
    {
        "id": 33,
        "name": "typical_solutions",
        "description": "What kinds of solution typically are produced for this kind of problem specification?"
    },
    {
        "id": 34,
        "name": "alternative_solutions",
        "description": "Given the problem specification and the current best solution, have a guess about other possible solutions."
    },
    {
        "id": 35,
        "name": "radical_rethinking",
        "description": "Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    },
    {
        "id": 36,
        "name": "solution_modification",
        "description": "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    },
    {
        "id": 37,
        "name": "novel_solution",
        "description": "Ignoring the current best solution, create an entirely new solution to the problem."
    },
    {
        "id": 38,
        "name": "step_by_step",
        "description": "Let's think step by step."
    },
    {
        "id": 39,
        "name": "step_by_step_plan",
        "description": "Let's make a step by step plan and implement it with good notion and explanation."
    }
]

def get_all_modules():
    """Return all 39 reasoning modules."""
    return REASONING_MODULES

def get_modules_by_category():
    """Categorize modules by their primary focus."""
    categories = {
        "analytical": [1, 3, 5, 10, 14, 17, 20, 23, 24, 25, 29],
        "creative": [2, 4, 11, 30, 34, 35, 37],
        "systematic": [9, 13, 16, 18, 22, 31, 33, 36, 38, 39],
        "collaborative": [7, 12, 15, 21],
        "risk_oriented": [6, 8, 14, 19],
        "behavioral": [27, 28],
        "constraint_focused": [26, 32]
    }
    
    return {
        category: [REASONING_MODULES[i-1] for i in indices]
        for category, indices in categories.items()
    }

def get_modules_by_ids(module_ids):
    """Get specific modules by their IDs."""
    return [module for module in REASONING_MODULES if module["id"] in module_ids]

def get_module_descriptions():
    """Get just the descriptions for prompting."""
    return [f"{module['name']}: {module['description']}" for module in REASONING_MODULES]
