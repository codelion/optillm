"""
Configuration settings for the System Prompt Learning (SPL) plugin.
"""

import os
from typing import List

# Plugin identifier
SLUG = "spl"

# Base directory for storing strategy data
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PLUGIN_DIR, 'data')
STRATEGY_DB_PATH = os.path.join(DATA_DIR, 'strategies.json')
STRATEGY_METRICS_PATH = os.path.join(DATA_DIR, 'metrics.json')

# Default max tokens for reasoning LLMs
DEFAULT_MAX_TOKENS = 4096

# How often to perform maintenance operations (merge, prune)
MAINTENANCE_INTERVAL = 40

# Strategy selection thresholds
STRATEGY_CREATION_THRESHOLD = 0.7  # Higher threshold to avoid creating similar strategies
STRATEGY_MERGING_THRESHOLD = 0.6   # Lower threshold to merge more similar strategies
MIN_SUCCESS_RATE_FOR_INFERENCE = 0.4  # Minimum success rate for a strategy to be used during inference

# Limits for strategy management
MAX_STRATEGIES_PER_TYPE = 10  # Maximum strategies to store in DB per problem type
MAX_STRATEGIES_FOR_INFERENCE = 3  # Maximum strategies to use during inference

# Define valid problem types (used for strict classification)
VALID_PROBLEM_TYPES: List[str] = [
    "arithmetic_calculation",
    "algebraic_equation",
    "statistical_analysis",
    "logical_reasoning",
    "word_problem",
    "coding_problem",
    "algorithm_design",
    "creative_writing",
    "text_summarization",
    "information_retrieval",
    "planning_task",
    "decision_making",
    "knowledge_question",
    "language_translation",
    "sequence_completion",
    "general_problem"  # Fallback type
]

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
