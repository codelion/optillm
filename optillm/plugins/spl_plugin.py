"""
System Prompt Learning (SPL) Plugin for OptiLLM

This plugin implements Andrej Karpathy's proposed system prompt learning paradigm,
allowing LLMs to improve their problem-solving capabilities by:
1. Identifying problem types
2. Generating and refining strategies for solving different problems
3. Building a knowledge base of problem-solving techniques
4. Applying these techniques to new instances of similar problems
5. Tracking the success of different strategies to prioritize effective ones

The plugin maintains a database of strategies that evolves over time, making the
LLM incrementally better at solving problems by learning from its experiences.
"""

import json
import os
import re
import time
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Plugin identifier
SLUG = "spl"

# Setup logging
logger = logging.getLogger(__name__)

# Base directory for storing strategy data
STRATEGY_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spl_data')
STRATEGY_DB_PATH = os.path.join(STRATEGY_DB_DIR, 'strategies.json')
STRATEGY_METRICS_PATH = os.path.join(STRATEGY_DB_DIR, 'metrics.json')

# Default max tokens for reasoning LLMs
DEFAULT_MAX_TOKENS = 4096

# How often to perform maintenance operations (merge, prune)
MAINTENANCE_INTERVAL = 40

# Similarity thresholds
STRATEGY_CREATION_THRESHOLD = 0.7  # Higher threshold to avoid creating similar strategies
STRATEGY_MERGING_THRESHOLD = 0.6   # Lower threshold to merge more similar strategies

# Maximum strategies per problem type
MAX_STRATEGIES_PER_TYPE = 10

# Ensure data directory exists
os.makedirs(STRATEGY_DB_DIR, exist_ok=True)

# Define valid problem types (used for strict classification)
VALID_PROBLEM_TYPES = [
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

class Strategy:
    """Represents a problem-solving strategy learned by the system."""
    
    def __init__(
        self, 
        strategy_id: str, 
        problem_type: str, 
        strategy_text: str, 
        examples: List[str] = None, 
        success_count: int = 0, 
        total_attempts: int = 0,
        created_at: str = None,
        last_used: str = None,
        last_updated: str = None,
        confidence: float = 0.5,
        tags: List[str] = None,
        reasoning_examples: List[str] = None  # Store reasoning examples
    ):
        self.strategy_id = strategy_id
        # Ensure problem_type is always a valid type
        self.problem_type = problem_type if problem_type in VALID_PROBLEM_TYPES else "general_problem"
        self.strategy_text = strategy_text
        self.examples = examples or []
        self.success_count = success_count
        self.total_attempts = total_attempts
        self.created_at = created_at or datetime.now().isoformat()
        self.last_used = last_used
        self.last_updated = last_updated or self.created_at
        self.confidence = confidence
        self.tags = tags or []
        self.reasoning_examples = reasoning_examples or []
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this strategy."""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the strategy to a dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "problem_type": self.problem_type,
            "strategy_text": self.strategy_text,
            "examples": self.examples,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "last_updated": self.last_updated,
            "confidence": self.confidence,
            "tags": self.tags,
            "reasoning_examples": self.reasoning_examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create a Strategy instance from a dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            problem_type=data["problem_type"],
            strategy_text=data["strategy_text"],
            examples=data.get("examples", []),
            success_count=data.get("success_count", 0),
            total_attempts=data.get("total_attempts", 0),
            created_at=data.get("created_at"),
            last_used=data.get("last_used"),
            last_updated=data.get("last_updated"),
            confidence=data.get("confidence", 0.5),
            tags=data.get("tags", []),
            reasoning_examples=data.get("reasoning_examples", [])
        )
    
    def record_attempt(self, success: bool) -> None:
        """Record an attempt to use this strategy."""
        self.total_attempts += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now().isoformat()
        # Update confidence using a smoothed estimate
        alpha = 0.1  # Learning rate
        self.confidence = (1 - alpha) * self.confidence + alpha * (1.0 if success else 0.0)
    
    def update_strategy(self, new_strategy_text: str) -> None:
        """Update the strategy text with a refined version."""
        self.strategy_text = new_strategy_text
        self.last_updated = datetime.now().isoformat()
    
    def add_reasoning_example(self, reasoning: str) -> None:
        """Add a reasoning example to the strategy."""
        if reasoning and reasoning.strip():
            # Keep only a reasonable number of examples (max 5)
            if len(self.reasoning_examples) >= 5:
                self.reasoning_examples.pop(0)  # Remove oldest example
            self.reasoning_examples.append(reasoning.strip())
    
    def add_example(self, example: str) -> None:
        """Add an example to the strategy."""
        if example and example not in self.examples:
            self.examples.append(example)

class StrategyDatabase:
    """Manages a collection of problem-solving strategies."""
    
    def __init__(self, db_path: str = STRATEGY_DB_PATH, metrics_path: str = STRATEGY_METRICS_PATH):
        self.db_path = db_path
        self.metrics_path = metrics_path
        self.strategies: List[Strategy] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectors = None
        self.metrics = {
            "total_queries": 0,
            "strategy_applications": 0,
            "strategies_created": 0,
            "strategies_refined": 0,
            "successful_resolutions": 0,
            "last_strategy_id": 0,
            "reasoning_examples_collected": 0,
            "strategies_merged": 0
        }
        self._load()
    
    def _load(self) -> None:
        """Load strategies and metrics from disk."""
        # Load strategies
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.strategies = [Strategy.from_dict(s) for s in data]
                    
                    # Update last_strategy_id based on loaded strategies
                    for strategy in self.strategies:
                        # Extract the numeric part from strategy_id (e.g., "strategy_42" -> 42)
                        if strategy.strategy_id.startswith("strategy_"):
                            try:
                                strategy_num = int(strategy.strategy_id.split("_")[1])
                                self.metrics["last_strategy_id"] = max(
                                    self.metrics["last_strategy_id"], 
                                    strategy_num
                                )
                            except ValueError:
                                # Skip if the ID isn't in the expected format
                                pass
                            
                logger.info(f"Loaded {len(self.strategies)} strategies from {self.db_path}")
                logger.info(f"Last strategy ID is {self.metrics['last_strategy_id']}")
            except Exception as e:
                logger.error(f"Error loading strategies: {str(e)}")
                self.strategies = []
        
        # Load metrics
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                    # Update metrics but keep the last_strategy_id we calculated from the strategies
                    last_id = self.metrics["last_strategy_id"]
                    self.metrics.update(metrics)
                    
                    # Use the larger of the two values to be safe
                    if "last_strategy_id" in metrics:
                        self.metrics["last_strategy_id"] = max(last_id, metrics["last_strategy_id"])
                    
                logger.info(f"Loaded metrics from {self.metrics_path}")
                logger.info(f"Last strategy ID is {self.metrics['last_strategy_id']}")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save(self) -> None:
        """Save strategies and metrics to disk."""
        # Save strategies
        try:
            with open(self.db_path, 'w') as f:
                json.dump([s.to_dict() for s in self.strategies], f, indent=2)
            logger.info(f"Saved {len(self.strategies)} strategies to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving strategies: {str(e)}")
        
        # Save metrics
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Saved metrics to {self.metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the database."""
        # Extract the ID number from the strategy_id
        if strategy.strategy_id.startswith("strategy_"):
            try:
                strategy_num = int(strategy.strategy_id.split("_")[1])
                # Update last_strategy_id if this is a higher number
                self.metrics["last_strategy_id"] = max(self.metrics["last_strategy_id"], strategy_num)
            except ValueError:
                # If the format is unexpected, don't update the counter
                pass
                
        # First, check if this is a new problem type that we don't have any strategies for
        exists = any(s.problem_type == strategy.problem_type for s in self.strategies)
        
        # Add strategy to database
        self.strategies.append(strategy)
        self.vectors = None  # Invalidate vector cache
        self.metrics["strategies_created"] += 1
        self._save()
        
        # Log whether this is a new problem type
        if not exists:
            logger.info(f"Added first strategy for problem type: {strategy.problem_type}")
        else:
            logger.info(f"Added additional strategy for problem type: {strategy.problem_type}")
    
    def get_strategies_for_problem(self, problem_type: str) -> List[Strategy]:
        """Get all strategies for a specific problem type."""
        return [s for s in self.strategies if s.problem_type == problem_type]
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by its ID."""
        for strategy in self.strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None
    
    def update_strategy_performance(self, strategy_id: str, success: bool) -> None:
        """Update the performance metrics for a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            strategy.record_attempt(success)
            self.metrics["strategy_applications"] += 1
            if success:
                self.metrics["successful_resolutions"] += 1
            self._save()
    
    def refine_strategy(self, strategy_id: str, refined_text: str) -> None:
        """Refine a strategy based on new insights."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            strategy.update_strategy(refined_text)
            self.metrics["strategies_refined"] += 1
            self._save()
    
    def add_reasoning_example(self, strategy_id: str, reasoning: str) -> None:
        """Add a reasoning example to a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy and reasoning:
            strategy.add_reasoning_example(reasoning)
            self.metrics["reasoning_examples_collected"] += 1
            self._save()
    
    def add_example_to_strategy(self, strategy_id: str, example: str) -> None:
        """Add an example to a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy and example:
            strategy.add_example(example)
            self._save()
    
    def get_similar_strategies(self, query: str, n: int = 5) -> List[Tuple[Strategy, float]]:
        """Find strategies similar to a query using TF-IDF similarity."""
        if not self.strategies:
            return []
        
        # Extract strategy texts and update vectorizer
        strategy_texts = [s.strategy_text for s in self.strategies]
        if self.vectors is None or len(self.vectors.shape) == 0 or self.vectors.shape[0] != len(strategy_texts):
            try:
                self.vectors = self.vectorizer.fit_transform(strategy_texts)
            except Exception as e:
                logger.error(f"Error creating strategy vectors: {str(e)}")
                return []
        
        # Convert query to vector and find similarities
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top strategies with their similarity scores
            sorted_indices = similarities.argsort()[::-1]
            return [(self.strategies[i], float(similarities[i])) for i in sorted_indices[:n]]
        except Exception as e:
            logger.error(f"Error finding similar strategies: {str(e)}")
            return []
    
    def find_similar_strategy(self, problem_type: str, query: str, threshold: float = STRATEGY_CREATION_THRESHOLD) -> Optional[Tuple[Strategy, float]]:
        """
        Find a strategy of the same problem type that is similar to the query.
        
        Args:
            problem_type: The problem type to match
            query: The query to find similarity against
            threshold: The similarity threshold to consider a match
            
        Returns:
            Optional[Tuple[Strategy, float]]: The most similar strategy and its similarity score,
                                             or None if no similar strategy is found
        """
        if not self.strategies:
            return None
        
        # Get strategies of the specified problem type
        type_strategies = [s for s in self.strategies if s.problem_type == problem_type]
        if not type_strategies:
            return None
        
        try:
            # Vectorize strategy texts
            strategy_texts = [s.strategy_text for s in type_strategies]
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(strategy_texts + [query])
            
            # Calculate similarities
            query_vector = vectors[-1]
            strategy_vectors = vectors[:-1]
            similarities = cosine_similarity(query_vector, strategy_vectors).flatten()
            
            # Find the most similar strategy
            if len(similarities) > 0:
                max_idx = similarities.argmax()
                max_similarity = similarities[max_idx]
                
                if max_similarity >= threshold:
                    return (type_strategies[max_idx], float(max_similarity))
        
        except Exception as e:
            logger.error(f"Error finding similar strategy: {str(e)}")
        
        return None
    
    def find_similar_examples(self, problem_type: str, query: str, threshold: float = STRATEGY_CREATION_THRESHOLD) -> Optional[Tuple[Strategy, float]]:
        """
        Find a strategy of the same problem type with examples similar to the query.
        
        Args:
            problem_type: The problem type to match
            query: The query to find similarity against
            threshold: The similarity threshold to consider a match
            
        Returns:
            Optional[Tuple[Strategy, float]]: The strategy with the most similar examples and the similarity score,
                                             or None if no similar strategy is found
        """
        if not self.strategies:
            return None
        
        # Get strategies of the specified problem type
        type_strategies = [s for s in self.strategies if s.problem_type == problem_type]
        if not type_strategies:
            return None
        
        max_similarity = 0.0
        most_similar_strategy = None
        
        try:
            for strategy in type_strategies:
                if not strategy.examples:
                    continue
                
                # Vectorize examples and query
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform(strategy.examples + [query])
                
                # Calculate similarities
                query_vector = vectors[-1]
                example_vectors = vectors[:-1]
                similarities = cosine_similarity(query_vector, example_vectors).flatten()
                
                # Get the maximum similarity for this strategy
                if len(similarities) > 0:
                    strategy_max_similarity = similarities.max()
                    
                    if strategy_max_similarity > max_similarity:
                        max_similarity = strategy_max_similarity
                        most_similar_strategy = strategy
            
            if most_similar_strategy and max_similarity >= threshold:
                return (most_similar_strategy, float(max_similarity))
                
        except Exception as e:
            logger.error(f"Error finding similar examples: {str(e)}")
        
        return None
    
    def get_next_strategy_id(self) -> str:
        """Generate a unique ID for a new strategy."""
        self.metrics["last_strategy_id"] += 1
        new_id = f"strategy_{self.metrics['last_strategy_id']}"
        logger.info(f"Generated new strategy ID: {new_id}")
        return new_id
    
    def increment_query_count(self) -> None:
        """Increment the total query count."""
        self.metrics["total_queries"] += 1
        self._save()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return self.metrics.copy()
    
    def prune_strategies(self, min_success_rate: float = 0.3, min_attempts: int = 5) -> int:
        """Prune strategies with poor performance."""
        initial_count = len(self.strategies)
        self.strategies = [
            s for s in self.strategies if 
            s.total_attempts < min_attempts or s.success_rate >= min_success_rate
        ]
        pruned_count = initial_count - len(self.strategies)
        if pruned_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
        return pruned_count
    
    def merge_similar_strategies(self, similarity_threshold: float = STRATEGY_MERGING_THRESHOLD) -> int:
        """Merge strategies that are very similar to each other."""
        if len(self.strategies) <= 1:
            return 0
        
        merged_count = 0
        i = 0
        
        while i < len(self.strategies):
            j = i + 1
            while j < len(self.strategies):
                # Check if strategies are of the same problem type
                if self.strategies[i].problem_type == self.strategies[j].problem_type:
                    # Get strategy texts
                    text_i = self.strategies[i].strategy_text
                    text_j = self.strategies[j].strategy_text
                    
                    # Calculate similarity using TF-IDF
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform([text_i, text_j])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    
                    if similarity >= similarity_threshold:
                        # Merge strategies
                        merged_strategy = self._merge_two_strategies(self.strategies[i], self.strategies[j])
                        # Replace the first strategy with the merged one
                        self.strategies[i] = merged_strategy
                        # Remove the second strategy
                        self.strategies.pop(j)
                        merged_count += 1
                        self.metrics["strategies_merged"] += 1
                        logger.info(f"Merged strategies {self.strategies[i].strategy_id} and {merged_strategy.strategy_id} with similarity {similarity:.2f}")
                    else:
                        j += 1
                else:
                    j += 1
            i += 1
        
        if merged_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
        
        return merged_count
    
    def _merge_two_strategies(self, strategy1: Strategy, strategy2: Strategy) -> Strategy:
        """Merge two similar strategies into one."""
        # Use the strategy with the higher success rate as the base
        if strategy1.success_rate >= strategy2.success_rate:
            base, other = strategy1, strategy2
        else:
            base, other = strategy2, strategy1
        
        # Create a new merged strategy
        merged = Strategy(
            strategy_id=base.strategy_id,
            problem_type=base.problem_type,
            strategy_text=base.strategy_text,
            examples=list(set(base.examples + other.examples)),
            success_count=base.success_count + other.success_count,
            total_attempts=base.total_attempts + other.total_attempts,
            created_at=min(base.created_at, other.created_at) if base.created_at and other.created_at else base.created_at,
            last_used=max(base.last_used, other.last_used) if base.last_used and other.last_used else base.last_used,
            last_updated=datetime.now().isoformat(),
            confidence=(base.confidence + other.confidence) / 2,
            tags=list(set(base.tags + other.tags)),
            reasoning_examples=base.reasoning_examples + other.reasoning_examples
        )
        
        # Keep only a reasonable number of reasoning examples (max 5)
        if len(merged.reasoning_examples) > 5:
            merged.reasoning_examples = merged.reasoning_examples[-5:]
        
        return merged
    
    def limit_strategies_per_type(self, max_per_type: int = MAX_STRATEGIES_PER_TYPE) -> int:
        """
        Limit the number of strategies per problem type to the specified maximum.
        Keeps the best performing strategies based on success rate and recency.
        
        Args:
            max_per_type: Maximum number of strategies to keep per problem type
            
        Returns:
            int: Number of strategies removed
        """
        # Group strategies by problem type
        strategies_by_type = {}
        for strategy in self.strategies:
            if strategy.problem_type not in strategies_by_type:
                strategies_by_type[strategy.problem_type] = []
            strategies_by_type[strategy.problem_type].append(strategy)
        
        # Keep track of strategies to remove
        to_remove = []
        
        # For each problem type, keep only the best max_per_type strategies
        for problem_type, strategies in strategies_by_type.items():
            if len(strategies) <= max_per_type:
                continue
            
            # Score strategies based on success rate (70%) and recency (30%)
            scored_strategies = []
            for strategy in strategies:
                recency_score = 0
                if strategy.last_used:
                    last_used = datetime.fromisoformat(strategy.last_used)
                    days_since = (datetime.now() - last_used).days
                    recency_score = max(0, 1.0 - min(1.0, days_since / 30.0))
                
                score = (0.7 * strategy.success_rate) + (0.3 * recency_score)
                scored_strategies.append((strategy, score))
            
            # Sort by score (descending)
            scored_strategies.sort(key=lambda x: x[1], reverse=True)
            
            # Mark excess strategies for removal
            for strategy, _ in scored_strategies[max_per_type:]:
                to_remove.append(strategy)
        
        # Remove marked strategies
        initial_count = len(self.strategies)
        self.strategies = [s for s in self.strategies if s not in to_remove]
        removed_count = initial_count - len(self.strategies)
        
        if removed_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
            logger.info(f"Removed {removed_count} excess strategies to maintain max {max_per_type} per type")
        
        return removed_count

def extract_thinking(response: str) -> Tuple[str, Optional[str]]:
    """
    Extract thinking content from <think>...</think> tags and the response after.
    
    Args:
        response: The model's response
    
    Returns:
        Tuple[str, Optional[str]]: The cleaned response and the thinking content (if any)
    """
    thinking_content = None
    final_response = response
    
    # Check if there are thinking tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)
    
    if think_matches:
        # Extract thinking content (concatenate if multiple blocks)
        thinking_content = "\n".join(think_matches)
        
        # Extract the response part (everything after the last </think> tag)
        final_parts = response.split('</think>')
        if len(final_parts) > 1:
            final_response = final_parts[-1].strip()
    
    return final_response, thinking_content

def classify_problem(content: str, client, model: str) -> str:
    """
    Use the LLM to classify the problem type, ensuring the result is one of the valid types.
    
    Args:
        content: The query/problem to classify
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        str: The problem type classification (always a valid type)
    """
    # Format problem types as a comma-separated list
    problem_types_str = ", ".join(VALID_PROBLEM_TYPES[:-1])  # Exclude the general_problem fallback
    
    try:
        messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a problem classifier. Your task is to analyze a given problem and categorize it into "
                    f"the most appropriate problem type. You must select EXACTLY ONE problem type from this list: {problem_types_str}.\n\n"
                    f"DO NOT make up new categories. Only use the exact problem types from the list above.\n\n"
                    f"You can use <think>...</think> tags to work through your reasoning process before making a decision.\n\n"
                    f"After your thinking, respond with ONLY the problem type, exactly as it appears in the list. No explanations, no extra words."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Classify the following problem into ONE of these types: {problem_types_str}\n\n"
                    f"Problem: {content}"
                )
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        # Extract final response and thinking content
        raw_response = response.choices[0].message.content
        final_response, thinking = extract_thinking(raw_response)
        
        # Clean and normalize the response
        final_response = final_response.strip().lower()
        
        logger.debug(f"Problem classification - raw response: '{raw_response}'")
        logger.debug(f"Problem classification - final response after removing thinking: '{final_response}'")
        
        # Find the exact match from our list of valid types
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() == final_response:
                logger.info(f"Classified problem as '{valid_type}' (exact match)")
                return valid_type
        
        # If no exact match, look for partial matches
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() in final_response:
                logger.info(f"Classified problem as '{valid_type}' (partial match from '{final_response}')")
                return valid_type
        
        # If still no match, return the general_problem fallback
        logger.warning(f"Could not match '{final_response}' to any valid problem type, using 'general_problem'")
        return "general_problem"
    
    except Exception as e:
        logger.error(f"Error classifying problem: {str(e)}")
        return "general_problem"  # Default fallback

def generate_strategy(problem: str, problem_type: str, client, model: str, db: StrategyDatabase) -> Strategy:
    """
    Generate a new problem-solving strategy using the LLM.
    
    Args:
        problem: The problem that needs a strategy
        problem_type: The type of problem
        client: LLM client for making API calls
        model: Model identifier
        db: The strategy database to use for generating IDs
    
    Returns:
        Strategy: A new strategy for solving this type of problem
    """
    try:
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert problem-solving strategist. Your task is to create a general strategy "
                    "for solving a particular type of problem. Focus on creating a step-by-step approach that "
                    "would help in solving not just the specific example provided, but any problem of this type.\n\n"
                    "Your strategy should be:\n"
                    "1. Clear and concise\n"
                    "2. Step-by-step\n"
                    "3. Generalizable to similar problems\n"
                    "4. Include specific techniques, not just general advice\n\n"
                    "First think through your approach using <think>...</think> tags. Then provide your "
                    "final strategy after the thinking section. Make sure your strategy is clear, specific, "
                    "and actionable."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Create a problem-solving strategy for the following {problem_type} problem:\n\n"
                    f"{problem}\n\n"
                    f"This strategy should help solve not just this specific problem, but any {problem_type} problem."
                )
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Medium temperature for creative but focused output
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        response_text = response.choices[0].message.content
        
        # Extract final strategy and thinking
        strategy_text, thinking = extract_thinking(response_text)
        if not strategy_text.strip():
            strategy_text = response_text  # Use full response if extraction failed
        
        logger.debug(f"Generated strategy - raw response: '{response_text}'")
        logger.debug(f"Generated strategy - final text after removing thinking: '{strategy_text}'")
        
        # Create a new strategy object using the provided database for ID generation
        strategy = Strategy(
            strategy_id=db.get_next_strategy_id(),  # Use the provided DB instance
            problem_type=problem_type,
            strategy_text=strategy_text.strip(),
            examples=[problem],
            created_at=datetime.now().isoformat(),
            reasoning_examples=[thinking] if thinking else []
        )
        
        logger.info(f"Generated new strategy for {problem_type}: ID {strategy.strategy_id}")
        return strategy
    
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        # Create a minimal fallback strategy with a unique ID
        fallback_id = f"fallback_{uuid.uuid4().hex[:8]}"
        logger.info(f"Using fallback strategy with ID: {fallback_id}")
        return Strategy(
            strategy_id=fallback_id,
            problem_type=problem_type,
            strategy_text=(
                f"When solving {problem_type} problems:\n"
                "1. Break down the problem into smaller parts\n"
                "2. Solve each part systematically\n"
                "3. Combine the solutions"
            ),
            examples=[problem]
        )

def should_create_new_strategy(problem_type: str, query: str, existing_strategies: List[Strategy], db: StrategyDatabase) -> Tuple[bool, Optional[Strategy]]:
    """
    Determine whether to create a new strategy or update an existing one.
    
    Args:
        problem_type: The type of problem
        query: The current query/problem
        existing_strategies: Existing strategies for this problem type
        db: Strategy database
    
    Returns:
        Tuple[bool, Optional[Strategy]]: 
            - Boolean indicating if a new strategy should be created
            - The similar strategy to update (if any)
    """
    # If there are no existing strategies, definitely create one
    if not existing_strategies:
        return True, None
    
    # If we already have enough strategies for this problem type, check if the query
    # is similar to any existing strategy
    if len(existing_strategies) >= MAX_STRATEGIES_PER_TYPE:
        # First, check similarity based on strategy text
        similar_strategy_result = db.find_similar_strategy(problem_type, query)
        if similar_strategy_result:
            similar_strategy, similarity = similar_strategy_result
            logger.info(f"Found similar strategy {similar_strategy.strategy_id} with text similarity {similarity:.2f}")
            return False, similar_strategy
            
        # Next, check similarity based on examples
        similar_examples_result = db.find_similar_examples(problem_type, query)
        if similar_examples_result:
            similar_strategy, similarity = similar_examples_result
            logger.info(f"Found strategy {similar_strategy.strategy_id} with similar examples, similarity {similarity:.2f}")
            return False, similar_strategy
            
        # If we're at or above max strategies and no similar strategy was found,
        # use the strategy with the lowest success rate for this new example
        if existing_strategies:
            # Sort by success rate (ascending)
            existing_strategies.sort(key=lambda s: s.success_rate)
            worst_strategy = existing_strategies[0]
            logger.info(f"At maximum strategies for {problem_type}, updating lowest performing strategy {worst_strategy.strategy_id}")
            return False, worst_strategy
    
    # If we have fewer than the maximum allowed strategies for this type,
    # check strategy similarity before creating a new one
    similar_strategy_result = db.find_similar_strategy(problem_type, query, threshold=STRATEGY_CREATION_THRESHOLD)
    if similar_strategy_result:
        similar_strategy, similarity = similar_strategy_result
        logger.info(f"Found similar strategy {similar_strategy.strategy_id} with text similarity {similarity:.2f}")
        return False, similar_strategy
    
    # If we get here, we should create a new strategy
    logger.info(f"No similar strategy found for {problem_type}, creating a new one")
    return True, None

def select_relevant_strategies(query: str, problem_type: str, db: StrategyDatabase, max_strategies: int = MAX_STRATEGIES_PER_TYPE) -> List[Strategy]:
    """
    Select the most relevant strategies for a given problem.
    
    Args:
        query: The problem/query text
        problem_type: The type of problem
        db: Strategy database
        max_strategies: Maximum number of strategies to return
    
    Returns:
        List[Strategy]: The selected strategies
    """
    # First, get strategies specifically for this problem type
    type_specific = db.get_strategies_for_problem(problem_type)
    
    # If we have more than we need, sort by success rate and recency
    if len(type_specific) > max_strategies:
        # Score each strategy based on success rate and recency
        scored_strategies = []
        for strategy in type_specific:
            recency_score = 0
            if strategy.last_used:
                # Calculate days since last use
                last_used = datetime.fromisoformat(strategy.last_used)
                days_since = (datetime.now() - last_used).days
                recency_score = max(0, 1.0 - min(1.0, days_since / 30.0))  # Higher for more recent
            
            # Combined score with success rate weighing more
            score = (0.7 * strategy.success_rate) + (0.3 * recency_score)
            scored_strategies.append((strategy, score))
        
        # Sort by score (descending) and take top strategies
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_strategies[:max_strategies]]
    
    # If we don't have enough type-specific strategies, also get similar strategies
    if len(type_specific) < max_strategies:
        # Get similar strategies, excluding those already in type_specific
        type_specific_ids = {s.strategy_id for s in type_specific}
        similar_strategies = []
        
        for s, score in db.get_similar_strategies(query, n=max_strategies*2):  # Get more than needed to filter
            if s.strategy_id not in type_specific_ids and s.problem_type != problem_type:
                similar_strategies.append(s)
                if len(similar_strategies) >= (max_strategies - len(type_specific)):
                    break
        
        # Return a combination of type-specific and similar strategies
        combined = type_specific + similar_strategies
        return combined[:max_strategies]
    
    return type_specific[:max_strategies]

def evaluate_strategy_effectiveness(response: str, thinking: Optional[str], selected_strategies: List[Strategy], client, model: str) -> Dict[str, bool]:
    """
    Evaluate how effective each strategy was in generating the response.
    
    Args:
        response: The LLM's final response to the query
        thinking: The LLM's reasoning process (if any)
        selected_strategies: The strategies that were used
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Dict[str, bool]: Mapping from strategy ID to effectiveness (True/False)
    """
    if not selected_strategies:
        return {}
    
    results = {}
    
    try:
        for strategy in selected_strategies:
            # Include thinking in the evaluation if available
            full_response = thinking + "\n\n" + response if thinking else response
            
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are evaluating the effectiveness of a problem-solving strategy. "
                        "Analyze the provided response and determine if it shows evidence that the strategy was "
                        "successfully applied. You can use <think>...</think> tags to work through your reasoning process, "
                        "but your final answer must be either YES or NO only."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Strategy:\n{strategy.strategy_text}\n\n"
                        f"Response (including reasoning):\n{full_response}\n\n"
                        f"Does the response show clear evidence that the strategy was effectively applied? "
                        f"Answer with ONLY YES or NO."
                    )
                }
            ]
            
            eval_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
            )
            
            # Get the response and extract final answer (remove thinking blocks)
            result_text = eval_response.choices[0].message.content
            final_result, eval_thinking = extract_thinking(result_text)
            
            # Clean up and normalize the result
            final_result = final_result.strip().upper()
            
            logger.debug(f"Strategy evaluation - raw response: '{result_text}'")
            logger.debug(f"Strategy evaluation - final result after removing thinking: '{final_result}'")
            
            # Check for YES in the final answer (not in thinking blocks)
            is_effective = "YES" in final_result
            
            results[strategy.strategy_id] = is_effective
            logger.info(f"Strategy {strategy.strategy_id} evaluation: {final_result} -> {is_effective}")
    
    except Exception as e:
        logger.error(f"Error evaluating strategy effectiveness: {str(e)}")
        # Default to neutral results if evaluation fails
        for strategy in selected_strategies:
            results[strategy.strategy_id] = False
    
    return results

def refine_strategy(strategy: Strategy, problem: str, response: str, thinking: Optional[str], client, model: str) -> Strategy:
    """
    Refine a strategy based on its application to a specific problem.
    
    Args:
        strategy: The strategy to refine
        problem: The problem that was solved
        response: The LLM's final response to the problem
        thinking: The LLM's reasoning process (if any)
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Strategy: The refined strategy
    """
    try:
        # Include thinking in refinement if available
        full_response = thinking + "\n\n" + response if thinking else response
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are tasked with refining a problem-solving strategy based on a new example. "
                    "Analyze the original strategy, the problem, and the detailed solution process. "
                    "Then provide an improved version of the strategy that would be more effective for "
                    "solving similar problems in the future. Focus on making the strategy more clear, "
                    "more general, and more effective.\n\n"
                    "You can use <think>...</think> tags to explore your refinement process in detail.\n\n"
                    "After your thinking, provide ONLY the refined strategy text, no introduction or explanation."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Original strategy for {strategy.problem_type} problems:\n{strategy.strategy_text}\n\n"
                    f"New problem:\n{problem}\n\n"
                    f"Solution process (including reasoning):\n{full_response}\n\n"
                    f"Provide a refined version of the original strategy that incorporates "
                    f"any insights from this new example."
                )
            }
        ]
        
        refine_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        response_text = refine_response.choices[0].message.content
        
        # Extract refined strategy and thinking
        refined_text, refinement_thinking = extract_thinking(response_text)
        if not refined_text.strip():
            refined_text = response_text  # Use full response if extraction failed
        
        logger.debug(f"Strategy refinement - raw response: '{response_text}'")
        logger.debug(f"Strategy refinement - final text after removing thinking: '{refined_text}'")
        
        # Create a copy of the strategy with the refined text
        refined_strategy = Strategy(
            strategy_id=strategy.strategy_id,
            problem_type=strategy.problem_type,
            strategy_text=refined_text.strip(),
            examples=strategy.examples + [problem],
            success_count=strategy.success_count,
            total_attempts=strategy.total_attempts,
            created_at=strategy.created_at,
            last_used=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            confidence=strategy.confidence,
            tags=strategy.tags,
            reasoning_examples=strategy.reasoning_examples.copy()
        )
        
        # Add the refinement thinking if available
        if refinement_thinking:
            refined_strategy.add_reasoning_example(refinement_thinking)
        
        return refined_strategy
    
    except Exception as e:
        logger.error(f"Error refining strategy: {str(e)}")
        # Return the original strategy if refinement fails
        return strategy

def augment_system_prompt(system_prompt: str, strategies: List[Strategy]) -> str:
    """
    Augment the system prompt with selected strategies and reasoning examples.
    
    Args:
        system_prompt: The original system prompt
        strategies: A list of strategies to add to the prompt
    
    Returns:
        str: The augmented system prompt
    """
    if not strategies:
        return system_prompt
    
    # Create the strategy section
    strategy_section = "\n\n## Problem-Solving Strategies\n\n"
    
    for i, strategy in enumerate(strategies, 1):
        strategy_section += f"### Strategy {i} for {strategy.problem_type} problems\n{strategy.strategy_text}\n\n"
        
        # Add a sample reasoning example if available
        if strategy.reasoning_examples:
            # Use the most recent reasoning example (last one)
            reasoning = strategy.reasoning_examples[-1]
            if reasoning:
                strategy_section += f"#### Example reasoning process:\n<think>\n{reasoning}\n</think>\n\n"
    
    # Add encouragement to use thinking tags
    strategy_section += (
        "Feel free to use <think>...</think> tags to work through your reasoning process "
        "before providing the final answer. This helps with complex problem-solving.\n\n"
    )
    
    # Append the strategy section to the system prompt
    augmented_prompt = system_prompt + strategy_section
    
    return augmented_prompt

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None) -> Tuple[str, int]:
    """
    Main plugin function that implements system prompt learning.
    
    Args:
        system_prompt: The system prompt
        initial_query: The user's query
        client: The LLM client
        model: The model identifier
        request_config: Optional request configuration
                       Can include {'spl_inference_only': True} to run in inference-only mode
    
    Returns:
        Tuple[str, int]: The LLM response and token count
    """
    start_time = time.time()
    logger.info(f"Starting SPL plugin execution for query: {initial_query[:100]}...")
    
    # Check if we should run in inference-only mode
    inference_only = False
    if request_config and 'spl_inference_only' in request_config:
        inference_only = request_config['spl_inference_only']
        logger.info(f"Running in inference-only mode: {inference_only}")
        
    # Initialize the strategy database
    db = StrategyDatabase()
    logger.info(f"Current strategy count: {len(db.strategies)}")
    logger.info(f"Last strategy ID: {db.metrics.get('last_strategy_id', 0)}")
    
    # Only increment query count in learning mode
    if not inference_only:
        db.increment_query_count()
        db._save()  # Save immediately to ensure counter is persisted
    
    # 1. Classify the problem type
    problem_type = classify_problem(initial_query, client, model)
    logger.info(f"Classified problem as: {problem_type}")
    
    # 2. Get existing strategies for this problem type
    existing_strategies = db.get_strategies_for_problem(problem_type)
    logger.info(f"Found {len(existing_strategies)} existing strategies for {problem_type}")
    
    # 3. Determine if we need to create a new strategy or update an existing one
    similar_strategy = None
    
    if not inference_only:
        # In learning mode, check if we should create a new strategy or update an existing one
        should_create, similar_strategy = should_create_new_strategy(
            problem_type, 
            initial_query, 
            existing_strategies, 
            db
        )
        
        if should_create:
            # Create a new strategy
            logger.info(f"Creating new strategy for {problem_type}")
            new_strategy = generate_strategy(initial_query, problem_type, client, model, db)
            db.add_strategy(new_strategy)
            logger.info(f"Added new strategy with ID: {new_strategy.strategy_id}")
        
        elif similar_strategy:
            # Update existing strategy with new example
            logger.info(f"Updating existing strategy {similar_strategy.strategy_id} with new example")
            db.add_example_to_strategy(similar_strategy.strategy_id, initial_query)
    
    # 4. Perform database maintenance (more frequently than before)
    if not inference_only and db.metrics["total_queries"] % MAINTENANCE_INTERVAL == 0:
        # 4.1 Merge similar strategies
        merged_count = db.merge_similar_strategies(similarity_threshold=STRATEGY_MERGING_THRESHOLD)
        logger.info(f"Merged {merged_count} similar strategies")
        
        # 4.2 Limit strategies per problem type
        limited_count = db.limit_strategies_per_type(max_per_type=MAX_STRATEGIES_PER_TYPE)
        
        # 4.3 Prune low-performing strategies
        pruned_count = db.prune_strategies()
        logger.info(f"Pruned {pruned_count} low-performing strategies")
    
    # 5. Re-select strategies (in case the database changed in step 4)
    existing_strategies = db.get_strategies_for_problem(problem_type)
    
    # 6. Select relevant strategies for this problem
    selected_strategies = select_relevant_strategies(initial_query, problem_type, db)
    
    # Log the selected strategies
    for i, strategy in enumerate(selected_strategies, 1):
        logger.info(f"Selected strategy {i}: {strategy.strategy_id} (success rate: {strategy.success_rate:.2f})")
    
    # 7. If no strategies selected, use fallback
    if not selected_strategies:
        logger.info(f"No strategies selected, using fallback strategy")
        fallback_strategy = Strategy(
            strategy_id="fallback_temporary",
            problem_type=problem_type,
            strategy_text=(
                f"When solving {problem_type} problems:\n"
                "1. Break down the problem into manageable parts\n"
                "2. Analyze each part systematically\n"
                "3. Apply appropriate techniques for each component\n"
                "4. Combine the results into a cohesive solution"
            ),
            examples=[initial_query]
        )
        selected_strategies = [fallback_strategy]
    
    # 8. Augment the system prompt with the selected strategies
    augmented_prompt = augment_system_prompt(system_prompt, selected_strategies)
    logger.info(f"Augmented system prompt with {len(selected_strategies)} strategies")
    
    # 9. Forward the request to the LLM with the augmented prompt
    try:
        # Create a copy of request_config without spl_inference_only
        request_params = {}
        if request_config:
            request_params = {k: v for k, v in request_config.items() if k != 'spl_inference_only'}
        
        # Ensure max_tokens is set to at least DEFAULT_MAX_TOKENS for reasoning LLMs
        if 'max_tokens' not in request_params:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
        elif request_params['max_tokens'] < DEFAULT_MAX_TOKENS:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": augmented_prompt},
                {"role": "user", "content": initial_query}
            ],
            **request_params
        )
        
        completion_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content
        
        # Extract final response and thinking content
        final_response, thinking = extract_thinking(response_text)
        
        logger.debug(f"Main response - raw: '{response_text}'")
        if thinking:
            logger.debug(f"Main response - thinking extracted: '{thinking}'")
            logger.debug(f"Main response - final answer after removing thinking: '{final_response}'")
        
        # Only perform learning operations if not in inference-only mode
        if not inference_only:
            # 10. Evaluate the effectiveness of the strategies
            strategy_effectiveness = evaluate_strategy_effectiveness(
                final_response,
                thinking,
                selected_strategies,
                client,
                model
            )
            
            # 11. Update strategy metrics based on effectiveness
            for strategy_id, effective in strategy_effectiveness.items():
                # Skip temporary fallback strategies
                if strategy_id != "fallback_temporary":
                    db.update_strategy_performance(strategy_id, effective)
                    logger.info(f"Strategy {strategy_id} effectiveness: {effective}")
                    
                    # If the strategy was effective and thinking was used, add the thinking as a reasoning example
                    if effective and thinking and strategy_id != "fallback_temporary":
                        db.add_reasoning_example(strategy_id, thinking)
                        logger.info(f"Added reasoning example to strategy {strategy_id}")
            
            # 12. Periodically refine strategies (after every 10 uses)
            for strategy in selected_strategies:
                # Skip temporary fallback strategies
                if (strategy.strategy_id != "fallback_temporary" and 
                    strategy.total_attempts % 10 == 0 and 
                    strategy.total_attempts > 0):
                    logger.info(f"Refining strategy {strategy.strategy_id} after {strategy.total_attempts} attempts")
                    refined_strategy = refine_strategy(strategy, initial_query, final_response, thinking, client, model)
                    db.refine_strategy(strategy.strategy_id, refined_strategy.strategy_text)
        else:
            logger.info("Skipping strategy evaluation and refinement in inference-only mode")
        
        # Log execution time and status after run
        execution_time = time.time() - start_time
        logger.info(f"SPL plugin execution completed in {execution_time:.2f} seconds")
        logger.info(f"Final strategy count: {len(db.strategies)}")
        logger.info(f"Final last strategy ID: {db.metrics.get('last_strategy_id', 0)}")
        
        # Return the original response to preserve the thinking tag format
        return response_text, completion_tokens
    
    except Exception as e:
        logger.error(f"Error in SPL plugin: {str(e)}")
        # Fall back to regular completion on error
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ],
            max_tokens=DEFAULT_MAX_TOKENS  # Ensure fallback also uses sufficient tokens
        )
        return response.choices[0].message.content, response.usage.completion_tokens
