"""
Steering vector manager for AutoThink.

This module provides functionality to load and apply steering vectors
from Hugging Face datasets during inference.
"""

import torch
import logging
import random
import json
import datasets
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

class SteeringVectorManager:
    """
    Manager for loading and applying steering vectors from a dataset.
    """
    
    def __init__(
        self, 
        dataset_name: str,
        target_layer: int = 19,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the steering vector manager.
        
        Args:
            dataset_name: Name of the HuggingFace dataset containing steering vectors
            target_layer: Target layer for applying steering vectors
            cache_dir: Directory for caching the dataset
            device: Device to use for tensors
        """
        self.dataset_name = dataset_name
        self.target_layer = target_layer
        self.cache_dir = cache_dir
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        # Storage for steering vectors
        self.steering_vectors = []
        self.pattern_to_vectors = {}
        self.tokenized_contexts = {}
        
        # Default steering strengths
        self.default_strength = 2.0
        self.pattern_strengths = {
            "depth_and_thoroughness": 2.5,
            "numerical_accuracy": 2.0, 
            "self_correction": 3.0,
            "exploration": 2.0,
            "organization": 1.5,
            "unknown": 1.0
        }
        
        # If dataset is provided, load it
        if dataset_name:
            self.load_dataset()
        
    def load_dataset(self):
        """Load steering vectors from the HuggingFace dataset."""
        try:
            logger.info(f"Loading steering vectors from dataset: {self.dataset_name}")
            
            # Load the dataset
            dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)
            
            # Get the main split (usually 'train')
            main_split = list(dataset.keys())[0]
            vector_data = dataset[main_split]
            
            # Load each item as a steering vector
            for item in vector_data:
                # Convert dataset item to proper format
                vector = self._process_dataset_item(item)
                if vector:
                    self.steering_vectors.append(vector)
                    
                    # Group by reasoning pattern
                    pattern = vector.get("reasoning_pattern", "unknown")
                    if pattern not in self.pattern_to_vectors:
                        self.pattern_to_vectors[pattern] = []
                    self.pattern_to_vectors[pattern].append(vector)
            
            logger.info(f"Loaded {len(self.steering_vectors)} steering vectors")
            logger.info(f"Found {len(self.pattern_to_vectors)} reasoning patterns: {list(self.pattern_to_vectors.keys())}")
            
            # Log the first vector for debugging
            if self.steering_vectors:
                first_vector = self.steering_vectors[0]
                logger.info(f"First vector sample - pattern: {first_vector.get('reasoning_pattern', 'missing')}")
                if 'pivot_context' in first_vector:
                    context_len = len(first_vector['pivot_context'])
                    logger.info(f"First vector pivot_context length: {context_len}")
        
        except Exception as e:
            logger.error(f"Error loading steering vectors: {e}")
            self.steering_vectors = []
            self.pattern_to_vectors = {}
    
    def _process_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a dataset item into a steering vector.
        
        Args:
            item: Dataset item
            
        Returns:
            Processed steering vector or None if invalid
        """
        try:
            # Check if item has the required fields
            required_fields = ["pivot_context", "steering_vector", "reasoning_pattern"]
            if not all(field in item for field in required_fields):
                return None
            
            # Convert steering_vector to a proper format if it's a string or list
            steering_vector = item["steering_vector"]
            if isinstance(steering_vector, str):
                # Try to parse JSON string
                try:
                    steering_vector = json.loads(steering_vector)
                except json.JSONDecodeError:
                    # Try comma-separated format
                    steering_vector = [float(x) for x in steering_vector.strip("[]").split(",")]
            
            # Ensure we have a proper list
            if not isinstance(steering_vector, list):
                logger.warning(f"Invalid steering vector format: {type(steering_vector)}")
                return None
            
            # Create the steering vector dictionary
            vector = {
                "pivot_context": item["pivot_context"],
                "pivot_token": item.get("pivot_token", ""),
                "pivot_token_id": item.get("pivot_token_id", -1),
                "prob_before": item.get("prob_before", 0.0),
                "prob_after": item.get("prob_after", 0.0),
                "prob_delta": item.get("prob_delta", 0.0),
                "model_id": item.get("model_id", ""),
                "task_type": item.get("task_type", "unknown"),
                "steering_vector": steering_vector,
                "cluster_id": item.get("cluster_id", -1),
                "reasoning_pattern": item.get("reasoning_pattern", "unknown"),
                "cluster_vector": item.get("cluster_vector", steering_vector),
                "steering_layer": item.get("steering_layer", self.target_layer),
            }
            
            return vector
        
        except Exception as e:
            logger.error(f"Error processing dataset item: {e}")
            return None
    
    def create_tokenized_contexts(self, tokenizer):
        """
        Pre-tokenize context patterns for efficient matching during generation.
        
        Args:
            tokenizer: Tokenizer for encoding contexts
        """
        # Get configurations
        max_pts_tokens = 256  # Maximum tokens to store for matching
        
        count = 0
        for vector in self.steering_vectors:
            # Get the context
            context = vector.get("pivot_context", "")
            if not context:
                continue
                
            # Pre-tokenize the context for faster matching
            tokenized_context = tokenizer.encode(context, add_special_tokens=False)
            
            # Keep only up to max_pts_tokens
            if len(tokenized_context) > max_pts_tokens:
                tokenized_context = tokenized_context[-max_pts_tokens:]
            
            # Store the tokenized context with its vector
            tuple_key = tuple(tokenized_context)
            self.tokenized_contexts[tuple_key] = vector
            
            # Store additional shorter versions for partial matching
            for suffix_len in [4, 8, 12]:
                if len(tokenized_context) > suffix_len:
                    suffix = tokenized_context[-suffix_len:]
                    suffix_tuple = tuple(suffix)
                    if suffix_tuple not in self.tokenized_contexts:
                        self.tokenized_contexts[suffix_tuple] = vector
            
            count += 1
            
        # Log statistics
        logger.info(f"Pre-tokenized {count} contexts into {len(self.tokenized_contexts)} token patterns")
        
        # Count patterns by length for debugging
        length_counts = {}
        for key in self.tokenized_contexts.keys():
            length = len(key)
            if length not in length_counts:
                length_counts[length] = 0
            length_counts[length] += 1
        
        logger.info(f"Token pattern length distribution: {sorted(length_counts.items())}")
    
    def get_steering_strength(self, pattern: str) -> float:
        """
        Get the steering strength for a specific pattern.
        
        Args:
            pattern: The reasoning pattern
            
        Returns:
            The steering strength
        """
        return self.pattern_strengths.get(pattern, self.default_strength)
    
    def set_steering_strength(self, pattern: str, strength: float):
        """
        Set the steering strength for a specific pattern.
        
        Args:
            pattern: The reasoning pattern
            strength: The steering strength
        """
        self.pattern_strengths[pattern] = strength
        logger.info(f"Set strength for {pattern} to {strength}")
    
    def get_pattern_vectors(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Get all steering vectors for a specific reasoning pattern.
        
        Args:
            pattern: The reasoning pattern
            
        Returns:
            List of steering vectors
        """
        return self.pattern_to_vectors.get(pattern, [])

class SteeringHook:
    """Hook for applying steering vectors during generation."""
    
    def __init__(self, manager: SteeringVectorManager, layer_num: int, tokenizer=None):
        """
        Initialize the steering hook.
        
        Args:
            manager: The steering vector manager
            layer_num: The layer number to apply steering to
            tokenizer: Tokenizer for token-based matching
        """
        self.manager = manager
        self.layer_num = layer_num
        self.tokenizer = tokenizer
        
        # For token-based matching
        self.token_history = []  # Store token IDs for matching
        self.max_history = 256   # Maximum tokens to keep in history
        
        # State tracking
        self.match_found = False
        self.current_vector = None
        self.last_pattern = None
        
        # Single pattern for entire request
        self.active_pattern = None  # Currently active pattern
        self.generation_started = False
        
        logger.info(f"Initialized hook for layer {layer_num}")
    
    def __call__(self, module, input_tensors, output):
        """
        Apply steering to the output of a layer.
        
        Args:
            module: The module being hooked
            input_tensors: The input tensors
            output: The output tensor
            
        Returns:
            Modified output tensor
        """
        try:
            # Skip if no active pattern is set
            if not self.active_pattern:
                return output
            
            # Apply steering vector if available
            if self.current_vector is not None:
                # Get the appropriate steering strength
                pattern = self.current_vector.get("reasoning_pattern", "unknown")
                strength = self.manager.get_steering_strength(pattern)
                
                # Keep strength within safe bounds
                safe_strength = min(max(strength, 0.1), 2.0)
                
                # Log when pattern changes
                if pattern != self.last_pattern:
                    logger.info(f"Switching to {pattern} reasoning pattern with strength {safe_strength}")
                    self.last_pattern = pattern
                
                # Apply the steering vector
                try:
                    if isinstance(output, tuple):
                        # Some models return a tuple
                        hidden_states = output[0]
                        modified_hidden_states = self._apply_steering_vector(hidden_states, self.current_vector, safe_strength)
                        
                        # Validate the result
                        if modified_hidden_states.shape == hidden_states.shape:
                            return (modified_hidden_states,) + output[1:]
                        else:
                            logger.error(f"Modified hidden states have wrong shape. Expected {hidden_states.shape}, got {modified_hidden_states.shape}")
                            return output
                    else:
                        # Direct tensor output
                        return self._apply_steering_vector(output, self.current_vector, safe_strength)
                        
                except Exception as e:
                    logger.error(f"Error applying steering: {e}")
                    return output
            
            return output
        except Exception as e:
            logger.error(f"Critical error in hook: {e}")
            return output
    
    def _apply_steering_vector(self, hidden_states: torch.Tensor, 
                              steering_vector: Dict[str, Any],
                              scaling_factor: float = 2.0) -> torch.Tensor:
        """
        Apply a steering vector to hidden states.
        
        Args:
            hidden_states: The hidden states tensor
            steering_vector: Dictionary with steering vector data
            scaling_factor: Factor to scale the steering vector by
            
        Returns:
            Modified hidden states tensor
        """
        try:
            # Make a deep clone
            hidden_states_clone = hidden_states.clone().detach()
            
            # Check what kind of vector we're using
            vector_data = None
            if "steering_vector" in steering_vector:
                vector_data = steering_vector["steering_vector"]
                vector_type = "steering_vector"
            elif "cluster_vector" in steering_vector:
                vector_data = steering_vector["cluster_vector"]
                vector_type = "cluster_vector"
            else:
                logger.warning("No valid vector found in steering data")
                return hidden_states
            
            # Convert vector to tensor
            vector = torch.tensor(vector_data, 
                                 dtype=hidden_states.dtype, 
                                 device=hidden_states.device)
            
            # Log vector info
            pattern = steering_vector.get("reasoning_pattern", "unknown")
            logger.debug(f"Applying {vector_type} for pattern '{pattern}' with scaling {scaling_factor}")
            
            # Apply scaling based on prob_delta if available
            if "prob_delta" in steering_vector:
                prob_delta = abs(steering_vector["prob_delta"])
                prob_delta_capped = min(max(prob_delta, 0.1), 2.0)
                scaling_factor *= prob_delta_capped
            
            # Check if the token is positive or negative
            is_positive = steering_vector.get("is_positive", True)
            
            # Verify shapes are compatible
            hs_shape = hidden_states.shape
            vector_shape = vector.shape
            
            if len(vector_shape) != 1 or vector_shape[0] != hs_shape[-1]:
                logger.error(f"Shape mismatch - hidden_states: {hs_shape}, vector: {vector_shape}")
                return hidden_states
            
            # Bound scaling factor for safety
            safe_scaling = min(max(scaling_factor, 0.0), 3.0)
            
            # Apply steering
            if len(hs_shape) >= 3 and hs_shape[0] > 0 and hs_shape[1] > 0:
                # Apply to the last token's representation
                if is_positive:
                    # Normalize vector to prevent numerical instability
                    vector_norm = torch.nn.functional.normalize(vector, dim=0)
                    hidden_states_clone[-1, -1, :] = hidden_states_clone[-1, -1, :] + safe_scaling * vector_norm
                else:
                    vector_norm = torch.nn.functional.normalize(vector, dim=0)
                    hidden_states_clone[-1, -1, :] = hidden_states_clone[-1, -1, :] - safe_scaling * vector_norm
                
                # Check for NaN or inf values
                if torch.isnan(hidden_states_clone).any() or torch.isinf(hidden_states_clone).any():
                    logger.error("NaN or inf values detected after applying vector, reverting to original")
                    return hidden_states
            else:
                logger.error(f"Hidden states shape not suitable for steering: {hs_shape}")
                return hidden_states
            
            return hidden_states_clone
        except Exception as e:
            logger.error(f"Unexpected error applying steering vector: {e}")
            return hidden_states
    
    def update_token_history(self, new_tokens: List[int]):
        """
        Update the token history with new tokens.
        
        Args:
            new_tokens: New token IDs to add
        """
        # Add to token history
        self.token_history.extend(new_tokens)
        
        # Trim history if needed
        if len(self.token_history) > self.max_history:
            self.token_history = self.token_history[-self.max_history:]
        
        # Log token updates periodically
        if random.random() < 0.01:
            logger.debug(f"Token history updated, now has {len(self.token_history)} tokens")
    
    def try_match(self) -> bool:
        """
        Try to match the current context with a steering vector.
        
        Returns:
            Boolean indicating if a match was found
        """
        # If we already have an active pattern, don't try to match again
        if self.generation_started and self.active_pattern:
            return False
        
        # Only attempt pattern matching at the beginning of generation
        self.generation_started = True
        
        # Try token-based matching
        match_result = self._try_token_match()
        
        # If a match is found, set this as the permanent pattern for this generation
        if match_result and self.current_vector:
            new_pattern = self.current_vector.get("reasoning_pattern", "unknown")
            self.active_pattern = new_pattern
            logger.info(f"Selected '{new_pattern}' pattern for this request")
        
        return match_result
    
    def _try_token_match(self) -> bool:
        """
        Try to match using token-based context.
        
        Returns:
            Boolean indicating if a match was found
        """
        # Ensure we have enough tokens
        if len(self.token_history) < 4:
            return False
        
        # Track best match
        best_match = {
            'length': 0, 
            'vector': None,
            'is_partial': True
        }
        
        # Check for matches in tokenized contexts
        for tokenized_context, vector in self.manager.tokenized_contexts.items():
            token_list = list(tokenized_context)
            token_len = len(token_list)
            
            # Try partial matching for shorter contexts
            if len(self.token_history) < token_len:
                # Only try partial matching if we have enough context tokens
                if len(self.token_history) >= 4:
                    # Calculate how many tokens to match
                    match_len = min(len(self.token_history), max(4, token_len // 2))
                    # Try to match the end of the token sequence
                    if self.token_history[-match_len:] == token_list[-match_len:]:
                        # Track this match - prefer longer matches
                        if match_len > best_match['length']:
                            best_match = {
                                'length': match_len,
                                'vector': vector,
                                'is_partial': True,
                                'match_len': match_len,
                                'token_len': token_len
                            }
            else:
                # Full matching when we have enough tokens
                if self.token_history[-token_len:] == token_list:
                    # Track this match - full matches are preferred
                    if token_len >= best_match['length']:
                        best_match = {
                            'length': token_len,
                            'vector': vector,
                            'is_partial': False,
                            'match_len': token_len,
                            'token_len': token_len
                        }
        
        # Apply best match if found
        if best_match['vector'] is not None:
            match_type = "PARTIAL" if best_match['is_partial'] else "FULL"
            self.match_found = True
            self.current_vector = best_match['vector']
            pattern = best_match['vector'].get("reasoning_pattern", "unknown")
            logger.info(f"Found {match_type} token match ({best_match['match_len']}/{best_match['token_len']} tokens) for {pattern} pattern")
            return True
        
        return False
    
    def reset(self):
        """Reset the hook state."""
        self.match_found = False
        self.current_vector = None
        self.token_history = []
        self.last_pattern = None
        self.active_pattern = None
        self.generation_started = False

def install_steering_hooks(model, manager: SteeringVectorManager, tokenizer=None) -> List[Tuple]:
    """
    Install steering hooks on a model.
    
    Args:
        model: The model to install hooks on
        manager: The steering vector manager
        tokenizer: Tokenizer for token-based matching
        
    Returns:
        List of installed hooks
    """
    hooks = []
    
    # Target layer is specified in the manager
    layer_num = manager.target_layer
    logger.info(f"Attempting to install hook on layer {layer_num}")
    
    # First, log model structure to help with debugging
    model_type = type(model).__name__
    logger.info(f"Model type is {model_type}")
    
    # Find the appropriate module - depends on model architecture
    module = None
    if hasattr(model, 'transformer'):
        logger.info("Model has 'transformer' attribute")
        if hasattr(model.transformer, 'h') and layer_num < len(model.transformer.h):
            module = model.transformer.h[layer_num]
            logger.info(f"Using transformer.h[{layer_num}]")
    elif hasattr(model, 'model'):
        logger.info("Model has 'model' attribute")
        if hasattr(model.model, 'layers') and layer_num < len(model.model.layers):
            module = model.model.layers[layer_num]
            logger.info(f"Using model.layers[{layer_num}]")
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers') and layer_num < len(model.model.decoder.layers):
            module = model.model.decoder.layers[layer_num]
            logger.info(f"Using model.decoder.layers[{layer_num}]")
    elif hasattr(model, 'layers') and layer_num < len(model.layers):
        module = model.layers[layer_num]
        logger.info(f"Using layers[{layer_num}]")
    
    if module is None:
        logger.error(f"Could not find appropriate module for layer {layer_num}")
        logger.error("Model structure not compatible with current hook installation logic")
        return []
    
    # Create and register hook
    hook = SteeringHook(manager, layer_num, tokenizer)
    handle = module.register_forward_hook(hook)
    
    # Return both hook object and handle for later removal
    hooks.append((hook, handle))
    
    logger.info(f"Installed hook on layer {layer_num} successfully")
    
    return hooks

def remove_steering_hooks(hooks):
    """
    Remove steering hooks from a model.
    
    Args:
        hooks: List of (hook, handle) tuples
    """
    for _, handle in hooks:
        handle.remove()
    
    logger.info(f"Removed {len(hooks)} hooks")
