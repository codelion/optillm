"""
DeepConf main entry point.

Implements the deepconf_decode function that integrates with OptILLM's
local inference system for confidence-aware reasoning with early termination.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .processor import DeepConfProcessor, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def deepconf_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]],
    request_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, int]:
    """
    Main DeepConf decoding function for integration with OptILLM.
    
    Implements confidence-aware reasoning with early termination for local models.
    Uses online mode with warmup phase and dynamic threshold calibration.
    
    Args:
        model: The local language model
        tokenizer: The tokenizer for the model
        messages: List of input messages in chat format
        request_config: Optional configuration overrides
        
    Returns:
        Tuple of (generated_response, total_tokens_used)
        
    Raises:
        ValueError: If invalid configuration provided
        RuntimeError: If processing fails
    """
    logger.info("Starting DeepConf decoding")
    
    # Validate inputs
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    if not model or not tokenizer:
        raise ValueError("Model and tokenizer must be provided")
    
    # Merge configuration
    config = DEFAULT_CONFIG.copy()
    if request_config:
        # Validate and merge only known config keys
        valid_keys = set(DEFAULT_CONFIG.keys())
        for key, value in request_config.items():
            if key in valid_keys:
                config[key] = value
            else:
                logger.warning(f"Unknown configuration key ignored: {key}")
    
    # Log configuration
    logger.info(f"DeepConf configuration: variant={config['variant']}, "
               f"warmup_samples={config['warmup_samples']}, "
               f"max_traces={config['max_traces']}")
    
    try:
        # Initialize processor
        processor = DeepConfProcessor(model, tokenizer, config)
        
        # Process with online mode
        final_answer, stats = processor.process_online(messages)
        
        # Extract token usage
        total_tokens = stats.get('total_tokens_used', 0)
        
        # Format the response
        response = format_deepconf_response(final_answer, stats, config)
        
        logger.info(f"DeepConf decoding completed successfully. "
                   f"Total tokens: {total_tokens}, "
                   f"Traces: {stats['total_traces']}, "
                   f"Early terminations: {stats['early_terminations']}")
        
        return response, total_tokens
        
    except Exception as e:
        logger.error(f"DeepConf decoding failed: {str(e)}")
        raise RuntimeError(f"DeepConf processing error: {str(e)}") from e

def format_deepconf_response(answer: str, stats: Dict[str, Any], 
                           config: Dict[str, Any]) -> str:
    """
    Format the DeepConf response with optional statistics.
    
    Args:
        answer: The final answer from weighted voting
        stats: Processing statistics
        config: Configuration used
        
    Returns:
        Formatted response string
    """
    # Base response is just the answer
    response = answer.strip()
    
    # Optionally add statistics (for debugging)
    if config.get('include_stats', False):
        stats_text = f"""

DeepConf Statistics:
- Variant: {stats['variant']}
- Total traces: {stats['total_traces']} (warmup: {stats['warmup_traces']}, online: {stats['online_traces']})
- Early terminations: {stats['early_terminations']}
- Total tokens: {stats['total_tokens_used']}
- Confidence threshold: {stats['confidence_threshold']:.4f}
- Unique answers: {stats['num_unique_answers']}"""
        
        response += stats_text
    
    return response

def validate_deepconf_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize DeepConf configuration.
    
    Args:
        config: Input configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    validated = config.copy()
    
    # Validate variant
    if 'variant' in validated:
        if validated['variant'] not in ['low', 'high']:
            raise ValueError("variant must be 'low' or 'high'")
    
    # Validate numeric parameters
    numeric_params = {
        'warmup_samples': (1, 100),
        'max_traces': (1, 1000), 
        'window_size': (100, 10000),
        'top_k': (1, 100),
        'min_trace_length': (10, 10000),
        'max_tokens_per_trace': (100, 100000),
        'consensus_threshold': (0.5, 1.0),
        'temperature': (0.1, 2.0)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in validated:
            value = validated[param]
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                raise ValueError(f"{param} must be between {min_val} and {max_val}")
    
    # Ensure warmup_samples <= max_traces
    if (validated.get('warmup_samples', 0) >= validated.get('max_traces', 100)):
        raise ValueError("warmup_samples must be less than max_traces")
    
    return validated

def get_deepconf_info() -> Dict[str, Any]:
    """
    Get information about the DeepConf implementation.
    
    Returns:
        Dictionary with implementation details
    """
    return {
        "name": "DeepConf",
        "description": "Confidence-aware reasoning with early termination",
        "paper": "Deep Think with Confidence (Fu et al., 2024)",
        "arxiv": "https://arxiv.org/abs/2508.15260",
        "local_models_only": True,
        "modes": ["online"],
        "variants": ["low", "high"],
        "default_config": DEFAULT_CONFIG,
        "features": [
            "Token-level confidence scoring",
            "Early termination based on confidence",
            "Warmup phase for threshold calibration", 
            "Consensus-based stopping",
            "Weighted majority voting"
        ]
    }