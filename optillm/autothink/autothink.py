"""
AutoThink main implementation.

This module provides the main implementation of AutoThink, combining
query complexity classification with steering vectors to enhance reasoning.
"""

import logging
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .processor import AutoThinkProcessor as InternalProcessor

logger = logging.getLogger(__name__)

class AutoThinkProcessor:
    """
    Main AutoThink processor class for external use.
    Wraps the internal processor implementation.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, Any] = None):
        """
        Initialize the AutoThink processor.
        
        Args:
            model: Language model
            tokenizer: Model tokenizer
            config: Configuration dictionary
        """
        self.config = config or {}
        self.processor = None
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Process messages with AutoThink's controlled thinking."""
        return self.process(messages)
        
    def process(self, messages: List[Dict[str, str]]) -> str:
        """Process messages with AutoThink's controlled thinking.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated response
        """
        # Create processor on first use to allow for model loading
        if self.processor is None:
            self.processor = self._create_processor()
        
        return self.processor.process(messages)
    
    def _create_processor(self):
        """Create the internal processor instance."""
        return InternalProcessor(self.config, self.tokenizer, self.model)

def autothink_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main plugin execution function with AutoThink's controlled thinking process.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        messages: List of message dictionaries
        request_config: Optional configuration dictionary
        
    Returns:
        Generated response with thinking process
    """
    logger.info("Starting AutoThink processing")
    
    # Create config dictionary
    config = {}
    if request_config:
        config.update(request_config)
    
    try:
        processor = AutoThinkProcessor(model, tokenizer, config)
        response = processor.process(messages)
        return response
        
    except Exception as e:
        logger.error(f"Error in AutoThink processing: {str(e)}")
        raise
