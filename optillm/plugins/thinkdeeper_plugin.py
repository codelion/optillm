"""
thinkdeeper_plugin.py - Plugin for enhanced thinking capabilities in optillm
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

# Plugin identifier
SLUG = "thinkdeeper"

# Default configurations
DEFAULT_CONFIG = {
    "replacements": ["\nWait, but", "\nHmm", "\nSo"],
    "min_thinking_tokens": 128,
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>"
}

logger = logging.getLogger(__name__)

class ThinkDeeperProcessor:
    def __init__(self, config: Dict[str, Any], tokenizer, model):
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model

        # Get the actual token IDs for think markers
        start_tokens = self.tokenizer.encode(self.config['start_think_token'])
        end_tokens = self.tokenizer.encode(self.config['end_think_token'])
        # Take the last token of start marker and first token of end marker 
        # in case they get split into multiple tokens
        self._start_think_token = start_tokens[-1]
        self.end_think_token = end_tokens[0]

    @torch.inference_mode()
    def reasoning_effort(self, question: str) -> str:
        """
        Generate an enhanced thinking response with extended reasoning.
        
        Args:
            question: The input question to process
            
        Returns:
            The generated response with enhanced thinking
        """
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"},
            ],
            continue_final_message=True,
            return_tensors="pt"
        )
        tokens = tokens.to(self.model.device)
        kv = DynamicCache()
        n_thinking_tokens = 0
        
        # Store response chunks
        response_chunks = [self.tokenizer.decode(list(tokens[0]))]
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            next_token = torch.multinomial(
                torch.softmax(out.logits[0, -1, :], dim=-1), 1
            ).item()
            kv = out.past_key_values

            if (
                next_token in (self.end_think_token, self.model.config.eos_token_id)
                and n_thinking_tokens < self.config["min_thinking_tokens"]
            ):
                replacement = random.choice(self.config["replacements"])
                response_chunks.append(replacement)
                replacement_tokens = self.tokenizer.encode(replacement)
                n_thinking_tokens += len(replacement_tokens)
                tokens = torch.tensor([replacement_tokens]).to(tokens.device)
            elif next_token == self.model.config.eos_token_id:
                break
            else:
                response_chunks.append(self.tokenizer.decode([next_token]))
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)

        return "".join(response_chunks)

def run(system_prompt: str, initial_query: str, client, model: str, request_config: Dict[str, Any] = None) -> Tuple[str, int]:
    """
    Main plugin execution function.
    
    Args:
        system_prompt: System prompt text
        initial_query: Query to process
        client: OpenAI client instance
        model: Model identifier
        request_config: Additional configuration from the request
        
    Returns:
        Tuple of (generated response, completion tokens)
    """
    logger.info("Starting ThinkDeeper processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        thinkdeeper_config = request_config.get("thinkdeeper_config", {})
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in thinkdeeper_config:
                config[key] = thinkdeeper_config[key]
    
    try:
         # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        
        # Create processor and generate response
        processor = ThinkDeeperProcessor(config, tokenizer, model)
        response = processor.reasoning_effort(initial_query)
        
        # Calculate actual completion tokens
        completion_tokens = len(processor.tokenizer.encode(response))
        
        return response, completion_tokens
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper processing: {str(e)}")
        # Fallback to standard response
        return f"Error in enhanced thinking process: {str(e)}", 0
