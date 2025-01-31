import torch
import random
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
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
        tokens = self.tokenizer.encode(f"{self.config['start_think_token']}{self.config['end_think_token']}")
        self._start_think_token = tokens[1]  # Start token is second token
        self.end_think_token = tokens[2]     # End token is third token
        logger.debug(f"Think token IDs - Start: {self._start_think_token} ({self.tokenizer.decode([self._start_think_token])}), End: {self.end_think_token} ({self.tokenizer.decode([self.end_think_token])})")
        

    @torch.inference_mode()
    def reasoning_effort(self, question: str) -> str:
        """
        Generate an enhanced thinking response with extended reasoning.
        
        Args:
            question: The input question to process
            
        Returns:
            The generated response with enhanced thinking
        """
        logger.debug(f"Starting generation for question: {question}")
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
        seen_end_think = False
        response_chunks = []
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            next_token = torch.multinomial(
                torch.softmax(out.logits[0, -1, :], dim=-1), 1
            ).item()
            kv = out.past_key_values
            
            next_str = self.tokenizer.decode([next_token])
            logger.debug(f"Generated token {next_token} -> '{next_str}'")

            # Track if we've seen the end think token
            if next_token == self.end_think_token:
                seen_end_think = True
                logger.debug("Found end think token")

            # Need to continue generating if:
            # 1. We hit end think/eos before min tokens OR
            # 2. We hit eos without seeing end think token
            if ((next_token in (self.end_think_token, self.model.config.eos_token_id) 
                 and n_thinking_tokens < self.config["min_thinking_tokens"]) 
                or (next_token == self.model.config.eos_token_id and not seen_end_think)):
                
                replacement = random.choice(self.config["replacements"])
                logger.debug(f"Inserting replacement: '{replacement}' (tokens: {n_thinking_tokens}, seen_end_think: {seen_end_think})")
                response_chunks.append(replacement)
                replacement_tokens = self.tokenizer.encode(replacement)
                n_thinking_tokens += len(replacement_tokens)
                tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                # Reset seen_end_think as we're starting a new thinking sequence
                seen_end_think = False
                logger.debug("Reset seen_end_think flag after replacement")
                
                
            elif next_token == self.model.config.eos_token_id and seen_end_think:
                logger.debug("Reached EOS after end think token - stopping generation")
                break
                
            else:
                response_chunks.append(next_str)
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                logger.debug(f"Added token to response. Total thinking tokens: {n_thinking_tokens}")

        # Join all chunks and trim off the initial prompt
        response = "".join(response_chunks)
        full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response}"
        
        logger.debug(f"Final response length: {len(full_response)} chars")
        return full_response

def thinkdeeper_decode(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], request_config: Dict[str, Any] = None) -> Tuple[str, int]:
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
    device = get_device()
    model.to(device)

    # Use the chat template to format the input
    if tokenizer.chat_template:
        initial_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback for tokenizers without chat templates
        initial_query = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        thinkdeeper_config = request_config.get("thinkdeeper_config", {})
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in thinkdeeper_config:
                config[key] = thinkdeeper_config[key]
    
    try:      
        
        # Create processor and generate response
        processor = ThinkDeeperProcessor(config, tokenizer, model)
        response = processor.reasoning_effort(initial_query)
        
        # Calculate actual completion tokens
        completion_tokens = len(tokenizer.encode(response))
        logger.info(f"Generation complete. Used {completion_tokens} completion tokens")
        
        return response, completion_tokens
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper processing: {str(e)}")
        # Fallback to standard response
        return f"Error in enhanced thinking process: {str(e)}", 0
