import torch
import random
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "min_thinking_tokens": 512,
    "max_thinking_tokens": 2048,  # New parameter to cap thinking length
    "max_thoughts": 4,  # New parameter to limit number of thought transitions
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>",
    "thought_switch_tokens": [
        "Wait,",
        "Alternatively,",
    ],
}

class ThinkDeeperProcessor:
    def __init__(self, config: Dict[str, Any], tokenizer, model):
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model
        
        # Get token IDs for think markers
        start_tokens = self.tokenizer.encode(self.config['start_think_token'])
        end_tokens = self.tokenizer.encode(self.config['end_think_token'])
        self._start_think_token = start_tokens[0] if len(start_tokens) == 1 else start_tokens[1]
        self.end_think_token = end_tokens[0] if len(end_tokens) == 1 else end_tokens[1]
        
        # Get token IDs for thought switching indicators
        self.thought_switch_tokens = set()
        for phrase in self.config["thought_switch_tokens"]:
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            self.thought_switch_tokens.update(token_ids)
        
        # Track thought switches
        self.thought_count = 0
        
    @torch.inference_mode()
    def reasoning_effort(self, messages) -> str:
        """Generate response with ThinkDeeper's controlled thinking process"""
        
        messages.append({"role": "assistant", "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"})

        tokens = self.tokenizer.apply_chat_template(
            messages,
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
            logits = out.logits[0, -1, :]
            
            # Force end think token if we exceed limits
            if (n_thinking_tokens >= self.config["max_thinking_tokens"] or 
                self.thought_count >= self.config["max_thoughts"]):
                next_token = self.end_think_token
                logger.debug(f"Forcing end think token. Tokens: {n_thinking_tokens}, Thoughts: {self.thought_count}")
            else:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                ).item()
            
            kv = out.past_key_values
            next_str = self.tokenizer.decode([next_token])
            
            # Check if this is a thought-switching token
            if next_token in self.thought_switch_tokens:
                self.thought_count += 1
                logger.debug(f"Detected thought switch. Total thoughts: {self.thought_count}")

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
                
                # Insert thought transition
                replacement = random.choice(self.config["thought_switch_tokens"])
                logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens})")
                response_chunks.append(replacement)
                replacement_tokens = self.tokenizer.encode(replacement)
                n_thinking_tokens += len(replacement_tokens)
                tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                self.thought_count += 1
                seen_end_think = False
                
            elif next_token == self.model.config.eos_token_id and seen_end_think:
                logger.debug("Reached EOS after end think token - stopping generation")
                break
                
            else:
                response_chunks.append(next_str)
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)

        # Join all chunks and trim off the initial prompt
        response = "".join(response_chunks)
        full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response}"
        
        logger.debug(f"Final response length: {len(full_response)} chars, Total thoughts: {self.thought_count}")
        return full_response

def thinkdeeper_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Dict[str, Any] = None
) -> str:
    """Main plugin execution function with ThinkDeeper's controlled thinking process"""
    logger.info("Starting ThinkDeeper processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in request_config:
                config[key] = request_config[key]

    logger.info(f"Using config: {config}")
    
    try:
        processor = ThinkDeeperProcessor(config, tokenizer, model)
        response = processor.reasoning_effort(messages)
        return response
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper processing: {str(e)}")
        raise