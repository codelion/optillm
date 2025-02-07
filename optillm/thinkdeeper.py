import torch
import random
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_CONFIG = {
    "min_thinking_tokens": 512,
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>",
    
    # Combined thought transition markers and TIP configs
    "tip_alpha": 4.0,  # Penalty strength
    "tip_beta": 1024,   # Penalty duration (number of tokens)
    "thought_switch_tokens": [
        "Wait,",
        "Alternatively,",
    ],
}

class ThinkDeeperTIPProcessor:
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
        
        # Track when the last thought switch occurred
        self.last_thought_switch_pos = 0
        
    def adjust_logits_with_tip(self, logits: torch.Tensor, current_pos: int) -> torch.Tensor:
        """Apply Thought Switching Penalty (TIP) to logits"""
        tokens_since_last_switch = current_pos - self.last_thought_switch_pos
        
        if tokens_since_last_switch < self.config["tip_beta"]:
            penalty_mask = torch.zeros_like(logits)
            for token_id in self.thought_switch_tokens:
                if token_id < logits.size(-1):  # Ensure token_id is within valid range
                    penalty_mask[token_id] = self.config["tip_alpha"]
            
            adjusted_logits = logits - penalty_mask
            return adjusted_logits
        
        return logits

    @torch.inference_mode()
    def reasoning_effort(self, messages) -> str:
        """Generate response with ThinkDeeper + TIP"""
        
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
        current_pos = 0
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            
            # Apply TIP to logits
            logits = out.logits[0, -1, :]
            adjusted_logits = self.adjust_logits_with_tip(logits, current_pos)
            
            next_token = torch.multinomial(
                torch.softmax(adjusted_logits, dim=-1), 1
            ).item()
            kv = out.past_key_values
            
            next_str = self.tokenizer.decode([next_token])
            logger.debug(f"Generated token {next_token} -> '{next_str}'")

            # Check if this is a thought-switching token
            if next_token in self.thought_switch_tokens:
                self.last_thought_switch_pos = current_pos
                logger.debug(f"Detected thought switch at position {current_pos}")

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
                
                replacement = random.choice(self.config["thought_switch_tokens"])
                logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens}, seen_end_think: {seen_end_think})")
                response_chunks.append(replacement)
                replacement_tokens = self.tokenizer.encode(replacement)
                n_thinking_tokens += len(replacement_tokens)
                tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                seen_end_think = False
                logger.debug("Reset seen_end_think flag after replacement")
                
            elif next_token == self.model.config.eos_token_id and seen_end_think:
                logger.debug("Reached EOS after end think token - stopping generation")
                break
                
            else:
                response_chunks.append(next_str)
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                current_pos += 1
                logger.debug(f"Added token to response. Total thinking tokens: {n_thinking_tokens}")

        # Join all chunks and trim off the initial prompt
        response = "".join(response_chunks)
        full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response}"
        
        logger.debug(f"Final response length: {len(full_response)} chars")
        return full_response

def thinkdeeper_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Dict[str, Any] = None
) -> str:
    """Main plugin execution function with ThinkDeeper + TIP"""
    logger.info("Starting ThinkDeeper+TIP processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        thinkdeeper_config = request_config
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in thinkdeeper_config:
                config[key] = thinkdeeper_config[key]

    logger.info(f"Using config: {config}")
    
    try:
        processor = ThinkDeeperTIPProcessor(config, tokenizer, model)
        response = processor.reasoning_effort(messages)
        return response
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper+TIP processing: {str(e)}")
        raise