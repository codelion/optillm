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
        "wait",
        "alternatively",
        "instead",
        "however",
    ],
    
    # N-best trace generation settings
    "num_traces": 2,  # Number of thinking traces to generate
}

class ThinkDeeperTIPProcessor:
    def __init__(self, config: Dict[str, Any], tokenizer, model):
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model
        
        # Get token IDs for think markers
        tokens = self.tokenizer.encode(f"{self.config['start_think_token']}{self.config['end_think_token']}")
        self._start_think_token = tokens[1]
        self.end_think_token = tokens[2]
        
        # Get token IDs for thought switching indicators
        self.thought_switch_tokens = set()
        for phrase in self.config["thought_switch_tokens"]:
            token_ids = self.tokenizer.encode(phrase)[1:]  # Skip the first token which might be special
            self.thought_switch_tokens.update(token_ids)
        
        # Track when the last thought switch occurred
        self.last_thought_switch_pos = 0
        
    def adjust_logits_with_tip(self, logits: torch.Tensor, current_pos: int) -> torch.Tensor:
        """Apply Thought Switching Penalty (TIP) to logits"""
        tokens_since_last_switch = current_pos - self.last_thought_switch_pos
        
        if tokens_since_last_switch < self.config["tip_beta"]:
            penalty_mask = torch.zeros_like(logits)
            for token_id in self.thought_switch_tokens:
                penalty_mask[token_id] = self.config["tip_alpha"]
            
            adjusted_logits = logits - penalty_mask
            return adjusted_logits
        
        return logits

    def compute_reward(self, trace: str, token_count: int) -> float:
        """Compute reward for a thinking trace - currently rewards shorter traces"""
        return -token_count  # Negative because shorter is better
    
    def generate_thinking_trace(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Generate a single thinking trace"""

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

            if next_token in self.thought_switch_tokens:
                self.last_thought_switch_pos = current_pos

            if next_token == self.end_think_token:
                seen_end_think = True

            if next_token == self.end_think_token:
                if n_thinking_tokens >= self.config["min_thinking_tokens"]:
                    # We have enough tokens and found </think>, we can stop
                    response_chunks.append(next_str)
                    break
                else:
                    # Need more tokens, continue with replacement
                    replacement = random.choice(self.config["thought_switch_tokens"])
                    response_chunks.append(replacement)
                    replacement_tokens = self.tokenizer.encode(replacement)
                    n_thinking_tokens += len(replacement_tokens)
                    tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                
            else:
                response_chunks.append(next_str)
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                current_pos += 1

        response = "".join(response_chunks)
        return response, n_thinking_tokens

    def generate_final_response(self, messages: List[Dict[str, str]], selected_trace: str) -> str:
        """Generate final response using the selected thinking trace"""
        messages.pop()
        messages.append({"role": "assistant", "content": selected_trace})
        
        tokens = self.tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
            return_tensors="pt"
        )
        tokens = tokens.to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(outputs[0][len(tokens[0]):], skip_special_tokens=True)

    @torch.inference_mode()
    def reasoning_effort(self, messages: List[Dict[str, str]]) -> str:
        """Generate multiple thinking traces and select the best one"""
        messages.append({"role": "assistant", "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"})
        # Generate n thinking traces
        traces_with_rewards = []
        for i in range(self.config["num_traces"]):
            logger.info(f"Generating thinking trace {i+1}/{self.config['num_traces']}")
            trace, token_count = self.generate_thinking_trace(messages)
            reward = self.compute_reward(trace, token_count)
            traces_with_rewards.append((trace, reward))
            logger.info(f"Trace {i+1} generated with reward {reward}")
        
        # Select the best trace
        best_trace, best_reward = max(traces_with_rewards, key=lambda x: x[1])
        logger.info(f"Selected best trace with reward {best_reward}")
        
        # Generate final response using the best trace
        final_response = self.generate_final_response(messages, best_trace)
        
        return final_response

def thinkdeeper_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Dict[str, Any] = None
) -> str:
    """Main plugin execution function with ThinkDeeper + TIP and n-best selection"""
    logger.info("Starting ThinkDeeper+TIP processing with n-best selection")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        thinkdeeper_config = request_config
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in thinkdeeper_config:
                config[key] = thinkdeeper_config[key]

    logger.info(config)
    
    try:
        processor = ThinkDeeperTIPProcessor(config, tokenizer, model)
        response = processor.reasoning_effort(messages)
        return response
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper+TIP processing: {str(e)}")
        raise