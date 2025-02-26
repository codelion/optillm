import torch
import random
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy from logits tensor
    
    Args:
        logits: Raw logits from model output (pre-softmax)
    
    Returns:
        float: Entropy value
    """
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate entropy: -sum(p_i * log(p_i))
    # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    
    return entropy.item()

class EntropyTracker:
    """Tracks entropy over time and provides analysis capabilities"""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize entropy tracker
        
        Args:
            window_size: Number of tokens to track for moving averages
        """
        self.entropy_history = []
        self.window_size = window_size
        self.transition_entropies = {
            "before": {},
            "after": {}
        }
    
    def add_entropy(self, entropy: float) -> None:
        """Add entropy value to history"""
        self.entropy_history.append(entropy)
    
    def get_recent_avg_entropy(self) -> float:
        """Get average entropy over recent window"""
        if len(self.entropy_history) == 0:
            return 0.0
        
        window = self.entropy_history[-min(self.window_size, len(self.entropy_history)):]
        return sum(window) / len(window)
    
    def record_transition_entropy(self, transition_word: str, before: bool = True) -> None:
        """
        Record entropy around a transition word
        
        Args:
            transition_word: The transition word being tracked
            before: True if recording before transition, False if after
        """
        key = "before" if before else "after"
        if transition_word not in self.transition_entropies[key]:
            self.transition_entropies[key][transition_word] = []
            
        current_entropy = self.get_recent_avg_entropy()
        self.transition_entropies[key][transition_word].append(current_entropy)
    
    def get_entropy_change(self, transition_word: str) -> float:
        """
        Get entropy change for a specific transition word
        
        Args:
            transition_word: The transition to analyze
            
        Returns:
            float: Average entropy change (after - before)
        """
        if (transition_word not in self.transition_entropies["before"] or
            transition_word not in self.transition_entropies["after"]):
            return 0.0
            
        before_values = self.transition_entropies["before"][transition_word]
        after_values = self.transition_entropies["after"][transition_word]
        
        # Use only entries that have both before and after
        count = min(len(before_values), len(after_values))
        if count == 0:
            return 0.0
            
        before_avg = sum(before_values[:count]) / count
        after_avg = sum(after_values[:count]) / count
        
        return after_avg - before_avg
    
class InterventionHandler:
    """Handles detection and injection of verification prompts"""
    
    def __init__(self, entropy_thresholds: Dict[str, float] = None):
        """
        Initialize intervention handler
        
        Args:
            entropy_thresholds: Dict mapping transition words to threshold values
        """
        # Default entropy thresholds based on the logit analysis
        self.entropy_thresholds = {
            "However,": 1.45,
            "Wait,": 1.50,
            "Alternatively,": 1.45,
            "Additionally,": 1.40,
            # Default threshold for any other transition
            "default": 1.50
        }
        
        # Update with any user-provided thresholds
        if entropy_thresholds:
            self.entropy_thresholds.update(entropy_thresholds)
            
        # Track interventions to avoid repeating too frequently
        self.last_intervention_token = 0
        self.current_token_pos = 0
        self.min_tokens_between_interventions = 50
        
        # Verification prompts tailored to different transition words
        self.verification_prompts = {
            "However,": " Let me carefully verify if this contrary point actually changes my conclusion.",
            "Wait,": " Let me double-check my previous calculation before changing direction.",
            "Alternatively,": " Let me evaluate if this alternative approach is consistent with my previous reasoning.",
            "Additionally,": " Let me verify that this additional information is correctly incorporated.",
            # Default prompt
            "default": " Let me verify my reasoning step by step before continuing."
        }
    
    def increment_token_pos(self):
        """Increment current token position counter"""
        self.current_token_pos += 1
    
    def should_intervene(self, 
                         transition_word: str, 
                         current_entropy: float,
                         tokens_generated: int,
                         max_tokens: int) -> bool:
        """
        Determine if we should intervene based on current conditions
        
        Args:
            transition_word: The detected transition word
            current_entropy: Current entropy value
            tokens_generated: Number of tokens generated so far
            max_tokens: Maximum tokens allowed
            
        Returns:
            bool: True if should intervene, False otherwise
        """
        # Get appropriate threshold
        threshold = self.entropy_thresholds.get(transition_word, self.entropy_thresholds["default"])
        
        # Check if entropy exceeds threshold
        entropy_condition = current_entropy > threshold
        
        # Check if we're in the middle-to-late reasoning phase (40-80% of generation)
        generation_progress = tokens_generated / max_tokens if max_tokens > 0 else 0.5
        progress_condition = 0.4 < generation_progress < 0.8
        
        # Ensure we don't intervene too frequently
        frequency_condition = (self.current_token_pos - self.last_intervention_token) > self.min_tokens_between_interventions
        
        # Determine if we should intervene
        should_intervene = entropy_condition and progress_condition and frequency_condition
        
        if should_intervene:
            self.last_intervention_token = self.current_token_pos
            
        return should_intervene
    
    def get_verification_prompt(self, transition_word: str) -> str:
        """
        Get appropriate verification prompt for the transition word
        
        Args:
            transition_word: The transition word to get prompt for
            
        Returns:
            str: The verification prompt
        """
        return self.verification_prompts.get(transition_word, self.verification_prompts["default"])
    

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
        
        # Store thought switch markers as token sequences and their decoded forms
        self.thought_switch_sequences = []
        self.thought_switch_phrases = []
        for phrase in self.config["thought_switch_tokens"]:
            # Encode without adding special tokens to get exact sequence
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            self.thought_switch_sequences.append(token_ids)
            self.thought_switch_phrases.append(phrase)
            logger.debug(f"Encoded '{phrase}' to token sequence: {token_ids}")
            logger.debug(f"Decoded back: {self.tokenizer.decode(token_ids)}")
        
        # Track thought switches
        self.thought_count = 0
        self.current_sequence = []  # Track recent tokens for sequence matching
        self.max_sequence_length = max(len(seq) for seq in self.thought_switch_sequences)

        # Initialize entropy tracking
        self.entropy_tracker = EntropyTracker()
        
        # Initialize intervention handler
        entropy_thresholds = self.config.get("entropy_thresholds", None)
        self.intervention_handler = InterventionHandler(entropy_thresholds)
        
        # Track if we're currently in an intervention
        self.in_intervention = False
        self.current_intervention_tokens = []
        
        # Map token sequences to their phrases
        self.sequence_to_phrase = {}
        for phrase, sequence in zip(self.thought_switch_phrases, self.thought_switch_sequences):
            seq_tuple = tuple(sequence)
            self.sequence_to_phrase[seq_tuple] = phrase
            logger.debug(f"Thought switch marker '{phrase}' encoded as: {sequence}")
            logger.debug(f"Decoded back as: {self.tokenizer.decode(sequence)}")

    def is_thought_switch(self, token: int) -> Tuple[bool, Optional[str]]:
        """
        Check if adding this token creates a thought switch sequence.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_switch, transition_phrase)
        """
        # Add new token to current sequence
        self.current_sequence.append(token)
        
        # Keep only the most recent tokens that could match our sequences
        if len(self.current_sequence) > self.max_sequence_length:
            self.current_sequence = self.current_sequence[-self.max_sequence_length:]
        
        # Check if current sequence ends with any thought switch sequence
        for i, sequence in enumerate(self.thought_switch_sequences):
            if len(sequence) <= len(self.current_sequence) and \
               self.current_sequence[-len(sequence):] == sequence:
                return True, self.thought_switch_phrases[i]
        
        return False, None
        
    @torch.inference_mode()
    def reasoning_effort(self, messages) -> str:
        """Generate response with ThinkDeeper's controlled thinking process and entropy-based interventions"""
        
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
        
        # Reset tracking for new generation
        self.thought_count = 0
        self.current_sequence = []
        self.in_intervention = False
        self.current_intervention_tokens = []
        self.intervention_handler.last_intervention_token = 0
        self.intervention_handler.current_token_pos = 0
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            logits = out.logits[0, -1, :]
            
            # Calculate entropy and update tracker
            current_entropy = calculate_entropy(logits)
            self.entropy_tracker.add_entropy(current_entropy)
            
            # Update the token position counter
            self.intervention_handler.increment_token_pos()
            
            # Check if we need to force end thinking
            force_end = (n_thinking_tokens >= self.config["max_thinking_tokens"] or 
                        self.thought_count >= self.config["max_thoughts"])
            
            if force_end and not seen_end_think:
                logger.debug(f"Forcing end think token. Tokens: {n_thinking_tokens}, Thoughts: {self.thought_count}")
                next_token = self.end_think_token
                response_chunks.append(self.tokenizer.decode([next_token]))
                seen_end_think = True
                # Don't break - continue generating but with end_think token forced
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                continue
            else:
                # If we're in an intervention, continue with it
                if self.in_intervention and self.current_intervention_tokens:
                    next_token = self.current_intervention_tokens.pop(0)
                    logger.debug(f"Continuing intervention with token: {self.tokenizer.decode([next_token])}")
                    
                    # If we're done with the intervention, mark it as complete
                    if not self.current_intervention_tokens:
                        self.in_intervention = False
                        logger.debug("Intervention complete")
                else:
                    # Normal generation
                    next_token = torch.multinomial(
                        torch.softmax(logits, dim=-1), 1
                    ).item()
            
            kv = out.past_key_values
            next_str = self.tokenizer.decode([next_token])
            
            # Check if this is a thought-switching token (only if not in conclusion phase)
            if not seen_end_think:
                is_switch, transition_word = self.is_thought_switch(next_token)
                if is_switch:
                    # Record entropy before transition
                    self.entropy_tracker.record_transition_entropy(transition_word, before=True)
                    
                    self.thought_count += 1
                    logger.debug(f"Detected thought switch marker '{transition_word}'. Total thoughts: {self.thought_count}")
                    
                    # Decide if we should intervene at this transition
                    should_intervene = self.intervention_handler.should_intervene(
                        transition_word,
                        current_entropy,
                        n_thinking_tokens,
                        self.config["max_thinking_tokens"]
                    )
                    
                    if should_intervene and not self.in_intervention:
                        # Get verification prompt
                        verification_prompt = self.intervention_handler.get_verification_prompt(transition_word)
                        logger.debug(f"Intervening after '{transition_word}' with prompt: {verification_prompt}")
                        
                        # Tokenize the verification prompt and set up for injection
                        verification_tokens = self.tokenizer.encode(verification_prompt, add_special_tokens=False)
                        self.current_intervention_tokens = verification_tokens
                        self.in_intervention = True
                        
                        # Add the verification prompt to response
                        response_chunks.append(verification_prompt)
                        
                    # Record entropy after transition (will be in next token, but this helps with tracking)
                    self.entropy_tracker.record_transition_entropy(transition_word, before=False)

            # Handle natural end think token
            if next_token == self.end_think_token:
                seen_end_think = True
                logger.debug("Found end think token")
                
                # If we haven't reached minimum tokens, continue with thought transition
                if n_thinking_tokens < self.config["min_thinking_tokens"]:
                    replacement = random.choice(self.config["thought_switch_tokens"])
                    logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens})")
                    response_chunks.append(replacement)
                    replacement_tokens = self.tokenizer.encode(replacement)
                    n_thinking_tokens += len(replacement_tokens)
                    tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                    self.thought_count += 1
                    seen_end_think = False
                    continue

            # Handle EOS token
            if next_token == self.model.config.eos_token_id:
                logger.debug("Found eos token")
                if seen_end_think:
                    logger.debug("Reached EOS after end think token - stopping generation")
                    response_chunks.append(next_str)
                    break
                elif n_thinking_tokens < self.config["min_thinking_tokens"]:
                    # Continue with thought transition if under minimum tokens
                    replacement = random.choice(self.config["thought_switch_tokens"])
                    logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens})")
                    response_chunks.append(replacement)
                    replacement_tokens = self.tokenizer.encode(replacement)
                    n_thinking_tokens += len(replacement_tokens)
                    tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                    self.thought_count += 1
                    continue
                else:
                    # Force end think token and continue generating for natural conclusion
                    logger.debug("Reached EOS without end think token - adding end token and continuing generation")
                    response_chunks.append(self.tokenizer.decode([self.end_think_token]))
                    tokens = torch.tensor([[self.end_think_token]]).to(tokens.device)
                    seen_end_think = True
                    continue
            
            # Skip adding token to response if we're injecting an intervention
            if not self.in_intervention or not self.current_intervention_tokens:
                response_chunks.append(next_str)
                
            if not seen_end_think:
                n_thinking_tokens += 1
                
            # Set up next token
            if self.in_intervention and self.current_intervention_tokens:
                # Next token is from intervention
                tokens = torch.tensor([[self.current_intervention_tokens[0]]]).to(tokens.device)
            else:
                # Normal next token
                tokens = torch.tensor([[next_token]]).to(tokens.device)

        # Join all chunks and add framing tokens
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