import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
import bitsandbytes as bnb
from scipy.stats import entropy
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    base_model_id: str
    adapter_ids: Optional[List[str]] = None
    batch_size: int = 32
    max_cache_size: int = 5
    quantization_bits: int = 4
    device_preference: Optional[str] = None
    # Default generation parameters
    max_new_tokens: int = 512
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.7
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    # Advanced parameters
    use_memory_efficient_attention: bool = True
    enable_prompt_caching: bool = True
    dynamic_temperature: bool = True

class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention using linear attention mechanism.
    Supports automatic fallback to optimized implementations when available.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard projections with bias
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Try to import optimized attention implementations
        self.optimized_attention = None
        
        # Try Flash Attention
        try:
            from flash_attn import flash_attn_func
            self.optimized_attention = flash_attn_func
            print("Using Flash Attention")
        except ImportError:
            pass
            
        # Try xFormers
        if self.optimized_attention is None:
            try:
                import xformers.ops as xops
                self.optimized_attention = xops.memory_efficient_attention
                print("Using xFormers attention")
            except ImportError:
                pass

    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Implements linear attention for more memory efficiency"""
        # Scale query
        q = q * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            # Convert boolean mask to float mask if needed
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float()
            k = k * attention_mask.unsqueeze(-1)
        
        if causal:
            # Handle causal attention
            batch_size, num_heads, seq_length, head_dim = q.shape
            positions = torch.arange(seq_length, device=q.device)
            causal_mask = positions.view(1, 1, -1, 1) <= positions.view(1, 1, 1, -1)
            k = k * causal_mask.float()
        
        # Linear attention computation
        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Try using optimized attention if available
        if self.optimized_attention is not None and hidden_states.device.type == 'cuda':
            # Handle attention mask for optimized implementations
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask > 0
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            
            try:
                attn_output = self.optimized_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    causal=causal,
                    scale=self.scale
                )
            except Exception as e:
                print(f"Optimized attention failed, falling back to linear attention: {e}")
                attn_output = self._linear_attention(q, k, v, attention_mask, causal)
        else:
            # Use linear attention for CPU/MPS or when optimized attention is not available
            attn_output = self._linear_attention(q, k, v, attention_mask, causal)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class PromptCache:
    """Advanced caching system for frequent prompts and responses"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.prompt_stats = defaultdict(lambda: {"count": 0, "success_rate": 0.0})
        
    @lru_cache(maxsize=128)
    def _compute_prompt_signature(self, prompt: str) -> str:
        """Compute a signature for semantic similarity matching"""
        # Simple but effective signature based on key content words
        words = set(prompt.lower().split())
        return " ".join(sorted(list(words)))
    
    def get_cached_response(self, prompt: str, temperature: float, top_p: float) -> Optional[str]:
        """Get cached response with fuzzy matching"""
        signature = self._compute_prompt_signature(prompt)
        
        if signature in self.cache:
            cached_item = self.cache[signature]
            if abs(cached_item["temperature"] - temperature) < 0.1 and abs(cached_item["top_p"] - top_p) < 0.1:
                self.prompt_stats[signature]["count"] += 1
                return cached_item["response"]
        
        return None
    
    def add_to_cache(self, prompt: str, response: str, temperature: float, top_p: float):
        """Add response to cache with metadata"""
        signature = self._compute_prompt_signature(prompt)
        
        self.cache[signature] = {
            "response": response,
            "temperature": temperature,
            "top_p": top_p,
            "timestamp": torch.cuda.current_timestamp() if torch.cuda.is_available() else 0
        }
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def update_stats(self, prompt: str, success: bool):
        """Update prompt success statistics"""
        signature = self._compute_prompt_signature(prompt)
        stats = self.prompt_stats[signature]
        stats["count"] += 1
        stats["success_rate"] = (stats["success_rate"] * (stats["count"] - 1) + float(success)) / stats["count"]

class DynamicTemperature:
    """Implements dynamic temperature scaling based on input characteristics"""
    
    def __init__(self):
        self.token_entropy_cache = {}
    
    def _compute_token_entropy(self, tokens: List[int]) -> float:
        """Compute token distribution entropy"""
        token_counts = np.bincount(tokens)
        probabilities = token_counts / len(tokens)
        return entropy(probabilities)
    
    def get_optimal_temperature(self, prompt: str, tokenizer: AutoTokenizer, base_temperature: float) -> float:
        """Calculate optimal temperature based on prompt characteristics"""
        tokens = tokenizer.encode(prompt)
        
        # Calculate entropy-based scaling
        token_entropy = self._compute_token_entropy(tokens)
        
        # Scale temperature based on prompt entropy and length
        length_factor = np.clip(len(tokens) / 100, 0.5, 2.0)
        entropy_factor = np.clip(token_entropy / 4.0, 0.5, 1.5)
        
        optimal_temperature = base_temperature * length_factor * entropy_factor
        return np.clip(optimal_temperature, 0.1, 2.0)

class CacheManager:
    """Enhanced cache manager with advanced features"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.model_cache = OrderedDict()
        self.adapter_cache = OrderedDict()
        self.prompt_cache = PromptCache()
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
    
    def _cleanup_cache(self, cache: OrderedDict):
        while len(cache) > self.max_size:
            _, model = cache.popitem(last=False)
            if hasattr(model, 'cpu'):
                model.cpu()
            torch.cuda.empty_cache()
    
    def get_or_load_model(self, model_key: str, loader_fn) -> Any:
        """Get or load model with enhanced caching"""
        if model_key in self.model_cache:
            self.cache_stats[model_key]["hits"] += 1
            self.model_cache.move_to_end(model_key)
            return self.model_cache[model_key]
        
        self.cache_stats[model_key]["misses"] += 1
        model = loader_fn()
        self.model_cache[model_key] = model
        self._cleanup_cache(self.model_cache)
        return model
    
    def get_or_load_adapter(self, adapter_key: str, loader_fn):
        if adapter_key in self.adapter_cache:
            self.adapter_cache.move_to_end(adapter_key)
            return self.adapter_cache[adapter_key]
        
        adapter = loader_fn()
        self.adapter_cache[adapter_key] = adapter
        self._cleanup_cache(self.adapter_cache)
        return adapter

class DeviceManager:
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.device_stats = {device: {'memory_used': 0, 'active_models': 0} for device in self.available_devices}

    def _detect_devices(self) -> List[str]:
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if torch.backends.mps.is_available():
            devices.append('mps')
        return devices

    def get_optimal_device(self, model_size: int = 0) -> str:
        if not self.available_devices:
            return 'cpu'

        # Prefer CUDA devices if available
        cuda_devices = [d for d in self.available_devices if 'cuda' in d]
        if cuda_devices:
            # Find CUDA device with most free memory
            max_free_memory = 0
            optimal_device = cuda_devices[0]
            
            for device in cuda_devices:
                idx = int(device.split(':')[1])
                free_memory = torch.cuda.get_device_properties(idx).total_memory - torch.cuda.memory_allocated(idx)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    optimal_device = device
            
            return optimal_device
        
        # Fall back to MPS if available
        if 'mps' in self.available_devices:
            return 'mps'
        
        return 'cpu'

    def track_device_usage(self, device: str, memory_delta: int):
        if device in self.device_stats:
            self.device_stats[device]['memory_used'] += memory_delta

class ModelManager:
    def __init__(self, cache_manager: CacheManager, device_manager: DeviceManager):
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        
    def quantize_model(self, model):
        """Quantize model to 4-bit precision using bitsandbytes"""
        def _replace_linear_layers(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16
                    ))
                else:
                    _replace_linear_layers(child)
                    
        _replace_linear_layers(model)
        return model

    def load_base_model(self, model_id: str, quantize: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        def _load_model():
            logger.info(f"Loading base model: {model_id}")
            
            # Determine optimal device
            device = self.device_manager.get_optimal_device()
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load model with quantization and device mapping
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto' if 'cuda' in device else device,
                trust_remote_code=True
            )
            
            if quantize and 'cuda' in device:
                model = self.quantize_model(model)
            
            return model, tokenizer

        return self.cache_manager.get_or_load_model(model_id, _load_model)

class LoRAManager:
    """LoRA manager with enhanced error handling"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.loaded_adapters = {}  # Maps model -> list of loaded adapter_ids

    def validate_adapter(self, adapter_id: str) -> bool:
        """Validate if adapter exists and is compatible"""
        try:
            # First check if adapter config exists
            config = PeftConfig.from_pretrained(
                adapter_id,
                trust_remote_code=True,
                use_auth_token=os.getenv("HF_TOKEN")  # Support private repos
            )
            return True
        except Exception as e:
            logger.error(f"Error validating adapter {adapter_id}: {str(e)}")
            return False

    def load_adapter(self, base_model: PreTrainedModel, adapter_id: str) -> PreTrainedModel:
        """Load a LoRA adapter with enhanced error handling"""
        def _load_adapter():
            logger.info(f"Loading LoRA adapter: {adapter_id}")
            
            # Validate adapter before loading
            if not self.validate_adapter(adapter_id):
                error_msg = f"Adapter {adapter_id} not found or is not compatible"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                
                # Load adapter into existing PeftModel
                model = base_model
                model.load_adapter(
                    adapter_id,
                    token=os.getenv("HF_TOKEN"),
                )
                
                # Track loaded adapter
                if model not in self.loaded_adapters:
                    self.loaded_adapters[model] = []
                self.loaded_adapters[model].append(adapter_id)
                
                return model
            
            except Exception as e:
                # Provide more detailed error message
                error_msg = f"Failed to load adapter {adapter_id}: {str(e)}"
                logger.error(error_msg)
                
                # Check common issues
                if "not found" in str(e).lower():
                    error_msg += "\nPossible causes:\n" \
                               "1. Adapter ID is incorrect\n" \
                               "2. Adapter is in a private repository (set HF_TOKEN)\n" \
                               "3. No internet connection"
                elif "incompatible" in str(e).lower():
                    error_msg += "\nPossible causes:\n" \
                               "1. Adapter architecture mismatch with base model\n" \
                               "2. LoRA config incompatibility"
                
                raise RuntimeError(error_msg) from e

        return self.cache_manager.get_or_load_adapter(f"{base_model.config._name_or_path}_{adapter_id}", _load_adapter)

    def set_active_adapter(self, model: PeftModel, adapter_id: str = None) -> bool:
        """Set a specific adapter as active with error handling"""
        if not isinstance(model, PeftModel):
            logger.warning("Model is not a PeftModel, cannot set adapter")
            return False
            
        available_adapters = self.loaded_adapters.get(model, [])
        
        if not available_adapters:
            logger.warning("No adapters loaded in model")
            return False
            
        # If no adapter specified, use the last loaded one
        if adapter_id is None:
            adapter_id = available_adapters[-1]
            
        if adapter_id in available_adapters:
            try:
                model.enable_adapters()
                model.set_adapter(adapter_id)
                logger.info(f"Successfully set active adapter to: {adapter_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting adapter {adapter_id} out of {available_adapters}: {str(e)}")
                return False
        else:
            logger.warning(f"Requested adapter {adapter_id} not loaded. Available adapters: {available_adapters}")
            return False
        
    def get_loaded_adapters(self, model: PeftModel) -> List[str]:
        """Get list of loaded adapters for a model"""
        return self.loaded_adapters.get(model, [])
    
class InferencePipeline:
    """Enhanced inference pipeline with timestamp tracking"""
    
    def __init__(self, model_config: ModelConfig, cache_manager: CacheManager, 
                 device_manager: DeviceManager, model_manager: ModelManager, 
                 lora_manager: LoRAManager):
        self.model_config = model_config
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        self.model_manager = model_manager
        self.lora_manager = lora_manager
        self.last_used = time.time()
        
        # Initialize dynamic components
        self.dynamic_temperature = (
            DynamicTemperature() if model_config.dynamic_temperature else None
        )
        
        # Load base model and tokenizer
        self.base_model, self.tokenizer = self.model_manager.load_base_model(
            model_config.base_model_id,
            quantize=model_config.quantization_bits == 4
        )
        
        # Setup tokenizer
        self.tokenizer = self.setup_tokenizer(self.tokenizer)

        # Resize model embeddings if needed
        if self.base_model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Resized model embeddings to match tokenizer")
        
        # Load adapters if specified
        self.current_model = self.base_model
        if model_config.adapter_ids:
            for adapter_id in model_config.adapter_ids:
                try:
                    self.current_model = self.lora_manager.load_adapter(
                        self.current_model, adapter_id
                    )
                    logger.info(f"Loaded adapter: {adapter_id}")
                except Exception as e:
                    logger.error(f"Failed to load adapter {adapter_id}: {e}")
            
            if isinstance(self.current_model, PeftModel):
                self.lora_manager.set_active_adapter(self.current_model)
        
        # Setup optimizations
        if model_config.use_memory_efficient_attention:
            self.setup_efficient_attention()
            
        self.setup_mixed_precision()
        self.optimal_batch_size = self._find_optimal_batch_size()

        logger.info(f"Initialized with optimal batch size: {self.optimal_batch_size}")

    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
            """Ensure tokenizer has required special tokens"""
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info("Using EOS token as padding token")
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added new [PAD] token to tokenizer")
                    
            # Ensure we have EOS token
            if tokenizer.eos_token is None:
                if tokenizer.sep_token is not None:
                    tokenizer.eos_token = tokenizer.sep_token
                else:
                    tokenizer.eos_token = tokenizer.pad_token
                    
            # Ensure we have BOS token
            if tokenizer.bos_token is None:
                if tokenizer.cls_token is not None:
                    tokenizer.bos_token = tokenizer.cls_token
                else:
                    tokenizer.bos_token = tokenizer.eos_token
                    
            # Log token IDs for debugging
            logger.debug(f"Tokenizer special tokens - PAD: {tokenizer.pad_token_id}, "
                        f"EOS: {tokenizer.eos_token_id}, BOS: {tokenizer.bos_token_id}")
            
            return tokenizer
    
    def generate(
        self,
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[int]]:
        """Generate multiple responses for a prompt when n > 1"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.current_model.device)

        # Configure generation parameters
        gen_config = {
            "max_new_tokens": generation_params.get("max_new_tokens", 4096),
            "do_sample": generation_params.get("temperature", 1.0) > 0,
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", 1.0),
            "num_return_sequences": generation_params.get("num_return_sequences", 1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if generation_params:
            if generation_params.get("presence_penalty", 0) != 0:
                gen_config["presence_penalty"] = generation_params["presence_penalty"]
            if generation_params.get("frequency_penalty", 0) != 0:
                gen_config["repetition_penalty"] = 1.0 + generation_params["frequency_penalty"]
            if generation_params.get("stop_sequences"):
                gen_config["stopping_criteria"] = self._create_stopping_criteria(
                    generation_params["stop_sequences"],
                    inputs['input_ids'].shape[1]
                )
            if generation_params.get("seed") is not None:
                torch.manual_seed(generation_params["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(generation_params["seed"])

        # Generate responses
        with torch.amp.autocast('cuda', dtype=self.dtype):
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    **gen_config
                )

        # Process outputs - now handling multiple sequences
        input_length = inputs['input_ids'].shape[1]
        responses = []
        token_counts = []

        # For each generated sequence
        for output in outputs:
            response_tokens = output[input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response_text)
            token_counts.append(len(response_tokens))

        return responses, token_counts
    
    def setup_efficient_attention(self):
        """Replace standard attention with memory-efficient version"""
        if hasattr(self.current_model, 'config') and hasattr(self.current_model.config, 'hidden_size'):
            hidden_size = self.current_model.config.hidden_size
            num_attention_heads = self.current_model.config.num_attention_heads
            self.efficient_attention = MemoryEfficientAttention(hidden_size, num_attention_heads)
            
            # Monkey patch attention computation if possible
            if hasattr(self.current_model, 'encoder') and hasattr(self.current_model.encoder, 'layer'):
                for layer in self.current_model.encoder.layer:
                    if hasattr(layer, 'attention'):
                        layer.attention.self = self.efficient_attention
            logger.info("Memory-efficient attention mechanism enabled")

    def setup_mixed_precision(self):
        """Configure automated mixed precision based on device capabilities"""
        device = self.current_model.device
        dtype = torch.float32  # default
        
        if torch.cuda.is_available() and 'cuda' in str(device):
            compute_capability = torch.cuda.get_device_capability(device.index if hasattr(device, 'index') else 0)
            
            if compute_capability[0] >= 8:
                dtype = torch.bfloat16
            elif compute_capability[0] >= 7:
                dtype = torch.float16
                
        elif torch.backends.mps.is_available() and 'mps' in str(device):
            dtype = torch.float16
        
        if dtype != torch.float32:
            self.current_model = self.current_model.to(dtype)
            logger.info(f"Using mixed precision with dtype: {dtype}")
        
        self.dtype = dtype

    def _find_optimal_batch_size(self, initial_batch_size: int = 1, max_batch_size: int = 128) -> int:
        """Find optimal batch size through binary search with memory monitoring"""
        if not torch.cuda.is_available():
            return initial_batch_size

        device = self.current_model.device
        if 'cuda' not in str(device):
            return initial_batch_size

        left, right = initial_batch_size, max_batch_size
        optimal_size = initial_batch_size
        
        sample_text = "Sample input text for batch size optimization."
        
        while left <= right:
            mid = (left + right) // 2
            try:
                torch.cuda.empty_cache()
                
                inputs = self.tokenizer([sample_text] * mid, 
                                     padding=True, 
                                     truncation=True,
                                     return_tensors="pt").to(device)
                
                with torch.amp.autocast('cuda',dtype=self.dtype):
                    with torch.no_grad():
                        _ = self.current_model.generate(
                            **inputs,
                            max_new_tokens=1,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                
                optimal_size = mid
                left = mid + 1
                
                memory_used = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                if memory_used > 0.9 * total_memory:
                    break
                
            except torch.cuda.OutOfMemoryError:
                right = mid - 1
                torch.cuda.empty_cache()
        
        return max(1, int(optimal_size * 0.9))

    def optimize_generation_params(self, prompt: str) -> Dict[str, Any]:
        """Optimize generation parameters based on prompt characteristics"""
        base_params = {
            "max_new_tokens": self.model_config.max_new_tokens,
            "do_sample": self.model_config.do_sample,
            "top_p": self.model_config.top_p,
            "top_k": self.model_config.top_k,
            "temperature": self.model_config.temperature,
            "num_return_sequences": self.model_config.num_return_sequences,
            "repetition_penalty": self.model_config.repetition_penalty,
            "pad_token_id": self.model_config.pad_token_id or self.tokenizer.pad_token_id
        }
        
        if self.model_config.dynamic_temperature:
            base_params["temperature"] = self.dynamic_temperature.get_optimal_temperature(
                prompt, self.tokenizer, base_params["temperature"]
            )
        
        return base_params

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format the prompt according to model's chat template"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the model's built-in chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to a generic template
            return f"<|system|>{system_prompt}</s><|user|>{user_prompt}</s><|assistant|>"
        
    def _create_stopping_criteria(self, stop_sequences: List[str], input_length: int):
        """Create stopping criteria for generation"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, input_length):
                self.tokenizer = tokenizer
                self.stop_ids = [
                    self.tokenizer.encode(seq, add_special_tokens=False)
                    for seq in stop_sequences
                ]
                self.input_length = input_length

            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_ids:
                    if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                        return True
                return False

        return StoppingCriteriaList([
            StopSequenceCriteria(
                self.tokenizer,
                stop_sequences,
                input_length=input_length
            )
        ])
    def process_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        generation_params: Optional[Dict[str, Any]] = None,
        active_adapter: str = None,
        return_token_count: bool = True
    ) -> Tuple[List[str], List[int]]:
        """Process a batch of prompts with all optimizations"""
        
        # Set the requested adapter if specified
        if isinstance(self.current_model, PeftModel) and active_adapter is not None:
            self.lora_manager.set_active_adapter(self.current_model, active_adapter)

        all_responses = []
        token_counts = []
        
        # Format all prompts using chat template
        formatted_prompts = [
            self.format_chat_prompt(system_prompt, user_prompt)
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        
        # Get number of completions requested
        n = generation_params.get("num_return_sequences", 1) if generation_params else 1
        
        for i in range(0, len(formatted_prompts), self.optimal_batch_size):
            batch_prompts = formatted_prompts[i:i + self.optimal_batch_size]
            batch_system = system_prompts[i:i + self.optimal_batch_size]
            batch_user = user_prompts[i:i + self.optimal_batch_size]
            
            # Check cache first if enabled
            if self.model_config.enable_prompt_caching:
                cached_responses = []
                uncached_indices = []
                
                for idx, prompt in enumerate(batch_prompts):
                    temp = generation_params.get("temperature", self.model_config.temperature) if generation_params else self.model_config.temperature
                    top_p = generation_params.get("top_p", self.model_config.top_p) if generation_params else self.model_config.top_p
                    
                    cached_response = self.cache_manager.prompt_cache.get_cached_response(
                        prompt,
                        temp,
                        top_p
                    )
                    if cached_response is not None:
                        # For cached responses, replicate n times if multiple completions requested
                        cached_responses.extend([cached_response] * n)
                    else:
                        uncached_indices.append(idx)
                
                if uncached_indices:
                    batch_prompts = [batch_prompts[i] for i in uncached_indices]
                else:
                    batch_prompts = []
            
            if batch_prompts:  # If there are any uncached prompts
                # Configure generation parameters
                base_params = {
                    "max_new_tokens": generation_params.get("max_new_tokens", 4096) if generation_params else self.model_config.max_new_tokens,
                    "do_sample": generation_params.get("temperature", 1.0) > 0 if generation_params else self.model_config.do_sample,
                    "temperature": generation_params.get("temperature", 1.0) if generation_params else self.model_config.temperature,
                    "top_p": generation_params.get("top_p", 1.0) if generation_params else self.model_config.top_p,
                    "num_return_sequences": n,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                # Add optional parameters if specified
                if generation_params:
                    if generation_params.get("presence_penalty", 0) != 0:
                        base_params["presence_penalty"] = generation_params["presence_penalty"]
                    if generation_params.get("frequency_penalty", 0) != 0:
                        base_params["repetition_penalty"] = 1.0 + generation_params["frequency_penalty"]
                    if generation_params.get("logit_bias"):
                        base_params["logit_bias"] = generation_params["logit_bias"]
                    if generation_params.get("seed") is not None:
                        torch.manual_seed(generation_params["seed"])
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(generation_params["seed"])
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.current_model.device)

                # Get the length of each input
                input_lengths = inputs['input_ids'].shape[1]

                # Add stopping criteria if specified
                if generation_params and generation_params.get("stop_sequences"):
                    base_params["stopping_criteria"] = self._create_stopping_criteria(
                        generation_params["stop_sequences"],
                        input_lengths
                    )

                # Generate responses
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    with torch.no_grad():
                        outputs = self.current_model.generate(
                            **inputs,
                            **base_params
                        )
                
                # Decode outputs and remove input portion
                batch_responses = []
                batch_token_counts = []
                
                # Handle multiple sequences per input
                num_return_sequences = base_params["num_return_sequences"]
                for i in range(0, len(outputs), num_return_sequences):
                    sequences = outputs[i:i + num_return_sequences]
                    for seq in sequences:
                        response_tokens = seq[input_lengths:]
                        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                        batch_responses.append(response_text)
                        batch_token_counts.append(len(response_tokens))
                
                # Cache new responses if enabled
                if self.model_config.enable_prompt_caching:
                    for prompt, response in zip(batch_prompts, batch_responses[::n]):  # Only cache first response of each input
                        self.cache_manager.prompt_cache.add_to_cache(
                            prompt,
                            response,
                            base_params["temperature"],
                            base_params["top_p"]
                        )
                
                # Merge cached and new responses in correct order
                all_responses.extend(cached_responses)
                if uncached_indices:
                    response_idx = 0
                    for original_idx in range(len(formatted_prompts[i:i + self.optimal_batch_size])):
                        if original_idx in uncached_indices:
                            # Add n responses for this uncached prompt
                            for _ in range(n):
                                while len(all_responses) < original_idx * n + _:
                                    all_responses.append("")
                                if response_idx < len(batch_responses):
                                    all_responses.append(batch_responses[response_idx])
                                    response_idx += 1
                
                if return_token_count:
                    # Count tokens for responses
                    token_counts.extend([0] * len(cached_responses))  # 0 for cached responses
                    token_counts.extend(batch_token_counts)
        
        if return_token_count:
            return all_responses, token_counts
        return all_responses, [0] * len(all_responses)
    
class ChatCompletionMessage:
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role

class ChatCompletionChoice:
    def __init__(self, index: int, message: Dict[str, str], finish_reason: str = "stop"):
        self.index = index
        self.message = ChatCompletionMessage(**message)
        self.finish_reason = finish_reason

class ChatCompletionUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

class ChatCompletion:
    def __init__(self, response_dict: Dict):
        self.id = response_dict["id"]
        self.object = response_dict["object"]
        self.created = response_dict["created"]
        self.model = response_dict["model"]
        self.choices = [
            ChatCompletionChoice(
                index=choice["index"],
                message=choice["message"],
                finish_reason=choice["finish_reason"]
            )
            for choice in response_dict["choices"]
        ]
        self.usage = ChatCompletionUsage(**response_dict["usage"])
    
    def model_dump(self) -> Dict:
        """Convert back to dictionary format if needed"""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            }
        }
class InferenceClient:
    """OpenAI SDK Compatible client for local inference with dynamic model support"""
    
    def __init__(self):
        self.cache_manager = CacheManager(max_size=10)
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(self.cache_manager, self.device_manager)
        self.lora_manager = LoRAManager(self.cache_manager)
        self.chat = self.Chat(self)
        self.models = self.Models()
        self._pipeline_cache = {}

    def get_pipeline(self, model: str) -> 'InferencePipeline':
        if model not in self._pipeline_cache:
            model_config = parse_model_string(model)
            self._pipeline_cache[model] = InferencePipeline(
                model_config,
                self.cache_manager,
                self.device_manager,
                self.model_manager,
                self.lora_manager
            )
        return self._pipeline_cache[model]
    
    def clean_unused_pipelines(self, max_inactive: int = 5):
        """Clean up pipelines that haven't been used recently"""
        if len(self._pipeline_cache) > max_inactive:
            oldest_models = sorted(
                self._pipeline_cache.keys(),
                key=lambda x: self._pipeline_cache[x].last_used
            )[:-max_inactive]
            
            for model in oldest_models:
                del self._pipeline_cache[model]
                torch.cuda.empty_cache()

    class Chat:
        """OpenAI-compatible chat interface"""
        def __init__(self, client: 'InferenceClient'):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client: 'InferenceClient'):
                self.client = client

            def create(
                self,
                messages: List[Dict[str, str]],
                model: str,
                temperature: float = 1.0,
                top_p: float = 1.0,
                n: int = 1,
                stream: bool = False,
                stop: Optional[Union[str, List[str]]] = None,
                max_tokens: Optional[int] = None,
                presence_penalty: float = 0,
                frequency_penalty: float = 0,
                logit_bias: Optional[Dict[str, float]] = None,
                user: Optional[str] = None,
                seed: Optional[int] = None,
                **kwargs
            ) -> ChatCompletion:
                """Create a chat completion with OpenAI-compatible parameters"""
                if stream:
                    raise NotImplementedError("Streaming is not yet supported")

                pipeline = self.client.get_pipeline(model)
                
                # Apply chat template to messages
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Set generation parameters
                generation_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_return_sequences": n,
                    "max_new_tokens": max_tokens if max_tokens is not None else 4096,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "stop_sequences": [stop] if isinstance(stop, str) else stop,
                    "seed": seed,
                    "logit_bias": logit_bias
                }

                # Generate responses - now returns list of responses and token counts
                responses, token_counts = pipeline.generate(
                    prompt,
                    generation_params=generation_params
                )
                
                # Calculate prompt tokens
                prompt_tokens = len(pipeline.tokenizer.encode(prompt))
                completion_tokens = sum(token_counts)
                
                # Create OpenAI-compatible response format
                response_dict = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": idx,
                            "message": {
                                "role": "assistant",
                                "content": response
                            },
                            "finish_reason": "stop"
                        }
                        for idx, response in enumerate(responses)
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": completion_tokens + prompt_tokens
                    }
                }
                
                self.client.clean_unused_pipelines()
                # Return ChatCompletion object
                return ChatCompletion(response_dict)
        
    class Models:
        """OpenAI-compatible models interface"""
        def list(self):
            """Return list of supported models"""
            try:
                import requests
                response = requests.get(
                    "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=100"
                )
                models = response.json()
                model_list = []
                
                for model in models:
                    if 'pipeline_tag' in model and model['pipeline_tag'] == 'text-generation':
                        model_list.append({
                            "id": model['id'],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "huggingface",
                        })
                
                return {"data": model_list, "object": "list"}
            except Exception as e:
                logger.warning(f"Failed to fetch models: {e}")
                return {
                    "data": [{
                        "id": "HuggingFaceTB/SmolLM-135M-Instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "huggingface",
                    }],
                    "object": "list"
                }
            
def create_inference_client() -> InferenceClient:
    """Factory function to create an inference client"""
    return InferenceClient()

def parse_model_string(model: str) -> ModelConfig:
    """Parse the model string to extract base model and adapter IDs"""
    parts = model.split('+')
    base_model_id = parts[0]
    adapter_ids = parts[1:] if len(parts) > 1 else None
    
    return ModelConfig(
        base_model_id=base_model_id,
        adapter_ids=adapter_ids,
        use_memory_efficient_attention=True,
        enable_prompt_caching=True,
        dynamic_temperature=False,
    )
