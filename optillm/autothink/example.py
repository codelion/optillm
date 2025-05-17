#!/usr/bin/env python3
"""
Example usage of AutoThink.

This script demonstrates how to use AutoThink with a language model.
"""

import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from optillm.autothink import autothink_decode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run AutoThink demo")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-r1-llama-8b", 
                        help="Model name or path")
    parser.add_argument("--steering-dataset", type=str, 
                        default="codelion/Qwen3-0.6B-pts-steering-vectors",
                        help="Steering vectors dataset")
    parser.add_argument("--target-layer", type=int, default=19,
                        help="Target layer for steering")
    parser.add_argument("--query", type=str, 
                        default="Explain quantum computing to me in detail",
                        help="Query to process")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    try:
        logger.info(f"Loading model: {args.model}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Load model with appropriate configuration based on device
        model_kwargs = {"trust_remote_code": True}
        
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        
        # Ensure proper PAD token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully")
        
        # Create AutoThink configuration
        config = {
            "steering_dataset": args.steering_dataset,
            "target_layer": args.target_layer,
            "pattern_strengths": {
                "depth_and_thoroughness": 2.5,
                "numerical_accuracy": 2.0,
                "self_correction": 3.0,
                "exploration": 2.0,
                "organization": 1.5
            }
        }
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.query}
        ]
        
        # Process with AutoThink
        logger.info("Running AutoThink processing...")
        response = autothink_decode(model, tokenizer, messages, config)
        
        # Print response
        print("\n" + "=" * 80)
        print("QUERY:", args.query)
        print("-" * 80)
        print(response)
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in AutoThink demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()
