import os
import json
import argparse
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from openai import AsyncOpenAI
from typing import List, Dict, Any
import random

# OptILM approaches
APPROACHES = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]

async def generate_response(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate a response using the specified approach."""
    approach = kwargs.get("approach", "none")
    temperature = kwargs.get("temperature", 0.)
    if approach == "none":
        # Use the base model without any optimization technique
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }
    else:
        # Use OptILM with the specified approach
        client = AsyncOpenAI(api_key="none", base_url="http://localhost:8000/v1")
        response = await client.chat.completions.create(
            model=f"{approach}-{kwargs['model']}",  # Assuming OptILM uses this naming convention
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }

async def rank_responses(prompt: str, responses: List[Dict[str, Any]]) -> List[int]:
    """Rank the responses using the LLM."""
    ranking_prompt = f"Given the following prompt:\n\n{prompt}\n\nRank the following responses from best to worst, considering accuracy, completeness, and relevance. Provide the ranking as a comma-separated list of indices (0-indexed). Do not add any explanations or any other text other than the comma-separated list.\n\n"
    for i, response in enumerate(responses):
        ranking_prompt += f"Response {i}:\n{response['content']}\n\n"
    client = AsyncOpenAI()
    ranking_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ranking_prompt}],
    )
    
    ranking_str = ranking_response.choices[0].message.content.strip()
    return [int(idx) for idx in ranking_str.split(",")]

async def process_sample(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Process a single sample from the dataset."""
    prompt = sample[kwargs["prompt_column"]]
    approach = kwargs["approach"]
    results = []

    for _ in range(kwargs["num_completions_per_prompt"]):
        response = await generate_response(prompt, model=kwargs["model"], approach=approach, temperature=kwargs["temperature"])
        results.append({"approach": approach, **response})

    random.shuffle(results)
    # Rank the responses
    rankings = await rank_responses(prompt, results)

    # Add rankings to results
    for rank, idx in enumerate(rankings):
        results[idx]["rank"] = rank

    return {
        "prompt": prompt,
        "results": results,
    }

async def generate_dataset(dataset: str, output_file: str, **kwargs):
    """Generate the dataset and save it to a JSONL file."""
    dataset = load_dataset(dataset, split=f"{kwargs.get('split')}[:{kwargs.get('num_samples')}]")
    
    process_kwargs = {k: v for k, v in kwargs.items() if k not in ["dataset", "split", "output_file"]}


    with open(f"data/{output_file}", "w") as f:
        for sample in tqdm(dataset):
            result = await process_sample(sample, **process_kwargs)
            f.write(json.dumps(result) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate OptILM dataset")
    parser.add_argument("--approach", type=str, default="mcts", help="optillm approach")
    parser.add_argument("--dataset", type=str, default="AI-MO/NuminaMath-CoT", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--prompt_column", type=str, default="problem", help="Column name for the prompt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--num_completions_per_prompt", type=int, default=1, help="Number of completions per prompt")
    parser.add_argument("--temperature", type=float, default=0., help="Temperature for sampling")
    parser.add_argument("--output_file", type=str, default="optillm_dataset.jsonl", help="Output file path")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    args = parser.parse_args()

    asyncio.run(generate_dataset(**vars(args)))
    print(f"Dataset generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()
