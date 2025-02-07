import argparse
import json
import os
import logging
import re
from typing import Dict, Optional, Union
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")

SYSTEM_PROMPT = '''You are solving mathematics problems.

Important: Always end your solution with the final answer in this format:

\\[
\\boxed{your_answer_here}
\\]

Make sure to write your complete answer inside the \\boxed{} command.'''

def load_math500_dataset() -> list[dict]:
    """
    Load the MATH-500 dataset.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    dataset = dataset["test"]
    logging.debug(f"Dataset size: {len(dataset)}.")
    return dataset

def extract_answer(response: str) -> Optional[str]:
    """
    Extract the answer from a math solution response.
    Uses a simple pattern to extract content between \boxed{...} commands.
    Handles nested braces and complex mathematical expressions.
    """
    if not response:
        return None
    
    # Find the last \boxed{...} in the response
    start_idx = response.rfind('\\boxed{')
    if start_idx == -1:
        return None
        
    # Find the matching closing brace
    brace_count = 1
    pos = start_idx + 7  # length of '\boxed{'
    
    while pos < len(response) and brace_count > 0:
        if response[pos] == '{':
            brace_count += 1
        elif response[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        answer = response[start_idx + 7:pos - 1]
        return answer.strip()
        
    return None

def normalize_answer(answer: str) -> str:
    """
    Normalize the answer string for comparison.
    Handles common LaTeX formatting variations while preserving mathematical meaning.
    """
    if answer is None:
        return ""
        
    # Remove all whitespace
    answer = ''.join(answer.split())
    
    # Remove \left and \right commands
    answer = answer.replace('\\left', '').replace('\\right', '')
    
    # Normalize fraction commands
    answer = answer.replace('\\dfrac', '\\frac')
    
    # Remove any remaining extra backslashes before common symbols
    answer = answer.replace('\\(', '(').replace('\\)', ')')
    answer = answer.replace('\\[', '[').replace('\\]', ']')
    answer = answer.replace('\\{', '{').replace('\\}', '}')
    
    # Convert to lowercase for case-insensitive comparison
    return answer.lower()

def get_llm_response(problem: str, model: str) -> str:
    """
    Get response from the LLM for a given problem.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        
    Returns:
        str: Model's response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.6,  # Lower temperature for more consistent answers
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + "\n" + problem}
            ],
            max_tokens=4096, # for thinking models, we need to use a lot more tokens
            extra_body = {
                "decoding" : "thinkdeeper",
            }
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return ""

def load_existing_results(filename: str) -> list[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def analyze_results(results: list[Dict]):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (list[Dict]): List of evaluation results
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print("\n=== Results Summary ===")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print(f"Predicted: {r['predicted_answer']}")
            print("---")

def main(model: str):
    """Main evaluation function."""
    os.makedirs("results", exist_ok=True)
    results_file = f"evaluation_results_math500_{model.replace('/', '_')}.json"
    
    dataset = load_math500_dataset()
    existing_results = load_existing_results(results_file)
    
    # Create a set of already processed indexes for efficient lookup
    processed_indexes = {result['index'] for result in existing_results}
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        # Skip if this index has already been processed
        if idx in processed_indexes:
            continue
            
        problem_text = item['problem']
        correct_answer = item['answer']
        
        # Get model's response
        response = get_llm_response(problem_text, model)
        predicted_answer = extract_answer(response)
        
        # Compare answers after normalization
        is_correct = normalize_answer(predicted_answer) == normalize_answer(correct_answer)
        
        result = {
            "index": idx,
            "problem": problem_text,
            "response": response,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        }
        save_result(results_file, result)
    
    final_results = load_existing_results(results_file)
    analyze_results(final_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on MATH-500 problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    args = parser.parse_args()
    
    main(args.model)
