import argparse
import json
import os
import logging
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")

SYSTEM_PROMPT = '''You are solving AIME (American Invitational Mathematics Examination) problems.

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{X}.
   \\]

2. $n=\\boxed{X}$

where X is your integer answer between 0 and 999.'''

def load_2024_dataset() -> list[dict]:
    """
    Load the dataset of problems.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")
    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])
    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
    return dataset

def extract_answer(response: str) -> Optional[int]:
    """
    Extract the numerical answer from a math solution response.
    Handles various formats of boxed answers and falls back to last number if needed.
    Removes any content within <think> tags before processing.
    """
    if not response:
        return None
    
    # Remove content within <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Clean the response
    response = ' '.join(response.split())
    
    patterns = [
        r'\$n=\\boxed{(\d+)}\$',
        r'\\\[\\boxed{(\d+)}\\\]',
        r'\\\[\\boxed{(\d+)}\.\\\]',
        r'\\boxed{(\d+)}',
        r'\$\\boxed{(\d+)}\$',
        r'boxed{(\d+)}',
        r'\\boxed\s*{\s*(\d+)\s*}',
        r'\bboxed\s*{\s*(\d+)\s*}',
        r'final answer is[^\d]*(\d+)',
        r'answer is[^\d]*(\d+)',
        r'answer:[^\d]*(\d+)',
        r'= ?(\d+)$'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        last_match = None
        for match in matches:
            last_match = match
            
        if last_match:
            try:
                return int(last_match.group(1))
            except (ValueError, IndexError):
                continue
    
    numbers = re.findall(r'(\d+)', response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
            
    return None

def get_llm_response(problem: str, model: str) -> Union[str, List[Dict]]:
    """
    Get response from the LLM for a given problem.
    If multiple choices are returned, formats them as attempt dictionaries.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        
    Returns:
        Union[str, List[Dict]]: Either a string response or list of attempt dictionaries
    """
    try:
        response = client.with_options(timeout=600.0).chat.completions.create(
            model=model,
            temperature=0.6,
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + problem}
            ],
            max_tokens=16384, # for thinking models, we need to use a lot more tokens
            extra_body = {
                "decoding" : "thinkdeeper",
            }
        )
        
        # If there's more than one choice, format as attempts
        if len(response.choices) > 1:
            attempts = []
            for i, choice in enumerate(response.choices):
                response_text = choice.message.content.strip()
                predicted_answer = extract_answer(response_text)
                attempts.append({
                    "attempt_number": i + 1,
                    "response": response_text,
                    "predicted_answer": predicted_answer
                })
            return attempts
            
        # If single choice, return as before
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return ""

def make_n_attempts(problem: str, model: str, n: int) -> List[Dict]:
    """
    Make n attempts to solve a problem and return all responses and predictions.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        n (int): Number of attempts to make
        
    Returns:
        List[Dict]: List of dictionaries containing response and predicted answer for each attempt
    """
    attempts = []
    remaining_attempts = n
    
    while remaining_attempts > 0:
        response = get_llm_response(problem, model)
        
        # If response is already formatted as attempts
        if isinstance(response, list):
            attempts.extend(response)
            remaining_attempts = n - len(attempts)
        else:
            # Process single response as before
            predicted_answer = extract_answer(response)
            attempts.append({
                "attempt_number": len(attempts) + 1,
                "response": response,
                "predicted_answer": predicted_answer
            })
            remaining_attempts -= 1
    
    return attempts

def evaluate_pass_at_n(attempts: List[Dict], correct_answer: int) -> Tuple[bool, Optional[int]]:
    """
    Evaluate if any of the n attempts got the correct answer.
    
    Args:
        attempts (List[Dict]): List of attempt results
        correct_answer (int): The correct answer
        
    Returns:
        Tuple[bool, Optional[int]]: (whether any attempt was correct, first correct attempt number)
    """
    for attempt in attempts:
        if attempt["predicted_answer"] == correct_answer:
            return True, attempt["attempt_number"]
    return False, None

def load_existing_results(filename: str) -> List[Dict]:
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

def get_last_processed_index(results: List[Dict]) -> int:
    """Get the index of the last processed problem."""
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def analyze_results(results: List[Dict], n: int):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (List[Dict]): List of evaluation results
        n (int): Number of attempts per problem
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print("\n=== Results Summary ===")
    print(f"Evaluation mode: pass@{n}")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Calculate attempt statistics
    successful_attempts = [r['first_correct_attempt'] for r in results if r['is_correct']]
    if successful_attempts:
        avg_attempts = sum(successful_attempts) / len(successful_attempts)
        print(f"\nFor correct solutions:")
        print(f"Average attempts needed: {avg_attempts:.2f}")
        print(f"Attempt distribution:")
        for i in range(1, n + 1):
            count = sum(1 for x in successful_attempts if x == i)
            print(f"  Attempt {i}: {count} problems")
    
    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print("Predicted answers across attempts:", [
                attempt['predicted_answer'] for attempt in r['attempts']
            ])
            print("---")

def main(model: str, n_attempts: int):
    """Main evaluation function that handles gaps in processed indexes."""
    os.makedirs("results", exist_ok=True)
    
    results_file = f"evaluation_results_{model.replace('/', '_')}_pass_at_{n_attempts}.json"
    
    dataset = load_2024_dataset()
    existing_results = load_existing_results(results_file)
    
    # Create a set of already processed indexes for efficient lookup
    processed_indexes = {result['index'] for result in existing_results}
    
    for _, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        id = int(item['id'])
        # Skip if this index has already been processed
        if id in processed_indexes:
            continue
            
        problem_text = item['problem']
        correct_answer = int(item['answer'])
        
        # Make n attempts for each problem
        attempts = make_n_attempts(problem_text, model, n_attempts)
        is_correct, first_correct = evaluate_pass_at_n(attempts, correct_answer)
        
        result = {
            "index": id,
            "problem": problem_text,
            "attempts": attempts,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "first_correct_attempt": first_correct
        }
        save_result(results_file, result)
    
    final_results = load_existing_results(results_file)
    analyze_results(final_results, n_attempts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on AIME 2024 problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--n", type=int, default=1, help="Number of attempts per problem (for pass@n evaluation)")
    args = parser.parse_args()
    
    main(args.model, args.n)