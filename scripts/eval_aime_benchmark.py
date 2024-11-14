import argparse
import json
import os
import logging
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),  base_url="http://localhost:8000/v1")

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
    
    Args:
        response (str): The complete response text from the model
        
    Returns:
        Optional[int]: The extracted answer as an integer, or None if no valid answer found
    """
    if not response:
        return None
        
    # Clean the response: normalize whitespace and handle potential Unicode
    response = ' '.join(response.split())
    
    # List of regex patterns to try, in order of preference
    patterns = [
        # $n=\boxed{X}$ format
        r'\$n=\\boxed{(\d+)}\$',
        
        # LaTeX display style answer: \[\boxed{X}\] or \[\boxed{X}.\]
        r'\\\[\\boxed{(\d+)}\\\]',
        r'\\\[\\boxed{(\d+)}\.\\\]',
        
        # Inline LaTeX \boxed{X}
        r'\\boxed{(\d+)}',
        
        # Common variations
        r'\$\\boxed{(\d+)}\$',
        r'boxed{(\d+)}',
        
        # Less strict patterns
        r'\\boxed\s*{\s*(\d+)\s*}',
        r'\bboxed\s*{\s*(\d+)\s*}',
        
        # Plain text answer indicators
        r'final answer is[^\d]*(\d+)',
        r'answer is[^\d]*(\d+)',
        r'answer:[^\d]*(\d+)',
        r'= ?(\d+)$'
    ]
    
    # Try each pattern in order
    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        # Get the last match for this pattern (in case there are multiple)
        last_match = None
        for match in matches:
            last_match = match
            
        if last_match:
            try:
                return int(last_match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Fallback: Extract all numbers and take the last one
    # This is our last resort, assuming the answer typically comes last
    numbers = re.findall(r'(\d+)', response)
    if numbers:
        try:
            # Convert to int and return the last number found
            return int(numbers[-1])
        except ValueError:
            pass
            
    # If all methods fail, return None
    return None

def get_llm_response(problem: str, model: str) -> str:
    """
    Get response from the LLM for a given problem.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem}
            ],
            max_tokens=8192,
            # extra_body={
            #     "decoding": "entropy_decoding",
            # }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return ""

def evaluate_response(predicted_answer: Optional[int], correct_answer: int) -> bool:
    """
    Evaluate if the predicted answer matches the correct answer.
    """
    if predicted_answer is None:
        return False
    return predicted_answer == correct_answer

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

def analyze_results(results: List[Dict]):
    """Analyze and print summary statistics of the results."""
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print("\n=== Results Summary ===")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print incorrect problems for analysis
    print("\n=== Incorrect Answers ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print(f"Predicted: {r['predicted_answer']}")
            print("---")

def main(model: str):
    """Main evaluation function."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Setup results file
    results_file = f"evaluation_results_{model.replace('/', '_')}.json"
    
    # Load dataset
    dataset = load_2024_dataset()
    
    # Load existing results
    existing_results = load_existing_results(results_file)
    last_processed_index = get_last_processed_index(existing_results)
    
    # Process problems
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        if idx <= last_processed_index:
            continue
            
        problem_text = item['problem']
        correct_answer = int(item['answer'])
        
        # Get model response
        response = get_llm_response(problem_text, model)
        logger.debug(f"Response: {response}")
        predicted_answer = extract_answer(response)
        is_correct = evaluate_response(predicted_answer, correct_answer)
        
        # Save result
        result = {
            "index": idx,
            "problem": problem_text,
            "model_response": response,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
        save_result(results_file, result)
        
        # Optional: Add delay between requests if needed
        # time.sleep(5)
    
    # Analyze results
    final_results = load_existing_results(results_file)
    analyze_results(final_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on AIME 2024 problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    args = parser.parse_args()
    
    main(args.model)
