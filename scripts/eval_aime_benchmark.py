import argparse
import json
import os
import logging
import re
import time
from typing import List, Dict, Tuple, Optional, Union, Counter
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import statistics
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

client = OpenAI(api_key="optillm", base_url="http://localhost:8001/v1")

SYSTEM_PROMPT = '''You are solving AIME (American Invitational Mathematics Examination) problems.

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{X}.
   \\]

2. $n=\\boxed{X}$

where X is your integer answer between 0 and 999.'''

# Define the thought transition phrases to track
THOUGHT_TRANSITIONS = [
    "Wait,", 
    "Alternatively,", 
    "However,", 
    "Additionally,"
]

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
    """
    if not response:
        return None
        
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

def analyze_thinking(response: str) -> Dict:
    """
    Analyze thinking patterns in the response.
    Extract tokens between <think> and </think> tags and count thought transitions.
    
    Args:
        response (str): The model's response text
        
    Returns:
        Dict: Analysis metrics including thinking tokens and thought transitions
    """
    # Default result with zero values
    result = {
        "has_think_tags": False,
        "thinking_tokens": 0,
        "thinking_tokens_text": "",
        "total_tokens": len(response.split()),
        "thought_transitions": 0,
        "transition_counts": {phrase: 0 for phrase in THOUGHT_TRANSITIONS},
        "transition_positions": []
    }
    
    # Extract content between <think> and </think> tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_match = think_pattern.search(response)
    
    if think_match:
        thinking_text = think_match.group(1)
        result["has_think_tags"] = True
        result["thinking_tokens"] = len(thinking_text.split())
        result["thinking_tokens_text"] = thinking_text
        
        # Count thought transitions
        position = 0
        for phrase in THOUGHT_TRANSITIONS:
            # Find all occurrences of each transition phrase
            for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', thinking_text):
                result["transition_counts"][phrase] += 1
                # Record the approximate token position of the transition
                token_position = len(thinking_text[:match.start()].split())
                result["transition_positions"].append((phrase, token_position))
                
        # Sort transition positions by token position
        result["transition_positions"].sort(key=lambda x: x[1])
        
        # Calculate total transitions
        result["thought_transitions"] = sum(result["transition_counts"].values())
    
    return result

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
        response = client.with_options(timeout=1000.0).chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + problem}
            ],
            max_tokens=8192,
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

def make_n_attempts(problem: str, model: str, n: int, analyze_thoughts: bool = False) -> List[Dict]:
    """
    Make n attempts to solve a problem and return all responses and predictions.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        n (int): Number of attempts to make
        analyze_thoughts (bool): Whether to analyze thinking patterns
        
    Returns:
        List[Dict]: List of dictionaries containing response and predicted answer for each attempt
    """
    attempts = []
    remaining_attempts = n
    
    while remaining_attempts > 0:
        response = get_llm_response(problem, model)
        
        # If response is already formatted as attempts
        if isinstance(response, list):
            for attempt in response:
                if analyze_thoughts:
                    attempt["thought_analysis"] = analyze_thinking(attempt["response"])
            attempts.extend(response)
            remaining_attempts = n - len(attempts)
        else:
            # Process single response
            predicted_answer = extract_answer(response)
            attempt_data = {
                "attempt_number": len(attempts) + 1,
                "response": response,
                "predicted_answer": predicted_answer
            }
            if analyze_thoughts:
                attempt_data["thought_analysis"] = analyze_thinking(response)
            attempts.append(attempt_data)
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

def analyze_results(results: List[Dict], n: int, analyze_thoughts: bool = False):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (List[Dict]): List of evaluation results
        n (int): Number of attempts per problem
        analyze_thoughts (bool): Whether to analyze thinking patterns
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
    
    if analyze_thoughts:
        print("\n=== Thinking Pattern Analysis ===")
        
        # Collect metrics about thinking patterns for correct vs incorrect attempts
        correct_attempts = []
        incorrect_attempts = []
        
        for result in results:
            for attempt in result['attempts']:
                if 'thought_analysis' in attempt:
                    if result['is_correct'] and attempt['predicted_answer'] == result['correct_answer']:
                        correct_attempts.append(attempt)
                    else:
                        incorrect_attempts.append(attempt)
        
        # Function to calculate statistics for a group of attempts
        def calc_stats(attempts):
            if not attempts:
                return {
                    "count": 0,
                    "avg_thinking_tokens": 0,
                    "avg_thought_transitions": 0,
                    "transition_usage": {phrase: 0 for phrase in THOUGHT_TRANSITIONS},
                    "has_think_tags_pct": 0
                }
            
            thinking_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts]
            thought_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts]
            has_think_tags = sum(1 for a in attempts if a['thought_analysis']['has_think_tags'])
            
            # Count total transition usage
            transition_usage = defaultdict(int)
            for attempt in attempts:
                for phrase, count in attempt['thought_analysis']['transition_counts'].items():
                    transition_usage[phrase] += count
            
            return {
                "count": len(attempts),
                "avg_thinking_tokens": statistics.mean(thinking_tokens) if thinking_tokens else 0,
                "median_thinking_tokens": statistics.median(thinking_tokens) if thinking_tokens else 0,
                "min_thinking_tokens": min(thinking_tokens) if thinking_tokens else 0,
                "max_thinking_tokens": max(thinking_tokens) if thinking_tokens else 0,
                "avg_thought_transitions": statistics.mean(thought_transitions) if thought_transitions else 0,
                "median_thought_transitions": statistics.median(thought_transitions) if thought_transitions else 0,
                "transition_usage": dict(transition_usage),
                "has_think_tags_pct": (has_think_tags / len(attempts)) * 100 if attempts else 0
            }
        
        # Calculate statistics
        correct_stats = calc_stats(correct_attempts)
        incorrect_stats = calc_stats(incorrect_attempts)
        all_stats = calc_stats(correct_attempts + incorrect_attempts)
        
        # Print statistics
        print(f"\nOverall Thinking Statistics (All {all_stats['count']} Attempts):")
        print(f"- Average thinking tokens: {all_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {all_stats['median_thinking_tokens']}")
        print(f"- Range: {all_stats['min_thinking_tokens']} - {all_stats['max_thinking_tokens']} tokens")
        print(f"- Average thought transitions: {all_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {all_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {all_stats['has_think_tags_pct']:.2f}%")
        print(f"- Transition phrase usage:")
        for phrase, count in all_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        print(f"\nCorrect Attempts ({correct_stats['count']}):")
        print(f"- Average thinking tokens: {correct_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {correct_stats['median_thinking_tokens']}")
        print(f"- Average thought transitions: {correct_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {correct_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {correct_stats['has_think_tags_pct']:.2f}%")
        print(f"- Transition phrase usage:")
        for phrase, count in correct_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        print(f"\nIncorrect Attempts ({incorrect_stats['count']}):")
        print(f"- Average thinking tokens: {incorrect_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {incorrect_stats['median_thinking_tokens']}")
        print(f"- Average thought transitions: {incorrect_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {incorrect_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {incorrect_stats['has_think_tags_pct']:.2f}%")
        print(f"- Transition phrase usage:")
        for phrase, count in incorrect_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        # Calculate correlation between thinking tokens and correctness
        if correct_attempts and incorrect_attempts:
            print("\nCorrelation Analysis:")
            
            # Find problems with both correct and incorrect attempts for comparison
            problems_with_both = defaultdict(lambda: {"correct": [], "incorrect": []})
            
            for result in results:
                problem_id = result['index']
                for attempt in result['attempts']:
                    if 'thought_analysis' in attempt:
                        category = "correct" if attempt['predicted_answer'] == result['correct_answer'] else "incorrect"
                        problems_with_both[problem_id][category].append(attempt)
            
            # Filter to problems that have both correct and incorrect attempts
            valid_problems = {
                k: v for k, v in problems_with_both.items() 
                if v["correct"] and v["incorrect"]
            }
            
            if valid_problems:
                print(f"Found {len(valid_problems)} problems with both correct and incorrect attempts")
                
                # For each problem, compare thinking patterns
                avg_token_diff = []
                avg_transition_diff = []
                
                for problem_id, attempts in valid_problems.items():
                    correct_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['correct']]
                    incorrect_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['incorrect']]
                    
                    correct_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['correct']]
                    incorrect_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['incorrect']]
                    
                    avg_correct_tokens = statistics.mean(correct_tokens) if correct_tokens else 0
                    avg_incorrect_tokens = statistics.mean(incorrect_tokens) if incorrect_tokens else 0
                    
                    avg_correct_transitions = statistics.mean(correct_transitions) if correct_transitions else 0
                    avg_incorrect_transitions = statistics.mean(incorrect_transitions) if incorrect_transitions else 0
                    
                    avg_token_diff.append(avg_correct_tokens - avg_incorrect_tokens)
                    avg_transition_diff.append(avg_correct_transitions - avg_incorrect_transitions)
                
                print(f"Average token difference (correct - incorrect): {statistics.mean(avg_token_diff):.2f}")
                print(f"Average transition difference (correct - incorrect): {statistics.mean(avg_transition_diff):.2f}")
    
    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print("Predicted answers across attempts:", [
                attempt['predicted_answer'] for attempt in r['attempts']
            ])
            print("---")

def main(model: str, n_attempts: int, analyze_thoughts: bool = False):
    """Main evaluation function that handles gaps in processed indexes."""
    os.makedirs("results", exist_ok=True)
    
    suffix = "_thought_analysis" if analyze_thoughts else ""
    results_file = f"evaluation_results_{model.replace('/', '_')}_pass_at_{n_attempts}{suffix}.json"
    
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
        attempts = make_n_attempts(problem_text, model, n_attempts, analyze_thoughts)
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
    analyze_results(final_results, n_attempts, analyze_thoughts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on AIME 2024 problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--n", type=int, default=1, help="Number of attempts per problem (for pass@n evaluation)")
    parser.add_argument("--analyze-thoughts", action="store_true", help="Analyze thinking patterns in responses")
    args = parser.parse_args()
    
    main(args.model, args.n, args.analyze_thoughts)