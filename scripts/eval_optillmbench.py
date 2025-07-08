#!/usr/bin/env python3
import argparse
import time
import json
import os
from typing import Dict, List, Any, Tuple
import datasets
from datasets import load_dataset
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the approaches to test
# Each approach is (name, description, extra_body_params)
APPROACHES = [
    ("none", "Baseline without any optimization", {}),
    ("leap", "LEAP Approach", {}),
    ("rto", "Round Trip Optimization", {}),
    ("cot_reflection", "Chain of Thought with Reflection", {}),
    ("self_consistency", "Self Consistency Check", {}),
    ("plansearch", "Planning with Search", {}),
    ("re2", "ReRead Approach", {}),
    ("z3", "Z3 Solver for Mathematical Problems", {}),
    ("coc", "Chain of Code", {}),
    ("executecode" , "Execute Code", {}),
    ("spl", "System Prompt Learning", {})
]

# Define test-time compute approaches for sequential and parallel scaling
TEST_TIME_COMPUTE_APPROACHES = [
    # Baseline
    ("none", "Baseline without any optimization", {}),
    
    # Sequential test-time compute using thinkdeeper with different thinking budgets
    ("thinkdeeper_8k", "ThinkDeeper with 8K thinking tokens", {
        "decoding": "thinkdeeper",
        "max_thinking_tokens": 8000
    }),
    ("thinkdeeper_16k", "ThinkDeeper with 16K thinking tokens", {
        "decoding": "thinkdeeper", 
        "max_thinking_tokens": 16000
    }),
    ("thinkdeeper_32k", "ThinkDeeper with 32K thinking tokens", {
        "decoding": "thinkdeeper",
        "max_thinking_tokens": 32000
    }),
    
    # Parallel test-time compute using majority voting with different k values
    ("majority_voting_6", "Majority Voting with k=6", {"k": 6}),
    ("majority_voting_36", "Majority Voting with k=36", {"k": 36}),
    ("majority_voting_60", "Majority Voting with k=60", {"k": 60}),
]

def load_optillm_bench() -> datasets.Dataset:
    """Load the OptiLLM Bench dataset."""
    try:
        dataset = load_dataset("codelion/optillmbench")
        return dataset["test"]  # We use the test split for evaluation
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def extract_gsm8k_answer(text: str) -> float:
    """Extract numerical answer after ### from GSM8K responses."""
    match = re.search(r'###\s*(-?\d*\.?\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def remove_thinking_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from the response.
    If there's a </think> tag, only keep the content after it.
    """
    if not text:
        return text
        
    # Check if there's a thinking block
    if '</think>' in text:
        # Get everything after the last </think> tag
        parts = text.split('</think>')
        return parts[-1].strip()
    
    # If no thinking blocks, return original text
    return text

def extract_choice_index_from_question(question: str, answer: str) -> int:
    """
    Extract the index of the correct answer from a multiple-choice question.
    
    Args:
        question: The question text containing choices
        answer: The correct answer (just the text, no index)
    
    Returns:
        int: The index of the correct answer, or -1 if not found
    """
    # Look for a pattern like "N. answer" in the question
    answer_clean = answer.strip().lower()
    
    # Debug logging for critical examples
    logger.debug(f"Looking for answer: '{answer_clean}' in question")
    
    # Check for "Choices:" marker in the question
    if "choices:" in question.lower():
        # Split the question by lines after "Choices:"
        choices_section = question.lower().split("choices:")[1].strip()
        
        # Log the choices section
        logger.debug(f"Choices section: '{choices_section}'")
        
        # Try different approaches to extract choices
        
        # 1. If it's all on one line, use a more comprehensive regex
        if '\n' not in choices_section:
            # This pattern matches "N. text" where N is a digit and text is any text up to the next number or end
            all_choices = re.findall(r'(\d+)\s*\.\s*([^0-9.]+?)(?=\s*\d+\s*\.|$)', choices_section)
            
            logger.debug(f"Single line choices found: {all_choices}")
            
            for idx, choice_text in all_choices:
                choice_text_clean = choice_text.strip()
                if choice_text_clean.lower() == answer_clean:
                    logger.debug(f"Found match at index {idx}: '{choice_text_clean}'")
                    return int(idx)
        
        # 2. Try splitting by newlines
        choices = choices_section.split("\n")
        
        for i, choice in enumerate(choices):
            choice = choice.strip()
            if not choice:
                continue
                
            logger.debug(f"Checking choice {i}: '{choice}'")
            
            # Try to extract the index and choice text
            match = re.match(r'\s*(\d+)\s*\.\s*(.*)', choice)
            if match:
                idx = int(match.group(1))
                choice_text = match.group(2).strip()
                
                logger.debug(f"Parsed choice: index={idx}, text='{choice_text}'")
                
                if choice_text.lower() == answer_clean:
                    logger.debug(f"Found exact match at index {idx}")
                    return idx
        
        # 3. Fallback: just look for any occurrence of the number followed by the answer
        pattern = r'(\d+)\s*\.\s*' + re.escape(answer_clean)
        match = re.search(pattern, choices_section)
        if match:
            logger.debug(f"Fallback match found at index {match.group(1)}")
            return int(match.group(1))
    
    logger.debug("No match found for answer in choices")
    return -1

def is_numeric_only_response(response: str) -> Tuple[bool, int]:
    """
    Check if the response is just a numeric value, possibly with whitespace and newlines.
    
    Args:
        response: The response text to check
        
    Returns:
        Tuple of (is_numeric, value)
    """
    # Strip all whitespace, including newlines
    clean_response = re.sub(r'\s', '', response)
    
    # Check if it's just a number
    if clean_response.isdigit():
        return True, int(clean_response)
    
    return False, -1

def evaluate_response(response: str, ground_truth: str, category: str, question: str = None) -> bool:
    """
    Evaluate if the response matches the ground truth based on category.
    
    Args:
        response: Model's response
        ground_truth: Correct answer
        category: Problem category (gsm8k, mmlu_math, boolq, aqua_rat)
        question: Original question text, needed for MMLU evaluation
    
    Returns:
        bool: Whether the response is correct
    """
    if not response or not ground_truth:
        return False
    
    # First, remove any thinking blocks
    response = remove_thinking_blocks(response)
        
    if category == "gsm8k":
        # Extract numerical answers after ### and compare
        response_num = extract_gsm8k_answer(response)
        ground_truth_num = extract_gsm8k_answer(ground_truth)
        
        if response_num is None or ground_truth_num is None:
            return False
            
        # Compare with small tolerance for floating point
        return abs(response_num - ground_truth_num) < 1e-6
    elif category == "mmlu_math":
        # Special handling for MMLU-math multiple choice questions
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        
        # Case 1: Exact match of answer text
        if response_clean == ground_truth_clean:
            logger.debug("Exact text match")
            return True
            
        # For other cases, we need to find what index corresponds to the ground truth
        if question:
            correct_index = extract_choice_index_from_question(question, ground_truth)
            
            if correct_index >= 0:
                # Case 2: Check if response is just the digit (most common LLM response for indices)
                is_numeric, value = is_numeric_only_response(response)
                if is_numeric and value == correct_index:
                    logger.debug(f"Numeric match: response '{response}' -> {value} matches index {correct_index}")
                    return True
                
                # Case 3: Check if response is "index. answer"
                if re.search(fr"{correct_index}\s*\.\s*{re.escape(ground_truth_clean)}", response_clean):
                    logger.debug("Pattern match for 'index. answer'")
                    return True
                
                # Case 4: Check if response contains both the index and the answer text
                if str(correct_index) in response_clean and ground_truth_clean in response_clean:
                    logger.debug("Contains both index and answer")
                    return True
        
        return False
    else:
        # For boolq and aqua_rat, exact match is required
        # Clean up both strings for comparison
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        return response_clean == ground_truth_clean

def get_prompt_for_category(question: str, category: str) -> str:
    """
    Generate appropriate prompt based on category.
    """
    if category == "gsm8k":
        return (
            f"Solve this math problem step by step. After solving, provide the final "
            f"numerical answer after '### ' (three hash symbols and a space).\n\n"
            f"Question: {question}\n\n"
            f"Show your work, then give the final answer after '### '."
        )
    elif category == "mmlu_math":
        return (
            f"Solve this math problem. Provide only the answer with no explanation.\n\n"
            f"Question: {question}"
        )
    elif category == "boolq":
        return (
            f"Answer this yes/no question with only 'yes' or 'no'.\n\n"
            f"Question: {question}"
        )
    elif category == "aqua_rat":
        return (
            f"Choose the correct answer. Provide only the letter choice with no explanation.\n\n"
            f"Question: {question}"
        )
    else:
        return f"Question: {question}"

def evaluate_model(
    client: OpenAI,
    model: str,
    dataset: datasets.Dataset,
    approach: str,
    approach_extra_body: Dict[str, Any] = None,
    max_samples: int = None
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a model on the dataset using a specific approach.
    Returns metrics and detailed results.
    """
    metrics = {
        "total_correct": 0,
        "total_time": 0,
        "samples": 0,
    }
    
    # Initialize category-specific metrics
    category_metrics = {}
    
    # Detailed results for each example
    detailed_results = []
    
    # Prepare the dataset
    examples = dataset if max_samples is None else dataset.select(range(max_samples))
    
    # Create model name with approach - handle special cases
    if approach == "none":
        full_model_name = model
    elif approach.startswith("thinkdeeper_"):
        # For thinkdeeper, use base model name (decoding is passed in extra_body)
        full_model_name = model
    elif approach.startswith("majority_voting_"):
        # For majority voting, use majority_voting prefix
        full_model_name = f"majority_voting-{model}"
    else:
        # Standard approach prefix
        full_model_name = f"{approach}-{model}"
    
    for example in tqdm(examples, desc=f"Evaluating {approach}"):
        try:
            # Get appropriate prompt for the category
            prompt = get_prompt_for_category(example['question'], example['category'])
            
            # Record start time
            start_time = time.time()
            
            # Prepare extra_body parameters
            extra_body = {"spl_learning": False}
            if approach_extra_body:
                extra_body.update(approach_extra_body)
            
            # Make API call
            response = client.chat.completions.create(
                model=full_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant focused on providing precise answers in the requested format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
                extra_body=extra_body,
            )
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            # Get the response text
            response_text = response.choices[0].message.content
            
            # Also store the raw response for reference
            raw_response = response_text
            
            # Process the response to remove thinking blocks
            processed_response = remove_thinking_blocks(response_text)
            
            # Evaluate the processed response
            is_correct = evaluate_response(
                processed_response,
                example['answer'],
                example['category'],
                example['question']  # Pass the question for MMLU evaluation
            )
            
            # Update metrics
            metrics["total_correct"] += int(is_correct)
            metrics["total_time"] += time_taken
            metrics["samples"] += 1
            
            # Update category metrics
            if example['category'] not in category_metrics:
                category_metrics[example['category']] = {
                    "correct": 0,
                    "total": 0,
                    "time": 0
                }
            category_metrics[example['category']]["correct"] += int(is_correct)
            category_metrics[example['category']]["total"] += 1
            category_metrics[example['category']]["time"] += time_taken
            
            # Check if thinking blocks were removed
            has_thinking = '</think>' in raw_response
            
            # Record detailed result
            detailed_results.append({
                "id": example['id'],
                "category": example['category'],
                "correct": is_correct,
                "time_taken": time_taken,
                "raw_response": raw_response,
                "processed_response": processed_response if has_thinking else None,
                "has_thinking": has_thinking,
                "ground_truth": example['answer']
            })
            
        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            continue
    
    # Calculate final metrics
    final_metrics = {
        "accuracy": metrics["total_correct"] / metrics["samples"] if metrics["samples"] > 0 else 0,
        "average_time": metrics["total_time"] / metrics["samples"] if metrics["samples"] > 0 else 0,
        "total_time": metrics["total_time"],
        "total_samples": metrics["samples"],
    }
    
    # Add category-specific metrics
    for category, cat_metrics in category_metrics.items():
        final_metrics[f"{category}_accuracy"] = cat_metrics["correct"] / cat_metrics["total"]
        final_metrics[f"{category}_average_time"] = cat_metrics["time"] / cat_metrics["total"]
    
    return final_metrics, detailed_results

def save_results(metrics: Dict[str, float], detailed_results: List[Dict[str, Any]], 
                model: str, approach: str, output_dir: str):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model-specific directory
    model_dir = os.path.join(output_dir, model.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    base_filename = os.path.join(model_dir, f"{approach}_{timestamp}")
    
    # Save metrics
    with open(f"{base_filename}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    with open(f"{base_filename}_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create a summary DataFrame for easier analysis
    df = pd.DataFrame([
        {k: v for k, v in result.items() if k != 'raw_response' and k != 'processed_response'}
        for result in detailed_results
    ])
    df.to_csv(f"{base_filename}_summary.csv", index=False)
    
    logger.info(f"Results saved to {base_filename}_*")

def generate_report(all_metrics: Dict[str, Dict[str, float]], output_dir: str):
    """Generate a comprehensive report comparing all approaches."""
    report = []
    
    # Header
    report.append("# OptiLLM Bench Evaluation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall Results Table
    report.append("## Overall Results")
    headers = ["Approach", "Accuracy", "Avg Time (s)", "Total Time (s)"]
    rows = []
    
    for approach, metrics in all_metrics.items():
        rows.append([
            approach,
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['average_time']:.2f}",
            f"{metrics['total_time']:.2f}"
        ])
    
    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(rows, columns=headers)
    report.append(df.to_markdown())
    
    # Category-wise Results
    report.append("\n## Results by Category")
    categories = ["gsm8k", "mmlu_math", "boolq", "aqua_rat"]
    
    for category in categories:
        report.append(f"\n### {category.upper()}")
        headers = ["Approach", "Accuracy", "Avg Time (s)"]
        rows = []
        
        for approach, metrics in all_metrics.items():
            if f"{category}_accuracy" in metrics:
                rows.append([
                    approach,
                    f"{metrics[f'{category}_accuracy']*100:.2f}%",
                    f"{metrics[f'{category}_average_time']:.2f}"
                ])
        
        df = pd.DataFrame(rows, columns=headers)
        report.append(df.to_markdown())
    
    # Save report
    report_path = f"{output_dir}/evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n\n".join(report))
    
    logger.info(f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on OptiLLM Bench")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", 
                        help="Base URL for API endpoint")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output-dir", default="results", 
                        help="Directory to save results")
    parser.add_argument("--approaches", nargs="+", 
                        help="Specific approaches to evaluate (default: all)")
    parser.add_argument("--test-time-compute", action="store_true",
                        help="Evaluate test-time compute approaches (sequential and parallel scaling)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug logging if specified
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url
    )
    
    # Load dataset
    dataset = load_optillm_bench()
    
    # Determine which approaches to evaluate
    approaches_to_test = (
        [a[0] for a in APPROACHES if a[0] in args.approaches]
        if args.approaches
        else [a[0] for a in APPROACHES]
    )
    
    # Store all metrics for final report
    all_metrics = {}
    
    # Evaluate each approach
    for approach in approaches_to_test:
        logger.info(f"Evaluating approach: {approach}")
        
        try:
            metrics, detailed_results = evaluate_model(
                client,
                args.model,
                dataset,
                approach,
                args.max_samples
            )
            
            all_metrics[approach] = metrics
            
            # Save results for this approach
            save_results(metrics, detailed_results, args.model, approach, 
                        args.output_dir)
            
            logger.info(f"Completed evaluation for {approach}")
            logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
            logger.info(f"Average time per sample: {metrics['average_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error evaluating approach {approach}: {e}")
            continue
    
    # Generate final report
    generate_report(all_metrics, args.output_dir)

if __name__ == "__main__":
    main()