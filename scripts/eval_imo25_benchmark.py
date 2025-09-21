"""
Evaluation script for IMO 2025 problems using OptiLLM approaches
Designed to test MARS and other approaches on challenging proof-based problems
"""

import argparse
import json
import os
import logging
import re
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for local OptiLLM server
client = OpenAI(api_key="optillm", base_url="http://localhost:8001/v1")

# Import the actual IMO 2025 problems and reference solutions
from imo25_reference import IMO_2025_PROBLEMS, verify_answer_format, verify_key_insights

SYSTEM_PROMPT = '''You are solving IMO (International Mathematical Olympiad) problems - the most challenging mathematical competition problems for high school students.

Key requirements:
1. **Complete proofs**: Provide rigorous, step-by-step mathematical proofs
2. **Mathematical rigor**: Every step must be logically justified
3. **Clear structure**: Organize your solution with clear logical flow
4. **Proper notation**: Use correct mathematical notation and formatting
5. **Verification**: Double-check your reasoning and conclusions

For existence problems: Provide explicit constructions or proofs of non-existence
For optimization problems: Prove that your answer is optimal
For functional equations: Consider injectivity, surjectivity, and special values
For geometry: Use coordinate systems, trigonometry, or synthetic methods as appropriate
For number theory: Apply divisibility, modular arithmetic, and prime factorization
For combinatorics: Use counting techniques, pigeonhole principle, and extremal arguments

Always conclude with a clear statement of your final answer.

For problems with specific answers, put your final answer in \boxed{} format.'''

def extract_final_answer(solution: str, problem_id: int) -> Dict[str, any]:
    """
    Extract and verify the final answer using official IMO 2025 solutions
    """
    # Use the official answer verification from our reference module
    official_verification = verify_answer_format(problem_id, solution)

    # Legacy extraction for fallback
    result = {
        "extracted_answer": None,
        "confidence": 0.0,
        "extraction_method": None,
        "official_answer_found": official_verification["correct_answer_found"],
        "official_answer_score": official_verification["answer_score"]
    }

    if not solution:
        return result

    # If official answer was found, prioritize it
    if official_verification["correct_answer_found"]:
        result["extracted_answer"] = official_verification["extracted_answer"]
        result["confidence"] = 1.0
        result["extraction_method"] = "official_verification"
        return result

    # Look for boxed answers first
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, solution)
    if boxed_matches:
        result["extracted_answer"] = boxed_matches[-1].strip()  # Take the last one
        result["confidence"] = 0.9
        result["extraction_method"] = "boxed"
        return result

    # Look for "final answer" or "answer:" sections
    answer_patterns = [
        r'final answer[:\s]*([^\n]+)',
        r'answer[:\s]*([^\n]+)',
        r'therefore[:\s]*([^\n]+)',
        r'thus[:\s]*([^\n]+)'
    ]

    solution_lower = solution.lower()
    for pattern in answer_patterns:
        matches = re.findall(pattern, solution_lower)
        if matches:
            result["extracted_answer"] = matches[-1].strip()
            result["confidence"] = 0.5
            result["extraction_method"] = "answer_section"
            break

    return result


def verify_solution_with_llm(problem: str, solution: str, model: str) -> Dict[str, any]:
    """
    Use an LLM as a judge to verify the correctness of a solution
    """
    judge_prompt = f"""You are an expert mathematical judge evaluating IMO solutions.

PROBLEM:
{problem}

STUDENT SOLUTION:
{solution}

Please evaluate this solution and provide:
1. CORRECTNESS SCORE (0-10): How mathematically correct is this solution?
2. COMPLETENESS SCORE (0-10): How complete and rigorous is the proof?
3. KEY INSIGHTS: Did the solution identify the key mathematical insights needed?
4. ERRORS: List any mathematical errors or logical gaps
5. OVERALL ASSESSMENT: Is this solution likely correct?

Provide your assessment in the following format:
CORRECTNESS: [0-10]
COMPLETENESS: [0-10]
KEY_INSIGHTS: [Yes/No]
ERRORS: [List any errors]
OVERALL: [Correct/Incorrect/Partial]
REASONING: [Brief explanation]"""

    try:
        response = client.with_options(timeout=300).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert mathematician and IMO judge."},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=2048,
            temperature=0.1  # Low temperature for consistent judging
        )

        judge_response = response.choices[0].message.content.strip()

        # Parse the structured response
        result = {
            "judge_response": judge_response,
            "correctness_score": 0.0,
            "completeness_score": 0.0,
            "has_key_insights": False,
            "errors_found": [],
            "overall_assessment": "unknown",
            "judge_reasoning": "",
            "success": True
        }

        # Extract scores using regex
        correctness_match = re.search(r'CORRECTNESS:\s*([0-9.]+)', judge_response)
        if correctness_match:
            result["correctness_score"] = float(correctness_match.group(1)) / 10.0

        completeness_match = re.search(r'COMPLETENESS:\s*([0-9.]+)', judge_response)
        if completeness_match:
            result["completeness_score"] = float(completeness_match.group(1)) / 10.0

        insights_match = re.search(r'KEY_INSIGHTS:\s*(Yes|No)', judge_response, re.IGNORECASE)
        if insights_match:
            result["has_key_insights"] = insights_match.group(1).lower() == "yes"

        errors_match = re.search(r'ERRORS:\s*(.+?)(?=OVERALL:|$)', judge_response, re.DOTALL)
        if errors_match:
            errors_text = errors_match.group(1).strip()
            if errors_text and "none" not in errors_text.lower():
                result["errors_found"] = [errors_text]

        overall_match = re.search(r'OVERALL:\s*(Correct|Incorrect|Partial)', judge_response, re.IGNORECASE)
        if overall_match:
            result["overall_assessment"] = overall_match.group(1).lower()

        reasoning_match = re.search(r'REASONING:\s*(.+)', judge_response, re.DOTALL)
        if reasoning_match:
            result["judge_reasoning"] = reasoning_match.group(1).strip()

        return result

    except Exception as e:
        logger.error(f"Error in LLM judge verification: {e}")
        return {
            "judge_response": f"Error: {str(e)}",
            "correctness_score": 0.0,
            "completeness_score": 0.0,
            "has_key_insights": False,
            "errors_found": [f"Judge error: {str(e)}"],
            "overall_assessment": "error",
            "judge_reasoning": "",
            "success": False
        }


def verify_problem_specific_insights(problem_data: Dict, solution: str) -> Dict[str, any]:
    """
    Check for problem-specific insights using our enhanced verification system
    """
    problem_id = problem_data["id"]

    # Use the enhanced verification from our reference module
    insight_verification = verify_key_insights(problem_id, solution)

    return {
        "required_insights_found": len(insight_verification["insights_found"]),
        "total_required_insights": insight_verification["total_insights"],
        "specific_insights": insight_verification["insights_found"],
        "missing_insights": insight_verification["insights_missing"],
        "insight_score": insight_verification["insight_score"]
    }


def extract_solution_quality(response: str) -> Dict[str, any]:
    """
    Analyze the quality of an IMO solution based on mathematical rigor criteria
    """
    analysis = {
        "has_proof_structure": False,
        "uses_mathematical_notation": False,
        "has_logical_steps": False,
        "addresses_all_cases": False,
        "has_conclusion": False,
        "length_score": 0,
        "rigor_indicators": [],
        "completeness_score": 0
    }

    if not response:
        return analysis

    response_lower = response.lower()

    # Check for proof structure
    proof_keywords = ["proof:", "solution:", "we prove", "to show", "suppose", "assume", "let", "consider"]
    if any(keyword in response_lower for keyword in proof_keywords):
        analysis["has_proof_structure"] = True
        analysis["rigor_indicators"].append("proof_structure")

    # Check for mathematical notation
    math_patterns = [r'\$.*\$', r'\\[a-zA-Z]+', r'\\geq', r'\\leq', r'\\in', r'\\mathbb', r'\\sum', r'\\prod']
    if any(re.search(pattern, response) for pattern in math_patterns):
        analysis["uses_mathematical_notation"] = True
        analysis["rigor_indicators"].append("mathematical_notation")

    # Check for logical flow
    logical_words = ["therefore", "thus", "hence", "consequently", "since", "because", "implies", "follows"]
    logical_count = sum(1 for word in logical_words if word in response_lower)
    if logical_count >= 3:
        analysis["has_logical_steps"] = True
        analysis["rigor_indicators"].append("logical_flow")

    # Check for case analysis
    case_words = ["case", "cases", "if", "suppose", "when", "consider"]
    case_count = sum(1 for word in case_words if word in response_lower)
    if case_count >= 2:
        analysis["addresses_all_cases"] = True
        analysis["rigor_indicators"].append("case_analysis")

    # Check for conclusion
    conclusion_words = ["conclude", "final answer", "solution is", "answer:", "qed", "proven", "shown"]
    if any(word in response_lower for word in conclusion_words):
        analysis["has_conclusion"] = True
        analysis["rigor_indicators"].append("clear_conclusion")

    # Length scoring (longer solutions often more complete for IMO)
    word_count = len(response.split())
    if word_count >= 500:
        analysis["length_score"] = 3
    elif word_count >= 200:
        analysis["length_score"] = 2
    elif word_count >= 100:
        analysis["length_score"] = 1
    else:
        analysis["length_score"] = 0

    # Calculate completeness score
    completeness_factors = [
        analysis["has_proof_structure"],
        analysis["uses_mathematical_notation"],
        analysis["has_logical_steps"],
        analysis["addresses_all_cases"],
        analysis["has_conclusion"]
    ]
    analysis["completeness_score"] = sum(completeness_factors) / len(completeness_factors)

    return analysis

def get_llm_response(problem: str, model: str, extra_body: dict = None, timeout: int = 600) -> Dict[str, any]:
    """
    Get response from the LLM for an IMO problem with extended timeout for complex reasoning
    """
    try:
        kwargs = {}
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.with_options(timeout=timeout).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem}
            ],
            max_tokens=8192,  # Extended token limit for complex proofs
            **kwargs
        )

        solution_text = response.choices[0].message.content.strip()
        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
        total_tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0

        return {
            "solution": solution_text,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return {
            "solution": f"Error generating solution: {str(e)}",
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "success": False
        }

def evaluate_solution(problem_data: Dict, solution: str, model: str = "google/gemini-2.5-flash-lite") -> Dict[str, any]:
    """
    Enhanced multi-layer evaluation of IMO solution using:
    - Structural quality analysis (20%)
    - Problem-specific insights verification (40%)
    - LLM-as-judge verification (30%)
    - Overall completeness (10%)
    """
    logger.info(f"Running enhanced evaluation for problem {problem_data['id']}")

    # Layer 1: Structural quality analysis (20% weight)
    quality_analysis = extract_solution_quality(solution)
    structural_score = quality_analysis["completeness_score"]

    # Layer 2: Problem-specific insights verification (40% weight)
    insights_check = verify_problem_specific_insights(problem_data, solution)
    insights_score = insights_check["insight_score"]

    # Layer 3: LLM-as-judge verification (30% weight)
    llm_verification = verify_solution_with_llm(problem_data["problem"], solution, model)
    llm_score = 0.0
    if llm_verification["success"]:
        # Combine correctness and completeness from LLM judge
        llm_score = (llm_verification["correctness_score"] + llm_verification["completeness_score"]) / 2.0

    # Layer 4: Final answer extraction and verification
    answer_extraction = extract_final_answer(solution, problem_data["id"])

    # Use calibrated scoring based on problem type and official answers
    problem_type = problem_data.get("answer_type", "proof")

    if problem_type in ["set", "number", "formula", "threshold"]:
        # For problems with specific answers, heavily weight correct answer
        if answer_extraction["official_answer_found"]:
            answer_score = 1.0  # Perfect score for exact official answer
        else:
            answer_score = answer_extraction["confidence"] * 0.3  # Much lower for non-official

        # Adjust weights for problems with specific answers
        weights = {
            "structural": 0.10,
            "insights": 0.30,
            "llm_judge": 0.20,
            "answer": 0.40  # Higher weight for exact answer match
        }
    else:
        # For proof problems, weight insights and structure more heavily
        answer_score = answer_extraction["confidence"]
        weights = {
            "structural": 0.25,
            "insights": 0.35,
            "llm_judge": 0.30,
            "answer": 0.10
        }

    final_score = (
        structural_score * weights["structural"] +
        insights_score * weights["insights"] +
        llm_score * weights["llm_judge"] +
        answer_score * weights["answer"]
    )

    # Determine confidence based on agreement across layers
    layer_scores = [structural_score, insights_score, llm_score, answer_score]
    score_variance = sum((score - final_score) ** 2 for score in layer_scores) / len(layer_scores)

    if final_score >= 0.8 and score_variance < 0.05:
        confidence = "very_high"
    elif final_score >= 0.7 and score_variance < 0.1:
        confidence = "high"
    elif final_score >= 0.5 and score_variance < 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    # Overall assessment
    is_likely_correct = (
        final_score >= 0.6 and
        insights_score >= 0.5 and
        (llm_verification["overall_assessment"] in ["correct", "partial"] if llm_verification["success"] else True)
    )

    return {
        "correctness_score": final_score,
        "is_likely_correct": is_likely_correct,
        "confidence": confidence,

        # Detailed breakdown
        "layer_scores": {
            "structural_quality": structural_score,
            "insights_verification": insights_score,
            "llm_judge": llm_score,
            "answer_extraction": answer_score
        },
        "weights_used": weights,
        "score_variance": score_variance,

        # Detailed component results
        "quality_analysis": quality_analysis,
        "insights_check": insights_check,
        "llm_verification": llm_verification,
        "answer_extraction": answer_extraction,

        # Legacy compatibility
        "evaluation_method": "enhanced_multi_layer"
    }

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []

    results.append(result)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def analyze_results(results: List[Dict], approach_name: str = None):
    """Analyze and print comprehensive statistics of IMO evaluation results"""
    if not results:
        print("No results to analyze")
        return

    total_problems = len(results)
    likely_correct = sum(1 for r in results if r['evaluation']['is_likely_correct'])
    high_confidence = sum(1 for r in results if r['evaluation']['confidence'] == 'high')

    avg_correctness = sum(r['evaluation']['correctness_score'] for r in results) / total_problems
    avg_completeness = sum(r['evaluation']['quality_analysis']['completeness_score'] for r in results) / total_problems

    total_reasoning_tokens = sum(r['response']['reasoning_tokens'] for r in results)
    avg_reasoning_tokens = total_reasoning_tokens / total_problems

    print("\n" + "="*80)
    print(f"IMO 2025 Evaluation Results - {approach_name or 'Baseline'}")
    print("="*80)
    print(f"Total problems attempted: {total_problems}")
    print(f"Likely correct solutions: {likely_correct} ({likely_correct/total_problems:.1%})")
    print(f"High confidence solutions: {high_confidence} ({high_confidence/total_problems:.1%})")
    print(f"Average correctness score: {avg_correctness:.3f}")
    print(f"Average completeness score: {avg_completeness:.3f}")
    print(f"Total reasoning tokens used: {total_reasoning_tokens:,}")
    print(f"Average reasoning tokens per problem: {avg_reasoning_tokens:.0f}")

    # Problem type breakdown
    print(f"\nProblem Type Breakdown:")
    type_stats = {}
    for result in results:
        prob_type = result['problem_data']['type']
        if prob_type not in type_stats:
            type_stats[prob_type] = {'total': 0, 'correct': 0, 'scores': []}
        type_stats[prob_type]['total'] += 1
        if result['evaluation']['is_likely_correct']:
            type_stats[prob_type]['correct'] += 1
        type_stats[prob_type]['scores'].append(result['evaluation']['correctness_score'])

    for prob_type, stats in type_stats.items():
        accuracy = stats['correct'] / stats['total']
        avg_score = sum(stats['scores']) / len(stats['scores'])
        print(f"  {prob_type}: {stats['correct']}/{stats['total']} ({accuracy:.1%}) - Avg score: {avg_score:.3f}")

    # Detailed problem results
    print(f"\nDetailed Results:")
    print("-" * 80)
    for result in results:
        prob_id = result['problem_data']['id']
        prob_type = result['problem_data']['type']
        score = result['evaluation']['correctness_score']
        confidence = result['evaluation']['confidence']
        tokens = result['response']['reasoning_tokens']

        status = "✓" if result['evaluation']['is_likely_correct'] else "✗"
        print(f"Problem {prob_id} ({prob_type}): {status} Score: {score:.3f} ({confidence}) - {tokens:,} tokens")

    # Quality analysis summary
    print(f"\nSolution Quality Analysis:")
    print("-" * 40)
    quality_metrics = [
        "has_proof_structure", "uses_mathematical_notation", "has_logical_steps",
        "addresses_all_cases", "has_conclusion"
    ]

    for metric in quality_metrics:
        count = sum(1 for r in results if r['evaluation']['quality_analysis'][metric])
        percentage = count / total_problems
        print(f"{metric.replace('_', ' ').title()}: {count}/{total_problems} ({percentage:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on IMO 2025 problems")
    parser.add_argument("--model", type=str, required=True,
                       help="Model to use (e.g., google/gemma-2.5-flash-lite)")
    parser.add_argument("--approach", type=str, default="none",
                       help="OptiLLM approach to use (none, mars, moa, bon, etc.)")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for each problem (default: 600)")
    parser.add_argument("--problems", type=str,
                       help="Comma-separated list of problem IDs to evaluate (e.g., '1,3,5')")

    args = parser.parse_args()

    # Setup results directory and filename
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/imo25_{args.model.replace('/', '_')}_{args.approach}_{timestamp}.json"

    # Determine which problems to evaluate
    if args.problems:
        problem_ids = [int(x.strip()) for x in args.problems.split(',')]
        problems_to_evaluate = [p for p in IMO_2025_PROBLEMS if p['id'] in problem_ids]
    else:
        problems_to_evaluate = IMO_2025_PROBLEMS

    print(f"Evaluating {len(problems_to_evaluate)} IMO 2025 problems")
    print(f"Model: {args.model}")
    print(f"Approach: {args.approach}")
    print(f"Results will be saved to: {results_file}")

    # Prepare extra_body for approach
    extra_body = {"optillm_approach": args.approach} if args.approach != "none" else None

    # Evaluate each problem
    for problem_data in tqdm(problems_to_evaluate, desc="Solving IMO problems"):
        logger.info(f"Evaluating Problem {problem_data['id']}: {problem_data['type']}")

        start_time = time.time()

        # Get LLM response
        response = get_llm_response(
            problem_data['problem'],
            args.model,
            extra_body,
            args.timeout
        )

        solve_time = time.time() - start_time

        # Evaluate solution quality with enhanced multi-layer approach
        evaluation = evaluate_solution(problem_data, response['solution'], args.model)

        # Compile result
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "approach": args.approach,
            "problem_data": problem_data,
            "response": response,
            "evaluation": evaluation,
            "solve_time_seconds": solve_time
        }

        # Save result immediately
        save_result(results_file, result)

        logger.info(f"Problem {problem_data['id']} completed - Score: {evaluation['correctness_score']:.3f}")

    # Load all results and analyze
    final_results = load_existing_results(results_file)
    analyze_results(final_results, args.approach)

    print(f"\nEvaluation complete! Results saved to: {results_file}")

if __name__ == "__main__":
    main()