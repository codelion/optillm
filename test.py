import argparse
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import logging
from openai import OpenAI

from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.rto import round_trip_optimization
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.pvg import inference_time_pv_game
from optillm.z3_solver import Z3SolverSystem
from optillm.rstar import RStar
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.agent import agent_approach

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI API configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Mock OpenAI client for testing purposes
class MockOpenAIClient:
    def chat_completions_create(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'Mock response'})()})]
        return MockResponse()

# Configuration for approaches
APPROACHES = {
    'mcts': chat_with_mcts,
    'bon': best_of_n_sampling,
    'moa': mixture_of_agents,
    'rto': round_trip_optimization,
    'self_consistency': advanced_self_consistency_approach,
    'pvg': inference_time_pv_game,
    'z3': lambda s, q, c, m: Z3SolverSystem(s, c, m).process_query(q),
    'rstar': lambda s, q, c, m: RStar(s, c, m).solve(q),
    'cot_reflection': cot_reflection,
    'plansearch': plansearch,
    'leap': leap,
    'agent': agent_approach,
}

def load_test_cases(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def run_approach(approach_name: str, system_prompt: str, query: str, client, model: str) -> Dict:
    start_time = time.time()
    try:
        approach_func = APPROACHES[approach_name]
        result = approach_func(system_prompt, query, client, model)
        end_time = time.time()
        return {
            'approach': approach_name,
            'result': result,
            'time': end_time - start_time,
            'status': 'success'
        }
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error in {approach_name}: {str(e)}")
        return {
            'approach': approach_name,
            'result': str(e),
            'time': end_time - start_time,
            'status': 'error'
        }

def run_test_case(test_case: Dict, approaches: List[str], client, model: str) -> Dict:
    system_prompt = test_case['system_prompt']
    query = test_case['query']
    results = []

    with ThreadPoolExecutor() as executor:
        future_to_approach = {executor.submit(run_approach, approach, system_prompt, query, client, model): approach for approach in approaches}
        for future in as_completed(future_to_approach):
            results.append(future.result())

    return {
        'test_case': test_case,
        'results': results
    }

def run_tests(test_cases: List[Dict], approaches: List[str], client, model: str) -> List[Dict]:
    results = []
    for test_case in test_cases:
        result = run_test_case(test_case, approaches, client, model)
        results.append(result)
        logger.info(f"Completed test case: {test_case['name']}")
    return results

def print_summary(results: List[Dict]):
    print("\n=== Test Results Summary ===")
    for test_result in results:
        print(f"\nTest Case: {test_result['test_case']['name']}")
        for approach_result in test_result['results']:
            status = "✅" if approach_result['status'] == 'success' else "❌"
            print(f"  {status} {approach_result['approach']}: {approach_result['time']:.2f}s")
            if approach_result['status'] == 'error':
                print(f"     Error: {approach_result['result']}")

def main():
    parser = argparse.ArgumentParser(description="Test different LLM inference approaches.")
    parser.add_argument("--test_cases", type=str, default="test_cases.json", help="Path to test cases JSON file")
    parser.add_argument("--approaches", nargs='+', default=list(APPROACHES.keys()), help="Approaches to test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for testing")
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_cases)
    results = run_tests(test_cases, args.approaches, client, args.model)
    print_summary(results)

    # Optionally, save detailed results to a file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
