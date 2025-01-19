import argparse
import json
import os
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client (only used for chat completions now)
client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@dataclass
class RTCConfig:
    similarity_threshold: float = 0.6  # Adjusted threshold for TF-IDF similarity
    max_retries: int = 3  # Maximum number of retries for API calls
    retry_delay: int = 1  # Delay between retries in seconds

def extract_first_turn_content(turns: List[Dict]) -> str:
    """Extract the content from the first turn in the conversation."""
    if not turns or not isinstance(turns, list):
        return ""
    return turns[0].get("content", "")

def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF vectorization.
    This is a local implementation that doesn't require any external API.
    """
    try:
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def get_llm_response(messages: List[Dict], model: str) -> Optional[str]:
    """Get response from the LLM with retry logic."""
    for attempt in range(RTCConfig.max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
            if attempt < RTCConfig.max_retries - 1:
                time.sleep(RTCConfig.retry_delay)
            continue
    return None

def perform_rtc_evaluation(query: str, model: str) -> Tuple[bool, float, Dict]:
    """
    Perform Round-Trip Correctness evaluation.
    
    Args:
        query: Original query
        model: Model name to use
        
    Returns:
        Tuple of (passed_rtc, similarity_score, evaluation_details)
    """
    # Step 1: Get initial response
    response_1 = get_llm_response([
        {"role": "user", "content": query}
    ], model)
    
    if not response_1:
        return False, 0.0, {"error": "Failed to get initial response"}
        
    # Step 2: Generate alternate query
    inverse_prompt = f"""Given this query and response pair, generate a new query that would lead to a similar response. Focus on the key aspects that would generate equivalent content:

Original Query: {query}
Response: {response_1}

Generate a new query that would elicit a similar response:"""

    alternate_query = get_llm_response([
        {"role": "user", "content": inverse_prompt}
    ], model)
    
    if not alternate_query:
        return False, 0.0, {"error": "Failed to generate alternate query"}
        
    # Step 3: Get response for alternate query
    response_2 = get_llm_response([
        {"role": "user", "content": alternate_query}
    ], model)
    
    if not response_2:
        return False, 0.0, {"error": "Failed to get second response"}
        
    # Step 4: Compute similarity using local TF-IDF based similarity
    similarity_score = compute_similarity(response_1, response_2)
    
    evaluation_details = {
        "original_query": query,
        "response_1": response_1,
        "alternate_query": alternate_query,
        "response_2": response_2,
        "similarity_score": similarity_score
    }
    
    return similarity_score >= RTCConfig.similarity_threshold, similarity_score, evaluation_details

def evaluate_dataset(model: str, output_file: str):
    """Evaluate the dataset using RTC methodology."""
    # Load dataset
    dataset = load_dataset("lmarena-ai/arena-hard-auto-v0.1")
    
    results = []
    passed_rtc_count = 0
    total_examples = 0
    
    for item in tqdm(dataset["train"], desc="Evaluating examples"):
        query = extract_first_turn_content(item["turns"])
        if not query:
            continue
            
        passed_rtc, similarity_score, details = perform_rtc_evaluation(query, model)
        
        result = {
            "id": total_examples,
            "query": query,
            "passed_rtc": passed_rtc,
            "similarity_score": similarity_score,
            "evaluation_details": details
        }
        
        results.append(result)
        if passed_rtc:
            passed_rtc_count += 1
        total_examples += 1
        
        # Save results after each example
        with open(output_file, 'w') as f:
            json.dump({
                "model": model,
                "total_examples": total_examples,
                "passed_rtc": passed_rtc_count,
                "rtc_pass_rate": passed_rtc_count / total_examples if total_examples > 0 else 0,
                "results": results
            }, f, indent=2)
            
    # Print final summary
    logger.info(f"\nEvaluation Summary for {model}:")
    logger.info(f"Total examples evaluated: {total_examples}")
    logger.info(f"Examples passing RTC: {passed_rtc_count}")
    logger.info(f"RTC pass rate: {passed_rtc_count / total_examples * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on arena-hard-auto dataset using RTC")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use")
    parser.add_argument("--output", type=str, default="rtc_eval_results.json", 
                      help="Output file for results")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", args.output)
    
    # Run evaluation
    evaluate_dataset(args.model, output_file)

if __name__ == "__main__":
    main()