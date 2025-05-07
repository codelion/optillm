import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

def create_and_upload_dataset():
    # Define the filenames
    standard_file = "results/evaluation_results_math500_meta-llama_llama-3.2-1b-instruct_standard.json"
    cot_file = "results/evaluation_results_math500_meta-llama_llama-3.2-1b-instruct_cot.json"
    gibberish_file = "results/evaluation_results_math500_meta-llama_llama-3.2-1b-instruct_gibberish.json"
    
    # Load the JSON data
    print("Loading data files...")
    with open(standard_file, 'r') as f:
        standard_data = json.load(f)
    
    with open(cot_file, 'r') as f:
        cot_data = json.load(f)
    
    with open(gibberish_file, 'r') as f:
        gibberish_data = json.load(f)
    
    # Create datasets
    print("Creating dataset splits...")
    standard_dataset = Dataset.from_dict({
        'index': [item['index'] for item in standard_data],
        'problem': [item['problem'] for item in standard_data],
        'prompt_type': [item['prompt_type'] for item in standard_data],
        'response': [item['response'] for item in standard_data],
        'correct_answer': [item['correct_answer'] for item in standard_data],
        'predicted_answer': [item['predicted_answer'] if item['predicted_answer'] is not None else "" for item in standard_data],
        'is_correct': [item['is_correct'] for item in standard_data],
    })
    
    cot_dataset = Dataset.from_dict({
        'index': [item['index'] for item in cot_data],
        'problem': [item['problem'] for item in cot_data],
        'prompt_type': [item['prompt_type'] for item in cot_data],
        'response': [item['response'] for item in cot_data],
        'correct_answer': [item['correct_answer'] for item in cot_data],
        'predicted_answer': [item['predicted_answer'] if item['predicted_answer'] is not None else "" for item in cot_data],
        'is_correct': [item['is_correct'] for item in cot_data],
    })
    
    gibberish_dataset = Dataset.from_dict({
        'index': [item['index'] for item in gibberish_data],
        'problem': [item['problem'] for item in gibberish_data],
        'prompt_type': [item['prompt_type'] for item in gibberish_data],
        'response': [item['response'] for item in gibberish_data],
        'correct_answer': [item['correct_answer'] for item in gibberish_data],
        'predicted_answer': [item['predicted_answer'] if item['predicted_answer'] is not None else "" for item in gibberish_data],
        'is_correct': [item['is_correct'] for item in gibberish_data],
    })
    
    # Combine into a DatasetDict with three splits
    dataset_dict = DatasetDict({
        'standard': standard_dataset,
        'cot': cot_dataset,
        'gibberish': gibberish_dataset
    })
    
    # Login to Hugging Face Hub
    # You'll need to set the HF_TOKEN environment variable or enter when prompted
    print("Logging in to Hugging Face Hub...")
    try:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=True)
        else:
            login(add_to_git_credential=True)
    except Exception as e:
        print(f"Error logging in: {e}")
        print("You can set your HF token as an environment variable: export HF_TOKEN='your_token'")
        return
    
    # Define dataset name - adjust as needed
    dataset_name = "math500-cot-experiment"
    repository_id = f"{input('Enter your Hugging Face username: ')}/{dataset_name}"
    
    # Push to hub
    print(f"Uploading dataset to {repository_id}...")
    dataset_dict.push_to_hub(
        repository_id,
        private=False,
        token=os.environ.get("HF_TOKEN"),
    )
    
    print(f"Dataset uploaded successfully! View it at: https://huggingface.co/datasets/{repository_id}")
    
    # Create a README.md file with dataset information
    readme_content = f"""# MATH-500 Chain of Thought Experiment Dataset

This dataset contains the results of an experiment testing different prompting strategies (standard, chain of thought, and gibberish chain of thought) on the MATH-500 benchmark using the Llama-3.2-1B-Instruct model.

## Dataset Structure

The dataset is split into three parts:
- `standard`: Direct prompting with no reasoning steps ({len(standard_dataset)} examples)
- `cot`: Chain of thought prompting with structured reasoning ({len(cot_dataset)} examples)
- `gibberish`: Gibberish chain of thought with meaningless text before the answer ({len(gibberish_dataset)} examples)

## Fields

- `index`: Problem index
- `problem`: The math problem
- `prompt_type`: Type of prompting used (standard, cot, or gibberish)
- `response`: Full model response
- `correct_answer`: The correct answer from the MATH-500 benchmark
- `predicted_answer`: The model's predicted answer
- `is_correct`: Whether the prediction was correct

## Results Summary

| Prompt Type | Problems | Correct | Accuracy |
|-------------|----------|---------|----------|
| Standard | {len(standard_dataset)} | {sum(1 for item in standard_data if item['is_correct'])} | {sum(1 for item in standard_data if item['is_correct'])/len(standard_dataset):.2%} |
| CoT | {len(cot_dataset)} | {sum(1 for item in cot_data if item['is_correct'])} | {sum(1 for item in cot_data if item['is_correct'])/len(cot_dataset):.2%} |
| Gibberish | {len(gibberish_dataset)} | {sum(1 for item in gibberish_data if item['is_correct'])} | {sum(1 for item in gibberish_data if item['is_correct'])/len(gibberish_dataset):.2%} |

This dataset was created to help test whether Chain of Thought (CoT) reasoning provides value through coherent reasoning or simply by giving models more computation time/tokens to produce answers.
"""
    
    # Upload the README using the HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repository_id,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    
    print("README uploaded with dataset information.")

if __name__ == "__main__":
    create_and_upload_dataset()
