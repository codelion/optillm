import math
import time
import json
import pickle
import os
import random
import sys
from collections import namedtuple
from typing import Dict, List, Tuple, Union

from openai import OpenAI
import pandas as pd
from tqdm import tqdm


CHAT_MODELS = ['supernova-medius']
OPENAI_MODELS = CHAT_MODELS
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

# # load_dotenv()
dummy_api_key = None
client = OpenAI(
    api_key= None,
    base_url="https://inference-time.research.arcee.ai/v1"
)


def create_prompts(examples: List[Example]) -> Tuple[List[str], List[Example]]:
    """Creates a list of prompts given a list of examples.
    Also returns the list of examples, since we can't evaluate some examples if they're in the few-shot set.
    Model name parameter is only used for chain-of-thought prompts (because we need to call the model to generate the reasoning traces)."""
    return [zero_shot_prompt(example) for example in examples], examples


def generate_prompt_from_examples(json_data, with_explanations=True):
    output = ""
    for q in json_data["questions"]:
        output += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            output += f'({choice}) {value}\n'
        
        if with_explanations:
            output += f"Let's think step by step: \n{q['explanation']}\n"
        
        output += f'The correct answer is ({q["correct_answer"]})\n'
    
    return output


def load_from_json(path: str) -> Dict:
    """Load json from a file."""
    with open(path, 'r') as f:
        return json.load(f)


def base_prompt(example: Example) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight: 
    https://arxiv.org/abs/2211.03540"""
    prompt = f"What is the correct answer to this question: {example.question}"
    prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    return prompt


def zero_shot_prompt(example: Example) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight: 
    https://arxiv.org/abs/2211.03540"""
    prompt = base_prompt(example)
    prompt += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""
    return prompt


def load_from_text(path: str) -> str:
    """Load text from a file."""
    with open(path, 'r') as f:
        return f.read()


def load_examples(path: str, seed: int) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']))
        return example

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]


def get_api_type(model_name: str):
    if model_name in OPENAI_MODELS:
        return 'openai'
    else:
        raise ValueError(f"Model name {model_name} not supported.")


def select_and_call_model(prompt: str,
                          model_name: str,
                          call_type: str,
                          temperature: float = 0.0,
                          stop: Union[str, List[str]] = None,
                          max_tokens: int = 1000,
                          inference_method: str = "mcts") -> Union[str, Dict[str, List[Union[str, float]]]]:
    """Selects the appropriate model and calls it with the given prompt."""
    assert inference_method != "none", "Inference method cannot be 'none'"
    
    content = f"<optillm_approach>{inference_method}</optillm_approach> {prompt}"
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=messages,
    )
    return response


def call_model_with_retries(prompt: str,
                            model_name: str = None,
                            call_type: str = 'logprobs',
                            temperature: float = 0.0,
                            stop: str = None,
                            max_tokens: int = 1000,
                            inference_method: str = "mcts") -> Union[str, Dict[str, List[Union[str, float]]]]:
    if model_name is None:
        raise ValueError("Model name must be specified.")
   
    response = select_and_call_model(prompt, model_name, call_type, temperature, stop, max_tokens, inference_method)
 
    return response


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout