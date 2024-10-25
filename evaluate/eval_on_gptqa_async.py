import pickle
import numpy as np
import csv
import logging
import os
import re
import time
import json
import asyncio
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List

import fire
from tqdm import tqdm
from openai import AsyncOpenAI
from utils import create_prompts, get_api_type, load_examples, CHAT_MODELS

APPROACHES = ["mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]

class AnswerPredictor:

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_filename: str, model_name: str, prompt_type: str = 'zero_shot', 
                 call_type: str = 'sample', max_examples: int = None,
                 verbose: bool = False, seed: int = 0, num_gpus: int = 1, use_greedy_sampling: bool = False):
        self.model_name = model_name
        self.data_filename = data_filename
        self.prompt_type = prompt_type
        self.call_type = call_type
        self.max_examples = max_examples
        self.verbose = verbose
        self.seed = seed
        self.num_gpus = num_gpus
        self.use_greedy_sampling = use_greedy_sampling
        self.client = AsyncOpenAI(api_key="BLEH", base_url="https://inference-time.research.arcee.ai/v1")
    
        if self.prompt_type == 'few_shot':
            raise ValueError('Few-shot deprecated - use `5_shot` instead')
    
    async def get_response_from_model(self, prompt: str, inference_method: str = "mcts") -> Dict[str, Any]:
        content = f"<optillm_approach>{inference_method}</optillm_approach> {prompt}"
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.0
        )
        return response
           
    @staticmethod
    def parse_sampled_answer(answer: str) -> str:
        patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    
        for pattern in patterns:
            match = re.search(pattern, answer)
            if match and match.group(1) in AnswerPredictor.LETTER_TO_INDEX:
                return match.group(1)
        return None
    
    async def sample_answer(self, prompt: str, inference_method: str = "mcts"):
        response = await self.get_response_from_model(prompt, inference_method=inference_method)
        answer = response.choices[0].message.content
        return self.parse_sampled_answer(answer), answer

    async def process_example(self, question_id: int, prompt: str, example: Any, inference_method: str, csvwriter):
        if self.verbose:
            print(f"Question: {example.question}")

        start_time = time.time()
        sampled_answer, model_response = await self.sample_answer(prompt, inference_method=inference_method)
        response_time = time.time() - start_time

        if sampled_answer is None:
            print(f"Couldn't find an answer choice for prompt: {prompt}")
            logging.info("Couldn't find an answer choice!")
            csvwriter.writerow([question_id, example.question, example[example.correct_index + 1], 
                             "Couldn't find an answer choice!", False, model_response, response_time])
            return 0, 1, response_time
            
        ans_correct_str = f"Correct answer: {example[example.correct_index + 1]}\nChosen answer: {example[self.LETTER_TO_INDEX[sampled_answer] + 1]}"
        logging.info(ans_correct_str)
        
        if self.verbose:
            print(ans_correct_str)
            
        is_correct = self.LETTER_TO_INDEX[sampled_answer] == example.correct_index
        
        csvwriter.writerow([question_id, example.question, example[example.correct_index + 1], 
                         example[self.LETTER_TO_INDEX[sampled_answer] + 1], is_correct, model_response, response_time])
                         
        return int(is_correct), 0, response_time

    async def main(self):
        method_results = {}
        
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        examples = load_examples(self.data_filename, seed=self.seed)
        if self.max_examples:
            examples = examples[:self.max_examples]
        prompts, examples = create_prompts(examples)

        # Create summary CSV
        summary_filename = f"logs/summary_{self.model_name}.csv"
        
        for inference_method in APPROACHES:
            csv_filename = f"logs/{self.prompt_type}_{self.model_name}_{inference_method}.csv"
            log_filename = f"logs/{self.prompt_type}_{self.model_name}_{inference_method}.log"
            
            logging.basicConfig(filename=log_filename, level=logging.INFO)
            
            correct = 0
            refusals = 0
            total_time = 0
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Question id', 'Question', 'Correct answer', 'Model answer', 'Correct', 'Model response', 'Response time (s)'])
                
                for question_id, (prompt, example) in tqdm(enumerate(zip(prompts, examples)), total=len(examples)):
                    is_correct, refusal, response_time = await self.process_example(
                        question_id, prompt, example, inference_method, csvwriter
                    )
                    correct += is_correct
                    refusals += refusal
                    total_time += response_time

            accuracy = correct / len(examples)
            refusal_rate = refusals / len(examples)
            avg_time = total_time / len(examples)
            
            method_results[inference_method] = {
                'accuracy': accuracy,
                'refusal_rate': refusal_rate,
                'avg_time': avg_time
            }
            print(f"Method: {inference_method}")
            print(f"Accuracy: {accuracy}")
            print(f"Refusal fraction: {refusal_rate:.6f}")
            print(f"Average response time: {avg_time:.6f}s")
            logging.info(f"Method: {inference_method}")
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Refusal fraction: {refusal_rate:.6f}")
            logging.info(f"Average response time: {avg_time:.6f}s")

        # Write summary CSV
        with open(summary_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Method', 'Accuracy', 'Refusal Rate', 'Average Response Time (s)'])
            for method, results in method_results.items():
                csvwriter.writerow([method, results['accuracy'], results['refusal_rate'], results['avg_time']])

if __name__ == '__main__':
    predictor = fire.Fire(AnswerPredictor)
    asyncio.run(predictor.main())

#python eval_on_gptqa.py main --data_filename datasets/gpqa_diamond.csv --prompt_type zero_shot