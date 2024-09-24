import logging
import re
from typing import List, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LEAP:
    def __init__(self, system_prompt: str, client, model: str):
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.low_level_principles = []
        self.high_level_principles = []
        self.leap_completion_tokens = 0

    def extract_output(self, text: str) -> str:
        match = re.search(r'<output>(.*?)(?:</output>|$)', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_examples_from_query(self, initial_query: str) -> List[Tuple[str, str]]:
        logger.info("Extracting examples from initial query")
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                Analyze the following query and determine if it contains few-shot examples.
                If it does, extract the examples and their corresponding answers.
                Format the examples as a JSON array of objects, where each object has "question" and "answer" fields.
                If there are no examples, return an empty array.
                Enclose your response within <output></output> tags.
                Do not put any explanation or any other reponse other than the JSON array within the <output></output> tags.

                Example output format:
                <output>
                [
                    {{"question": "What is 2+2?", "answer": "4"}},
                    {{"question": "What is the capital of France?", "answer": "Paris"}}
                ]
                </output>

                Query: {initial_query}
                """}
            ]
        )
        self.leap_completion_tokens += response.usage.completion_tokens
        examples_str = self.extract_output(response.choices[0].message.content)
        logger.debug(f"Extracted examples: {examples_str}")
        examples = []
        if examples_str:
            try:
                examples_list = json.loads(examples_str)
                examples = [(example['question'], example['answer']) for example in examples_list]
            except json.JSONDecodeError:
                logger.warning("Failed to parse examples JSON, using empty list")
            except KeyError:
                logger.warning("Parsed JSON does not have the expected structure, using empty list")
        
        logger.debug(f"Extracted examples: {examples}")
        return examples

    def generate_mistakes(self, examples: List[Tuple[str, str]]) -> List[Tuple[str, str, str, str]]:
        logger.info("Generating mistakes for given examples")
        mistakes = []
        for question, correct_answer in examples:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
                    Instruction: Answer the following question step by step. To induce a mistake, 
                    deliberately introduce an error in your reasoning or calculation.
                    Question: {question}
                    Provide your step-by-step reasoning, then enclose your final answer within <output></output> tags.
                    Think step by step, but make sure to include a mistake.
                    """}
                ],
                temperature=0.7,
            )
            self.leap_completion_tokens += response.usage.completion_tokens
            generated_reasoning = response.choices[0].message.content
            generated_answer = self.extract_output(generated_reasoning)
            if generated_answer != correct_answer:
                mistakes.append((question, generated_reasoning, generated_answer, correct_answer))
        return mistakes

    def generate_low_level_principles(self, mistakes: List[Tuple[str, str, str, str]]) -> List[str]:
        logger.info("Generating low-level principles from mistakes")
        for question, generated_reasoning, generated_answer, correct_answer in mistakes:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
                    Question: {question}
                    Generated Reasoning: {generated_reasoning}
                    Generated Answer: {generated_answer}
                    Correct Answer: {correct_answer}
                    Instruction: Conduct a thorough analysis of the generated answer in comparison to the
                    correct answer. Also observe how the generated reasoning differs from the correct
                    reasoning. Identify any discrepancies, misunderstandings, or errors. Provide clear
                    insights, principles, or guidelines that can be derived from this analysis to improve
                    future responses. We are not focused on this one data point, but rather on the general
                    principle.
                    Reasoning: <discuss why the generated answer is wrong>
                    Insights: Enclose ONLY the principles or insights within <output></output> tags.
                    """}
                ]
            )
            self.leap_completion_tokens += response.usage.completion_tokens
            self.low_level_principles.append(self.extract_output(response.choices[0].message.content))
        return self.low_level_principles

    def generate_high_level_principles(self) -> List[str]:
        logger.info("Generating high-level principles from low-level principles")
        principles_text = "\n".join(self.low_level_principles)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                Low-level principles: {principles_text}
                Create a list of *unique* and insightful principles to improve future responses based
                on the analysis above.
                Focus on capturing the essence of the feedback while eliminating redundancies.
                Ensure that each point is clear, concise, and directly derived from the introspection
                results.
                Create a numbered list of principles. Leave specific details in place.
                Limit to at most 8 principles.
                Enclose your list of principles within <output></output> tags.
                """}
            ]
        )
        self.leap_completion_tokens += response.usage.completion_tokens
        self.high_level_principles = self.extract_output(response.choices[0].message.content).split("\n")
        return self.high_level_principles

    def apply_principles(self, query: str) -> str:
        logger.info("Applying learned principles to query")
        principles_text = "\n".join(self.high_level_principles)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                Please answer the following query. Keep in mind these principles:

                {principles_text}

                Query: {query}
                """}
            ]
        )
        self.leap_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def solve(self, initial_query: str) -> str:
        logger.info("Starting LEAP process")
        examples = self.extract_examples_from_query(initial_query)
        if not examples:
            logger.warning("No examples found in the query. Proceeding with direct answer.")
            return self.apply_principles(initial_query)
        
        mistakes = self.generate_mistakes(examples)
        self.generate_low_level_principles(mistakes)
        self.generate_high_level_principles()
        
        return self.apply_principles(initial_query)

def leap(system_prompt: str, initial_query: str, client, model: str) -> str:
    leap_solver = LEAP(system_prompt, client, model)
    return leap_solver.solve(initial_query), leap_solver.leap_completion_tokens