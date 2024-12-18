import re
from dataclasses import dataclass
import cerebras
import openai


class CepoConfig:
    verifier_n: int = 3
    planning_n: int = 3
    planning_m: int = 6
    planning_temperature: list[int] = [0.55, 0.25, 0.1, 0]  #[0.55, custom_temp, 0.1, 0]
    custom_max_tokens: int = 4096
    


def extract_question_only(task: str) -> str:
    """Remove the instruction part of the task and returns the question only."""
    question_only = task.replace('\n## Question: \n\n', '')
    question_only = question_only.replace('\n\n\n## Instruction \n\nPlease answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following json format. \nPlease show your final answer in the `answer` field, e.g.,`"answer": "42"`.\n\n```json\n{\n    "reasoning": "___",\n    "answer": "___"\n}\n```\n', '')
    return question_only


def generate_completion(system_prompt: str, task: str, client, model: str, n: int = 3, m: int = 6, temperature: list|None = None, custom_max_tokens:int = 4096) -> str:
    completion_tokens = 0
    if temperature is None:
        temperature = [0.55, 0.25, 0.0, 0.0]

    question_only = extract_question_only(task)

    cb_log = {}
    plans = []
    for i in range(m):  # m is the maximum number of attempts to generate n plans
        # Step 1
        content = f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
                  f"final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability "\
                  f"to execute it correctly. Here is the question:\n{question_only}\nRead the question again:\n\n{question_only}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=temperature[0],
            stream=False,
        )
        completion_tokens += response.usage.completion_tokens

        if response.choices[0].finish_reason == "length":
            #logger.debug(f"Skipping plan generation {i} due to length")
            continue

        # Step 2
        content = f"Can you execute the above plan step-by-step to produce the final answer. "\
                f"Be extra careful when executing steps where your confidence is lower."
        messages.extend([{"role": "assistant", "content": response.choices[0].message.content}, {"role": "user", "content": content}])
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=temperature[1],
            stream=False,
        )
        completion_tokens += response.usage.completion_tokens

        if response.choices[0].finish_reason == "length":
            # logger.debug(f"Skipping plan generation {i} due to length")
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            cb_log[f"messages_planning_{i}_rejected_due_to_length"] = messages
            continue

        plans.append(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        cb_log[f"messages_planning_{i}"] = messages
        
        if len(plans) == n:
            break
    if not plans:
        plans.append(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        cb_log[f"messages_planning_{i}_no_plans_so_taking_the_last_one"] = messages

    # Step 3
    try:
        plans_message = ""
        for i, plan in enumerate(plans):
            plans_message += f"Response {i + 1}:\n{plan}\n\n"
        plans_message = plans_message[:-2]  # remove the last 2x newline
        content = f"Can you review your last {len(plans)} responses and identify any inconsistency between them. After that, can you address "\
                  f"it and present a final step-by-step solution to the problem? Here is the question:\n{question_only}"
        messages = [{"role": "assistant", "content": plans_message}, {"role": "user", "content": content}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=temperature[2],
            stream=False,
        )
        final_solution = response.choices[0].message.content
        completion_tokens += response.usage.completion_tokens
    except (cerebras.cloud.sdk.BadRequestError, openai.BadRequestError) as e:
        # logger.debug("The following Cerebras API error occured:")
        # logger.debug(e)
        # logger.debug("Using only the first plan moving forward")
        final_solution = plans[0]
        messages = []

    # Step 4
    content = f"Use your final solution from above to correctly answer the question. Here is the question:\n{task}"
    messages = [{"role": "assistant", "content": final_solution}, {"role": "user", "content": content}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=custom_max_tokens,
        temperature=temperature[3],
        stream=False,
    )
    completion_tokens += response.usage.completion_tokens

    cb_log["messages"] = messages
    return response.choices[0].message.content, completion_tokens, cb_log


def generate_n_completions(
    system_prompt: str,
    initial_query: str,
    client, model: str,
    n: int,
    custom_temp:int = 0.25,
    custom_max_tokens:int = 4096
) -> tuple[list[str], int, dict]:
    
    completion_tokens = 0
    cb_log = {}
    completions = []
    for i in range(n):
        response_i, completion_tokens_i, cb_log_i = generate_completion(system_prompt, initial_query, client, model, n=3, m=6, temperature=[0.55, custom_temp, 0.1, 0], subvariant='cepo_v1_8_1_1', custom_max_tokens=custom_max_tokens)

        completions.append(response_i)
        completion_tokens += completion_tokens_i
        cb_log[f"completion_{i}_response"] = response_i
        cb_log[f"completion_{i}_log"] = cb_log_i
        cb_log[f"completion_{i}_completion_tokens"] = completion_tokens_i    
    return completions, completion_tokens, cb_log



def cepo(system_prompt: str, initial_query: str, client, model: str, cepo_config: CepoConfig | None = None) -> list[str, int, dict]: #, n: int = 3, custom_temp:int = 0.25, custom_max_tokens:int = 4096) -> str:
    custom_temp = 0.25,
    if cepo_config is None:
        cepo_config = CepoConfig()
    
    # cepo_config = {
    #     "verifier_n": 3,
    #     "planning_n": 3,
    #     "planning_m": 6,
    #     "planning_temperature": [0.55, custom_temp, 0.1, 0],
    #     "custom_max_tokens": 4096
    # }

    # Generate completions
    completions, completion_tokens, cb_log = generate_n_completions(system_prompt, initial_query, client, model, cepo_config)
    
    # Rate the completions
    rating_messages = [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": initial_query}]
    
    prompt_1 = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to "\
        "the user question displayed below. Your evaluation should consider correctness as a primary factor as "\
        "well as other factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of "\
        "detail of the response. Evaluation Criteria:\n"\
        "- Correctness: How free is it from errors or mistakes?\n"\
        "- Helpfulness: How effectively does the response meet the user's needs?\n"\
        "- Relevance: How directly does the response address the original question?\n"\
        "- Accuracy: Are the information and explanations factually correct?\n"\
        "- Depth: Does the response provide comprehensive and meaningful insights?\n"\
        "- Creativity: Does the response offer unique or innovative perspectives?\n"\
        "- Clarity: Is the response well-organized, coherent, and easy to understand?\n"\
        "Evaluation Process:\n"\
        "1. Carefully review the user question and the AI assistant's response.\n"\
        "2. Assess the response against each criterion.\n"\
        "3. Provide a concise explanation of your overall evaluation.\n"\
        "4. Rate the response on a 1-10 scale with the following guidelines:\n"\
        "    - 1-2: Completely inadequate, fails to address the question\n"\
        "    - 3-4: Minimal relevance, significant deficiencies\n"\
        "    - 5-6: Partially helpful, requires substantial improvement\n"\
        "    - 7-8: Good response with minor areas for enhancement\n"\
        "    - 9-10: Correct, comprehensive, and highly insightful.\n"\
        "Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your "\
        "explanation, please rate the response on a scale of 1 to 10 by strictly following this format:  \"Rating: "\
        "[[rating]]\", for example: \"Rating: [[5]]\""
    prompt_2 = "Rate the above response beginning with a small evaluation blurb followed by a rating on a scale of 1 to 10 "\
               "by strictly following this format: \"Explanation: <reason for your rating>\n\nRating: [[rating]]\"."
    
    max_tokens = 4096
    rating_messages.append({"role": "system", "content": prompt_1})
    
    
    ratings = []
    for i, completion in enumerate(completions):
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "system", "content": prompt_2})

        rating_response = client.chat.completions.create(
            model=model,
            messages=rating_messages,
            max_tokens=max_tokens,
            n=1,
            temperature=0.1
        )
        completion_tokens += rating_response.usage.completion_tokens
        
        rating_response = rating_response.choices[0].message.content.strip()
        cb_log[f"rating_response_{i}"] = rating_response

        pattern = r"Rating: \[\[(\d+)\]\]"
        match = re.search(pattern, rating_response)
        if match:
            rating_response = match.group(1)
        else:
            rating_response = "0"

        try:
            rating = float(rating_response)
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
        
        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    cb_log["ratings"] = ratings
    cb_log["best_index"] = best_index
    return completions[best_index], completion_tokens, cb_log






"""

import itertools
from .utils import extract_question_only, wait_for_rate_limit, logger






def cepo_v2_pairwise(system_prompt: str, initial_query: str, client, model: str, n: int = 3, prompt_variant: int = 0, generator: str = 'cepo_v1', custom_temp:int = 0.25, custom_max_tokens:int = 4096) -> str:
    logger.debug(f"Running {model}")
    assert prompt_variant == 0, "Invalid prompt variant in cepo_v2_pairwise"
    logger.debug(f"Using prompt variant: {prompt_variant}")
    
    # Generate completions
    completions, completion_tokens, cb_log = generate_completions(system_prompt, initial_query, client, model, n, generator, custom_temp, custom_max_tokens)
    
    # Rate the completions
    rating_messages = [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": initial_query}]
    rating_prompt_1 = "Please act as an impartial judge and compare the quality of the two responses provided by the AI assistant " \
                "to the user's question displayed below. Evaluation Criteria:\n" \
                "- Helpfulness: How effectively does the response meet the user's needs?\n" \
                "- Relevance: How directly does the response address the original question?\n" \
                "- Accuracy: Are the information and explanations factually correct?\n" \
                "- Depth: Does the response provide comprehensive and meaningful insights?\n" \
                "- Creativity: Does the response offer unique or innovative perspectives?\n" \
                "- Clarity: Is the response well-organized, coherent, and easy to understand?\n" \
                "Evaluation Process:\n" \
                "1. Carefully review the user's question and the AI assistant's responses.\n" \
                "2. Compare the responses against each other for each criterion.\n" \
                "3. Provide a concise explanation of your overall evaluation.\n" \
                "4. Select the response that is superior based on the above criteria.\n" \
                "Reply with \"Better Response: [[response id]]\".\n" \
                "If the first response is better, reply with \"Better Response: [[0]]\". " \
                "If the second response is better, reply with \"Better Response: [[1]]\"."
    rating_messages.append({"role": "system", "content": rating_prompt_1})

    ratings = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for pair in pairs:
        responses_pair = f"Response 0: {completions[pair[0]]}\n\nResponse 1: {completions[pair[1]]}"
        rating_messages.append({"role": "assistant", "content": responses_pair})
        rating_prompt_2 =  "Reply with \"Better Response: [[response id]]\".\n" \
                           "If the first response is better, reply with \"Better Response: [[0]]\". " \
                           "If the second response is better, reply with \"Better Response: [[1]]\"."
        max_tokens = 4096
        rating_messages.append({"role": "system", "content": rating_prompt_2})

        rating_response = client.chat.completions.create(
            model=model,
            messages=rating_messages,
            max_tokens=max_tokens,
            n=1,
            temperature=0.1
        )
        completion_tokens += rating_response.usage.completion_tokens
        
        rating_response = rating_response.choices[0].message.content.strip()
        cb_log[f"rating_response_for_pair_{pair[0]}_{pair[1]}"] = rating_response

        pattern = r"Better Response: \[\[(\d+)\]\]"
        match = re.search(pattern, rating_response)
        if match:
            rating_response = match.group(1)
            try:
                rating = int(rating_response)
                ratings[pair[rating]] += 1
            except ValueError:
                ratings[pair[0]] += 1  # if parsing unsuccessful, default to the first response
        else:
            ratings[pair[0]] += 1  # if parsing unsuccessful, default to the first response

        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    cb_log["ratings"] = ratings
    cb_log["best_index"] = best_index
    return completions[best_index], completion_tokens, cb_log
"""