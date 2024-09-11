import re
import logging

logger = logging.getLogger(__name__)

def initial_cot(system_prompt, initial_query, client, model: str):
    cot_prompt = f"""
    {system_prompt}

    You are an AI assistant that uses a Chain of Thought (CoT) approach to answer queries. Follow these steps:

    1. Think through the problem step by step within the <thinking> tags.
    2. Provide your initial answer within the <output> tags.

    Use the following format for your response:
    <thinking>
    [Your step-by-step reasoning goes here. This is your internal thought process.]
    </thinking>
    <output>
    [Your initial answer to the query.]
    </output>
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": initial_query}
        ],
        temperature=0.7,
        max_tokens=8192
    )

    full_response = response.choices[0].message.content
    logger.info(f"Initial CoT:\n{full_response}")

    thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
    output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
    initial_output = output_match.group(1).strip() if output_match else full_response

    return thinking, initial_output

def reflection_and_final_output(system_prompt, initial_query, thinking, initial_output, client, model: str):
    reflection_prompt = f"""
    You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to improve answers. 
    You will be presented with an initial thinking process and output for a given query. 
    Your task is to reflect on this, identify potential improvements or errors, and provide a final answer.

    Original query: {initial_query}

    Initial thinking:
    {thinking}

    Initial output:
    {initial_output}

    Please follow these steps:
    1. Reflect on the initial thinking and output, considering potential errors, biases, or missing information.
    2. Based on your reflection, provide an improved thinking process.
    3. Provide a final, potentially revised answer.

    Use the following format for your response:
    <reflection>
    [Your reflection on the initial thinking and output, identifying areas for improvement.]
    </reflection>
    <thinking>
    [Your improved step-by-step reasoning based on the reflection.]
    </thinking>
    <output>
    [Your final, potentially revised answer to the query.]
    </output>
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reflection_prompt}
        ],
        temperature=0.7,
        max_tokens=8192
    )

    full_response = response.choices[0].message.content
    logger.info(f"Reflection and Final Output:\n{full_response}")

    output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)
    final_output = output_match.group(1).strip() if output_match else full_response

    return final_output

def cot_reflection(system_prompt, initial_query, client, model: str, return_full_response: bool = False):
    # Step 1: Get initial thinking and output
    initial_thinking, initial_output = initial_cot(system_prompt, initial_query, client, model)

    # Step 2: Reflect and get final output
    final_output = reflection_and_final_output(system_prompt, initial_query, initial_thinking, initial_output, client, model)

    if return_full_response:
        full_response = f"""
        Initial Thinking:
        {initial_thinking}

        Initial Output:
        {initial_output}

        Final Output:
        {final_output}
        """
        return full_response
    else:
        return final_output
