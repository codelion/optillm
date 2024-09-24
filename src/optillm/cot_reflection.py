import re
import logging

logger = logging.getLogger(__name__)

def cot_reflection(system_prompt, initial_query, client, model: str, return_full_response: bool=False):
    cot_completion_tokens = 0
    cot_prompt = f"""
        {system_prompt}

        You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:

        1. Think through the problem step by step within the <thinking> tags.
        2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
        3. Make any necessary adjustments based on your reflection.
        4. Provide your final, concise answer within the <output> tags.

        Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
        Do not include any part of the final answer in these sections. 
        The actual response to the query must be entirely contained within the <output> tags.

        Use the following format for your response:
        <thinking>
        [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
        <reflection>
        [Your reflection on your reasoning, checking for errors or improvements]
        </reflection>
        [Any adjustments to your thinking based on your reflection]
        </thinking>
        <output>
        [Your final, concise answer to the query. This is the only part that will be shown to the user.]
        </output>
        """

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": initial_query}
        ],
        temperature=0.7,
        max_tokens=4096
    )

    # Extract the full response
    full_response = response.choices[0].message.content
    cot_completion_tokens += response.usage.completion_tokens
    logger.info(f"CoT with Reflection :\n{full_response}")

    # Use regex to extract the content within <thinking> and <output> tags
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
    output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
    output = output_match.group(1).strip() if output_match else full_response

    logger.info(f"Final output :\n{output}")

    if return_full_response:
        return full_response, cot_completion_tokens
    else:
        return output, cot_completion_tokens

