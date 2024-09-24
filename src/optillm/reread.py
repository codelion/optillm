import logging

logger = logging.getLogger(__name__)

def re2_approach(system_prompt, initial_query, client, model, n=1):
    """
    Implement the RE2 (Re-Reading) approach for improved reasoning in LLMs.
    
    Args:
    system_prompt (str): The system prompt to be used.
    initial_query (str): The initial user query.
    client: The OpenAI client object.
    model (str): The name of the model to use.
    n (int): Number of completions to generate.
    
    Returns:
    str or list: The generated response(s) from the model.
    """
    logger.info("Using RE2 approach for query processing")
    re2_completion_tokens = 0
    
    # Construct the RE2 prompt
    re2_prompt = f"{initial_query}\nRead the question again: {initial_query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": re2_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=n
        )
        re2_completion_tokens += response.usage.completion_tokens
        if n == 1:
            return response.choices[0].message.content.strip(), re2_completion_tokens
        else:
            return [choice.message.content.strip() for choice in response.choices], re2_completion_tokens
    
    except Exception as e:
        logger.error(f"Error in RE2 approach: {str(e)}")
        raise
