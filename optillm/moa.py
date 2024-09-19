import logging

logger = logging.getLogger(__name__)

def mixture_of_agents(system_prompt: str, initial_query: str, client, model: str) -> str:
    completions = []

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_query}
        ],
        max_tokens=4096,
        n=3,
        temperature=1
    )
    completions = [choice.message.content for choice in response.choices]
    
    critique_prompt = f"""
    Original query: {initial_query}

    I will present you with three candidate responses to the original query. Please analyze and critique each response, discussing their strengths and weaknesses. Provide your analysis for each candidate separately.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Please provide your critique for each candidate:
    """

    critique_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt}
        ],
        max_tokens=512,
        n=1,
        temperature=0.1
    )
    critiques = critique_response.choices[0].message.content
    
    final_prompt = f"""
    Original query: {initial_query}

    Based on the following candidate responses and their critiques, generate a final response to the original query.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Critiques of all candidates:
    {critiques}

    Please provide a final, optimized response to the original query:
    """

    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=8192,
        n=1,
        temperature=0.1
    )
    
    return final_response.choices[0].message.content
