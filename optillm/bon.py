import logging

logger = logging.getLogger(__name__)

def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int = 3) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}]
    
    completions = []
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        n=n,
        temperature=1
    )
    completions = [choice.message.content for choice in response.choices]
    
    # Rate the completions
    rating_messages = messages.copy()
    rating_messages.append({"role": "system", "content": "Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number."})
    
    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "system", "content": "Rate the above response:"})
        
        rating_response = client.chat.completions.create(
            model=model,
            messages=rating_messages,
            max_tokens=256,
            n=1,
            temperature=0.1
        )
        
        try:
            rating = float(rating_response.choices[0].message.content.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
        
        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    return completions[best_index]
