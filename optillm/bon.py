import logging
import optillm
from optillm import conversation_logger

logger = logging.getLogger(__name__)

def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int = 3, request_id: str = None) -> str:
    bon_completion_tokens = 0

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}]
    
    completions = []
    
    try:
        # Try to generate n completions in a single API call using n parameter
        provider_request = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "n": n,
            "temperature": 1
        }
        response = client.chat.completions.create(**provider_request)
        
        # Log provider call
        if request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        
        completions = [choice.message.content for choice in response.choices]
        logger.info(f"Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}")
        bon_completion_tokens += response.usage.completion_tokens
        
    except Exception as e:
        logger.warning(f"n parameter not supported by provider: {str(e)}")
        logger.info(f"Falling back to generating {n} completions one by one")
        
        # Fallback: Generate completions one by one in a loop
        for i in range(n):
            try:
                provider_request = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 1
                }
                response = client.chat.completions.create(**provider_request)
                
                # Log provider call
                if request_id:
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                    conversation_logger.log_provider_call(request_id, provider_request, response_dict)
                
                completions.append(response.choices[0].message.content)
                bon_completion_tokens += response.usage.completion_tokens
                logger.debug(f"Generated completion {i+1}/{n}")
                
            except Exception as fallback_error:
                logger.error(f"Error generating completion {i+1}: {str(fallback_error)}")
                continue
        
        if not completions:
            logger.error("Failed to generate any completions")
            return "Error: Could not generate any completions", 0
        
        logger.info(f"Generated {len(completions)} completions using fallback method. Total tokens used: {bon_completion_tokens}")
    
    # Rate the completions
    rating_messages = messages.copy()
    rating_messages.append({"role": "system", "content": "Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number."})
    
    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": "Rate the above response:"})
        
        provider_request = {
            "model": model,
            "messages": rating_messages,
            "max_tokens": 256,
            "n": 1,
            "temperature": 0.1
        }
        rating_response = client.chat.completions.create(**provider_request)
        
        # Log provider call
        if request_id:
            response_dict = rating_response.model_dump() if hasattr(rating_response, 'model_dump') else rating_response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        
        bon_completion_tokens += rating_response.usage.completion_tokens
        try:
            rating = float(rating_response.choices[0].message.content.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
        
        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    return completions[best_index], bon_completion_tokens
