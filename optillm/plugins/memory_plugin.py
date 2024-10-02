import re
from typing import Tuple, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SLUG = "memory"

class Memory:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.items: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.completion_tokens = 0

    def add(self, item: str):
        if len(self.items) >= self.max_size:
            self.items.pop(0)
        self.items.append(item)
        self.vectors = None  # Reset vectors to force recalculation

    def get_relevant(self, query: str, n: int = 5) -> List[str]:
        if not self.items:
            return []

        if self.vectors is None:
            self.vectors = self.vectorizer.fit_transform(self.items)

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = similarities.argsort()[-n:][::-1]
        
        return [self.items[i] for i in top_indices]

def extract_query(text: str) -> Tuple[str, str]:
    query_index = text.rfind("Query:")
    
    if query_index != -1:
        context = text[:query_index].strip()
        query = text[query_index + 6:].strip()
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) > 1:
            context = ' '.join(sentences[:-1])
            query = sentences[-1]
        else:
            context = text
            query = "What is the main point of this text?"
    return query, context

def extract_key_information(text: str, client, model: str) -> List[str]:
    # print(f"Prompt : {text}")
    prompt = f"""Extract key information from the following text. Provide a list of important facts or concepts, each on a new line:

{text}

Key information:"""

    try: 
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        key_info = response.choices[0].message.content.strip().split('\n')
    except Exception as e:
        print(f"Error parsing content: {str(e)}")
        return [],0
    
    return [info.strip('- ') for info in key_info if info.strip()], response.usage.completion_tokens

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    memory = Memory()
    query, context = extract_query(initial_query)
    completion_tokens = 0

    # Process context and add to memory
    chunk_size = 10000
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i+chunk_size]
        # print(f"chunk: {chunk}")
        key_info, tokens = extract_key_information(chunk, client, model)
        #print(f"key info: {key_info}")
        completion_tokens += tokens
        for info in key_info:
            memory.add(info)
    # print(f"query : {query}")
    # Retrieve relevant information from memory
    relevant_info = memory.get_relevant(query)
    # print(f"relevant_info : {relevant_info}")
    # Generate response using relevant information
    prompt = f"""System: {system_prompt}

Context: {' '.join(relevant_info)}

{query}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    print(f"response : {response}")
    final_response = response.choices[0].message.content.strip()
    completion_tokens += response.usage.completion_tokens
    print(f"final_response: {final_response}")
    return final_response, completion_tokens