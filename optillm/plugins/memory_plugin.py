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

    def get_relevant(self, query: str, n: int = 10) -> List[str]:
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

def classify_margin(margin):
        return margin.startswith("YES#")

def extract_key_information(system_message, text: str, query: str, client, model: str) -> List[str]:
    # print(f"Prompt : {text}")
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""
'''text
{text}
'''
Copy over all context relevant to the query: {query}
Provide the answer in the format: <YES/NO>#<Relevant context>.
Here are rules:
- If you don't know how to answer the query - start your answer with NO#
- If the text is not related to the query - start your answer with NO#
- If you can extract relevant information - start your answer with YES#
- If the text does not mention the person by name - start your answer with NO#
Example answers:
- YES#Western philosophy originated in Ancient Greece in the 6th century BCE with the pre-Socratics.
- NO#No relevant context.
"""}
    ]

    try: 
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        key_info = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error parsing content: {str(e)}")
        return [],0
    margins = []

    if classify_margin(key_info):
        margins.append(key_info.split("#", 1)[1])
    
    return margins, response.usage.completion_tokens

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    memory = Memory()
    query, context = extract_query(initial_query)
    completion_tokens = 0

    # Process context and add to memory
    chunk_size = 100000
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i+chunk_size]
        # print(f"chunk: {chunk}")
        key_info, tokens = extract_key_information(system_prompt, chunk, query, client, model)
        #print(f"key info: {key_info}")
        completion_tokens += tokens
        for info in key_info:
            memory.add(info)
    # print(f"query : {query}")
    # Retrieve relevant information from memory
    relevant_info = memory.get_relevant(query)
    # print(f"relevant_info : {relevant_info}")
    # Generate response using relevant information
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""

I asked my assistant to read and analyse the above content page by page to help you complete this task. These are margin notes left on each page:
'''text
{relevant_info}
'''
Read again the note(s), take a deep breath and answer the query.
{query}
"""}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # print(f"response : {response}")
    final_response = response.choices[0].message.content.strip()
    completion_tokens += response.usage.completion_tokens
    # print(f"final_response: {final_response}")
    return final_response, completion_tokens