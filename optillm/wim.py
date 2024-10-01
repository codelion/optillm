from collections import deque
import tiktoken
import re

class WiMInfiniteContextAPI:
    def __init__(self, system_prompt, client, model, max_context_tokens=64000, max_margins=10, chunk_size=16000):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_margins = max_margins
        self.chunk_size = chunk_size
        self.context_buffer = deque()
        self.margins = deque(maxlen=max_margins)
        try:
            self.tokenizer = tiktoken.encoding_for_model(model) 
        except:
            self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.system_message = system_prompt
        self.client = client
        self.win_completion_tokens = 0

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def trim_context_buffer(self):
        while self.count_tokens("".join(self.context_buffer)) > self.max_context_tokens:
            self.context_buffer.popleft()

    def generate_margin(self, chunk, query):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"""
'''text
{chunk}
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens = 512
        )
        self.win_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def classify_margin(self, margin):
        return margin.startswith("YES#")
    
    def extract_query(self, text):
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Check if the last sentence starts with "Query:"
        if sentences[-1].startswith("Query:"):
            return sentences[-1][6:].strip(), "".join(sentences[:-1])
        
        # If not, assume the last sentence is the query
        return sentences[-1].strip(), "".join(sentences[:-1])

    def process_chunk(self, chunk, query):
        self.context_buffer.append(chunk)
        self.trim_context_buffer()
        margin = self.generate_margin(chunk, query)
        if self.classify_margin(margin):
            self.margins.append(margin.split("#", 1)[1])

    def process_stream(self, text_stream, query):
        for chunk in text_stream:
            self.process_chunk(chunk, query)

    def generate_final_answer(self, query):
        context = "".join(self.context_buffer)
        margins = "\n".join(self.margins)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"""
'''text
{context}
'''
I asked my assistant to read and analyse the above content page by page to help you complete this task. These are margin notes left on each page:
'''text
{margins}
'''
Read again the note(s) and the provided content, take a deep breath and answer the query.
{self.instruction}
{query}
"""}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        self.win_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def run(self, text_stream, query):
        self.process_stream(text_stream, query)
        return self.generate_final_answer(query)

    @property
    def instruction(self):
        return "Answer the following question based on the provided context and margin notes:"

    # Usage
    def text_stream_generator(self, text):
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i+self.chunk_size]

    def process_query(self, initial_query):
        query, context = self.extract_query(initial_query)
        text_stream = self.text_stream_generator(context)
        final_answer = self.run(text_stream, query)
        return final_answer, self.win_completion_tokens