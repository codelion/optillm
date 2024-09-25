from collections import deque
import tiktoken

class WiMInfiniteContextAPI:
    def __init__(self, system_prompt, client, model, max_context_tokens=8192, max_margins=50, chunk_size=2000):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_margins = max_margins
        self.chunk_size = chunk_size
        self.context_buffer = deque()
        self.margins = deque(maxlen=max_margins)
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.system_message = system_prompt
        self.client = client

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
        response = self.client.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message['content']

    def classify_margin(self, margin):
        return margin.startswith("YES#")

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
        response = self.client.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message['content']

    def run(self, text_stream, query):
        self.process_stream(text_stream, query)
        return self.generate_final_answer(query)

    @property
    def instruction(self):
        return "Answer the following question based on the provided context and margin notes:"

# Usage
def text_stream_generator(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

api_key = "your-api-key-here"
wim = WiMInfiniteContextAPI(api_key)

text = "Very long text..."  # Your infinite context here
query = "What is the main topic?"

text_stream = text_stream_generator(text, wim.chunk_size)
final_answer = wim.run(text_stream, query)
print(final_answer)
