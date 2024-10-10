import os
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "http://localhost:8000/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

# calling OpenAI API
# client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

# response = client.chat.completions.create(
#   model="moa-gpt-4o-mini",
#   messages=[
#     {
#       "role": "user",
#       "content": "How many Rs are there in the word strawberry?"
#     }
#   ],
#   temperature=0.2
# )

# calling Openrouter's API
# client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# response = client.chat.completions.create(
#   model="moa-nousresearch/hermes-3-llama-3.1-405b:free",
#   messages=[
#     {
#       "role": "user",
#       "content": "How many Rs are there in the word strawberry?"
#     }
#   ],
#   temperature=0.2
# )

# calling Ollama API on Lightning
client = OpenAI(api_key="any", base_url=OLLAMA_BASE_URL)

response = client.chat.completions.create(
  model="moa-ollama/llama3.2",
  messages=[
    {
      "role": "user",
      "content": "How many Rs are there in the word strawberry?"
    }
  ],
  temperature=0.2
)

print(response.choices[0].message.content)
# print(response)