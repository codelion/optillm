"""The Long-Context Cerebras Planning and Optimization (LongCePO) Method

LongCePO is an inference-time computation method designed to provide LLMs with the capability to work with infinite context such as external knowledge bases that can run into millions of tokens. We achieve this goal through a combination of multiple strategies including planning (query decomposition) and divide-and-conquer long-context processing. This approach enables to use a limited context window (e.g. 8K) and outperform full-context processing with the same base model in many question-answering tasks.

If you have any questions or want to contribute, please reach out to us on [cerebras.ai/discord](https://cerebras.ai/discord).
"""

from typing import Tuple
from optillm.plugins.longcepo.main import run_longcepo


SLUG = "longcepo"

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    return run_longcepo(system_prompt, initial_query, client, model)
