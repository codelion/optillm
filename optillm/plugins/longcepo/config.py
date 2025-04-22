from dataclasses import dataclass

from optillm.plugins.longcepo.prompts import (
    MAPREDUCE_SYSTEM_PROMPT,
    QUERY_FORMAT_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    MAP_PROMPT,
    REDUCE_PROMPT,
    COLLAPSE_PROMPT,
    SUMMARY_PROMPT,
)


@dataclass
class LongCepoConfig:
    temperature_plan: float = 0.7  # Temperature for planning stage
    temperature_map: float = 0.7  # Temperature for map stage
    temperature_collapse: float = 0.7  # Temperature for collapse stage
    temperature_reduce: float = 0.7  # Temperature for reduce stage

    chunk_size: int = 4096  # Max tokens per chunk when splitting context
    max_output_tokens: int = 1024  # Max output tokens per LLM API call (except for summary generation)
    max_context_window: int = 8192  # Total model context window available
    max_output_tokens_summary: int = 300  # Max output tokens per LLM API call (summary generation)
    num_neighbor_summaries: int = 5  # Number of adjacent summaries from before/after in the context included in mapping stage

    system_prompt: str = MAPREDUCE_SYSTEM_PROMPT  # System prompt used in map/collapse/reduce stages
    summary_prompt: str = SUMMARY_PROMPT  # Prompt template for generating summaries in map phase
    map_prompt: str = MAP_PROMPT  # Prompt template for map stage
    collapse_prompt: str = COLLAPSE_PROMPT  # Prompt template for collapse stage
    reduce_prompt: str = REDUCE_PROMPT  # Prompt template for reduce stage
    query_format_prompt: str = QUERY_FORMAT_PROMPT  # Query normalization step prompt
    planning_system_prompt: str = PLANNING_SYSTEM_PROMPT  # Planning stage prompt

    context_query_delimiter: str = "<CONTEXT_END>"  # Delimiter used to split initial input into context and query
    tokenizer_name: str = "meta-llama/Llama-3.3-70B-Instruct"  # Tokenizer to use to determine token lengths
