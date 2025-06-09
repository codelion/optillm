import re
from typing import Tuple
from functools import partial

# Use relative imports that work within the dynamically loaded module
from .mapreduce import mapreduce
from .utils import (
    get_prompt_response,
    logger,
    longcepo_init,
    loop_until_match,
)


def run_longcepo(
    system_prompt: str, initial_query: str, client, model: str
) -> Tuple[str, int]:
    """
    Executes the full LongCePO multi-stage pipeline to answer a complex query from long context.

    The pipeline includes:
      - Normalizing the format of the original query
      - Generating a plan of sub-questions
      - Iteratively answering each sub-question using a MapReduce-style question-answering engine
      - Aggregating QA history and producing a final answer with MapReduce

    Args:
        system_prompt (str): System prompt string.
        initial_query (str): Raw input string containing context and query separated by delimiter string.
        client: LLM API client instance.
        model (str): Base model name.

    Returns:
        Tuple[str, int]: Final answer and total number of tokens used across the pipeline.
    """
    context, query, tokenizer, cb_log, longcepo_config = longcepo_init(initial_query)

    # Normalize query
    normalized_query, upd_log = get_prompt_response(
        client,
        model,
        longcepo_config.query_format_prompt.format(full_query=query),
        system_prompt,
        max_tokens=longcepo_config.max_output_tokens,
    )
    cb_log.update(upd_log)
    logger.info(f"Normalized query: {normalized_query}")

    # Planning stage
    prompt = f"The question is: {normalized_query}"
    gen_fn = partial(
        get_prompt_response,
        client=client,
        model=model,
        prompt=prompt,
        system_prompt=longcepo_config.planning_system_prompt,
        max_tokens=longcepo_config.max_output_tokens,
    )
    planning_response, upd_log = loop_until_match(
        gen_fn, pattern_list=("<SUB-QUESTIONS>",)
    )
    logger.info(f"Planning stage output:\n\n{planning_response}")
    questions = (
        re.findall(
            r"<SUB-QUESTIONS>\s*(.*?)\s*</SUB-QUESTIONS>", planning_response, re.DOTALL
        )[0]
        .strip()
        .splitlines()
    )

    # Loop to answer sub-queries from the plan
    qa_system_prompt = (
        longcepo_config.system_prompt
        if longcepo_config.system_prompt is not None
        else system_prompt
    )
    qa_history = ""
    for question in questions:
        if not question:
            continue
        question = re.sub(r"^\d+\.", "", question)
        answer, cb_log = mapreduce(
            qa_system_prompt,
            question,
            context,
            qa_history,
            client,
            model,
            tokenizer,
            longcepo_config=longcepo_config,
            cb_log=cb_log,
        )
        qa_history += f"- Previous question: {question}\n\n"
        answer = re.sub(r"^:+", "", answer)
        qa_history += f"- Previous answer: {answer}\n\n"
        logger.info(f"QA history:\n\n{qa_history}")

    # Final answer generation
    answer, cb_log = mapreduce(
        qa_system_prompt,
        query,
        context,
        qa_history,
        client,
        model,
        tokenizer,
        longcepo_config=longcepo_config,
        cb_log=cb_log,
    )
    return answer, cb_log["total_tokens"]
