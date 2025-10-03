"""
Answer extraction utilities for MARS
Extracts clean final answers from MARS synthesis output
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_clean_answer(text: str, mode: str = 'auto') -> str:
    """
    Extract clean final answer from MARS synthesis text

    Args:
        text: Full synthesis output with reasoning
        mode: 'auto', 'code', 'math', or 'none'

    Returns:
        Clean final answer without intermediate reasoning
    """
    if mode == 'none':
        return text

    # Auto-detect mode if not specified
    if mode == 'auto':
        mode = detect_answer_type(text)

    if mode == 'code':
        return extract_code_answer(text)
    elif mode == 'math':
        return extract_math_answer(text)
    else:
        return extract_generic_answer(text)


def detect_answer_type(text: str) -> str:
    """Detect whether this is a code, math, or generic problem"""
    # Check for code indicators
    code_indicators = ['```', 'def ', 'import ', 'class ', 'return ', 'for ', 'while ']
    has_code = any(indicator in text for indicator in code_indicators)

    # Check for math indicators
    math_indicators = ['\\boxed', '\\frac', '\\sum', '\\int', '$$', '$\\']
    has_math = any(indicator in text for indicator in math_indicators)

    if has_code:
        return 'code'
    elif has_math:
        return 'math'
    else:
        return 'generic'


def extract_code_answer(text: str) -> str:
    """
    Extract clean code from synthesis output
    Finds the last complete code block as the final answer
    """
    # Try to find code blocks with language specifier
    code_blocks = re.findall(r'```(?:python|cpp|java|javascript|go|rust)?\n(.*?)\n```', text, re.DOTALL)

    if code_blocks:
        # Return last code block (most likely the final solution)
        final_code = code_blocks[-1].strip()
        logger.info(f"📝 EXTRACTION: Found {len(code_blocks)} code blocks, using last one ({len(final_code)} chars)")
        return f"```python\n{final_code}\n```"

    # Fallback: Look for code after common section headers
    sections = re.split(r'\n#+\s+(?:Final Solution|Solution|Implementation|Code)\s*\n', text, flags=re.IGNORECASE)
    if len(sections) > 1:
        final_section = sections[-1].strip()
        logger.info(f"📝 EXTRACTION: Using code from final section ({len(final_section)} chars)")
        return final_section

    # Last resort: Return text after last heading
    parts = text.split('###')
    if len(parts) > 1:
        final_part = parts[-1].strip()
        logger.info(f"📝 EXTRACTION: Using text after last heading ({len(final_part)} chars)")
        return final_part

    logger.warning("⚠️  EXTRACTION: No clear code found, returning full text")
    return text


def extract_math_answer(text: str) -> str:
    """
    Extract clean math answer from synthesis output
    Finds the last \\boxed{} answer as the final answer
    """
    # Find all boxed answers
    boxed_answers = re.findall(r'\\boxed\{([^}]+)\}', text)

    if boxed_answers:
        # Return last boxed answer (most likely the final one)
        final_answer = boxed_answers[-1]
        logger.info(f"📝 EXTRACTION: Found {len(boxed_answers)} boxed answers, using last one: {final_answer}")
        return f"The final answer is $\\boxed{{{final_answer}}}$"

    # Fallback: Look for "final answer" or similar phrases
    final_patterns = [
        r'[Ff]inal answer[:\s]+(.+?)(?:\n|$)',
        r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)',
        r'[Tt]herefore[,\s]+(.+?)(?:\n|$)',
    ]

    for pattern in final_patterns:
        matches = re.findall(pattern, text)
        if matches:
            final_answer = matches[-1].strip()
            logger.info(f"📝 EXTRACTION: Found answer via pattern '{pattern}': {final_answer}")
            return final_answer

    # Last resort: Return last paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        final_para = paragraphs[-1]
        logger.info(f"📝 EXTRACTION: Using last paragraph ({len(final_para)} chars)")
        return final_para

    logger.warning("⚠️  EXTRACTION: No clear math answer found, returning full text")
    return text


def extract_generic_answer(text: str) -> str:
    """
    Extract answer for generic (non-code, non-math) problems
    Returns the last paragraph or sentence as the final answer
    For proof-based problems, may return the full text if no clear answer section exists
    """
    # Check if this looks like a proof problem (geometry, proofs, etc.)
    proof_indicators = ['proof', 'QED', 'proven', 'demonstrate', 'show that', 'prove that']
    is_proof = any(indicator.lower() in text.lower() for indicator in proof_indicators)

    # Try to find conclusion markers
    conclusion_markers = [
        'In conclusion',
        'Therefore',
        'Thus',
        'Hence',
        'Finally',
        'The answer is',
        'The final answer',
    ]

    for marker in conclusion_markers:
        if marker in text:
            # Get text after last occurrence of marker
            parts = text.rsplit(marker, 1)
            if len(parts) > 1:
                answer = parts[1].strip()
                # Get first sentence/paragraph after marker
                first_para = answer.split('\n\n')[0].strip()
                if len(first_para) > 20:  # Ensure it's substantial
                    logger.info(f"📝 EXTRACTION: Found answer after '{marker}' ({len(first_para)} chars)")
                    return first_para

    # For proof problems, return more context (last 2-3 paragraphs or full text if short)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if is_proof and paragraphs:
        # For proofs, include conclusion paragraphs (last 2-3 paragraphs)
        if len(paragraphs) >= 3:
            conclusion_text = '\n\n'.join(paragraphs[-3:])
            logger.info(f"📝 EXTRACTION: Proof detected, using last 3 paragraphs ({len(conclusion_text)} chars)")
            return conclusion_text
        else:
            # Short proof, return full text
            logger.info(f"📝 EXTRACTION: Short proof detected, returning full text ({len(text)} chars)")
            return text

    # For non-proof problems, return last paragraph
    if paragraphs:
        final_para = paragraphs[-1]
        logger.info(f"📝 EXTRACTION: Using last paragraph ({len(final_para)} chars)")
        return final_para

    # Last resort: Return last sentence
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if sentences:
        final_sentence = sentences[-1] + '.'
        logger.info(f"📝 EXTRACTION: Using last sentence ({len(final_sentence)} chars)")
        return final_sentence

    logger.warning("⚠️  EXTRACTION: No clear answer found, returning full text")
    return text


def wrap_with_thinking_tags(reasoning: str, final_answer: str) -> str:
    """
    Wrap reasoning in <think> tags and append clean final answer

    Args:
        reasoning: All intermediate reasoning, logs, agent outputs
        final_answer: Clean final answer extracted from synthesis

    Returns:
        Formatted output with thinking tags
    """
    return f"<think>\n{reasoning}\n</think>\n\n{final_answer}"


def strip_thinking_tags(text: str) -> str:
    """
    Remove <think></think> tags from text (for debugging/logging)

    Args:
        text: Text potentially containing thinking tags

    Returns:
        Text with thinking tags removed
    """
    # Remove thinking tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def get_answer_after_thinking(text: str) -> str:
    """
    Extract only the content after </think> tag

    Args:
        text: Text with thinking tags

    Returns:
        Content after </think> tag, or full text if no tags
    """
    match = re.search(r'</think>\s*(.+)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text