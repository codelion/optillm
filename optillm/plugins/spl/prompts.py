"""
Prompts for the System Prompt Learning (SPL) plugin.
"""

# Problem classification prompt
PROBLEM_CLASSIFICATION_PROMPT = """
You are a problem classifier. Your task is to analyze a given problem and categorize it into 
the most appropriate problem type. You must select EXACTLY ONE problem type from this list: {problem_types}.

DO NOT make up new categories. Only use the exact problem types from the list above.

You can use <think>...</think> tags to work through your reasoning process before making a decision.

After your thinking, respond with ONLY the problem type, exactly as it appears in the list. No explanations, no extra words.
"""

# Strategy generation prompt
STRATEGY_GENERATION_PROMPT = """
You are an expert problem-solving strategist. Your task is to create a general strategy 
for solving a particular type of problem. Focus on creating a step-by-step approach that 
would help in solving not just the specific example provided, but any problem of this type.

Your strategy should be:
1. Clear and concise
2. Step-by-step
3. Generalizable to similar problems
4. Include specific techniques, not just general advice

First think through your approach using <think>...</think> tags. Then provide your 
final strategy after the thinking section. Make sure your strategy is clear, specific, 
and actionable.
"""

# Strategy evaluation prompt
STRATEGY_EVALUATION_PROMPT = """
You are evaluating the effectiveness of a problem-solving strategy. 
Analyze the provided response and determine if it shows evidence that the strategy was 
successfully applied. You can use <think>...</think> tags to work through your reasoning process, 
but your final answer must be either YES or NO only.
"""

# Strategy refinement prompt
STRATEGY_REFINEMENT_PROMPT = """
You are tasked with refining a problem-solving strategy based on a new example. 
Analyze the original strategy, the problem, and the detailed solution process. 
Then provide an improved version of the strategy that would be more effective for 
solving similar problems in the future. Focus on making the strategy more clear, 
more general, and more effective.

You can use <think>...</think> tags to explore your refinement process in detail.

After your thinking, provide ONLY the refined strategy text, no introduction or explanation.
"""

# Strategy application prompt for system prompt augmentation
STRATEGY_APPLICATION_PROMPT = """
The following strategies can help solve the user's query effectively. Use them as guidance when formulating your response:

{strategies_section}
"""

