# The Cerebras Planning and Optimization (CePO) Method

CePO is an inference-time computation method designed to enhance the accuracy of large language models (LLMs) on tasks requiring reasoning and planning, such as solving math or coding problems. It integrates several advanced techniques, including Best of N, Chain of Thought (CoT), Self-Reflection, Self-Improvement, and Prompt Engineering.

If you have any questions or want to contribute, please reach out to us on [cerebras.ai/discord](https://cerebras.ai/discord)

## CePO Methodology

In CePO, the Best of N technique is applied to `bestofn_n` solution candidates. Optionally (when `cepo_use_plan_diversity` is set to `True`), the model will attempt to come up with diverse approaches for each of best of n completions. Each completion is generated through the following four steps:

**Step 1**: Plan Generation
The model generates a detailed, step-by-step plan to solve the problem, along with its confidence level for each step.

**Step 2**: Initial Solution
Using the plan from Step 1, the model produces an initial solution.

Steps 1 and 2 are repeated `planning_n` times to generate multiple solution proposals.
If the model exceeds the token budget during Step 1 or 2, the plan/solution is marked as incomplete, rejected, and regenerated. A maximum of `planning_m` attempts is made to generate `planning_n` valid proposals.

**Step 3**: Plan Refinement
The model reviews all generated solution proposals and their associated plans, identifying inconsistencies. Based on this analysis, a refined, final step-by-step plan is constructed.

**Step 4**: Final Solution
The model uses the refined plan from Step 3 to produce the final answer.

## Example Usage

Here’s an example of running Optillm using the CePO method for Qwen3 deployed with VLLM on port 8001:

```bash
OPENAI_API_KEY=serving-on-vllm \
python optillm.py \
  --base-url http://localhost:8001/v1 \
  --approach cepo \
  --port 8000 \
  --cepo_config_file ./optillm/cepo/cepo_configs/cepo_qwen3.yaml
```

## CePO Current Status

This project is a work in progress, and the provided code is in an early experimental stage. While the proposed approach works well across the benchmarks we tested, further improvements can be achieved by task-specific customizations to prompts.