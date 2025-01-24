# The Cerebras Planning and Optimization (CePO) Method

CePO is an inference-time computation method designed to enhance the accuracy of large language models (LLMs) on tasks requiring reasoning and planning, such as solving math or coding problems. It integrates several advanced techniques, including Best of N, Chain of Thought (CoT), Self-Reflection, Self-Improvement, and Prompt Engineering.

If you have any questions or want to contribute, please reach out to us on [cerebras.ai/discord](https://cerebras.ai/discord)

## CePO Methodology

In CePO, the Best of N technique is applied to `bestofn_n` solution candidates. Each solution is generated through the following four steps:

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

## CePO Current Status

This project is a work in progress, and the provided code is in an early experimental stage. While the proposed approach works well across the benchmarks we tested, further improvements can be achieved by task-specific customizations to prompts.

## CePO Ablation studies

We conducted ablation studies to evaluate the impact of various hyperparameters in the CePO framework. Our results indicate that the chosen hyperparameter settings strike a good balance between computational cost and accuracy.

Interestingly, the self-critique and quality improvement capabilities of existing off-the-shelf models do not always scale proportionally with increased inference compute. Addressing this limitation remains a key focus, and we plan to explore custom model fine-tuning as a potential solution in the future.

| bestofn_n | planning_n | planning_m | bestofn_rating_type | Math-L5 | MMLU-Pro (Math) | GPQA  | CRUX  | Comments       |
| :-------: | :--------: | :--------: | :-----------------: | :-----: | :-------------: | :---: | :---: | :------------- |
|     3     |      3     |      6     |       absolute      |  69.6   |      84.8       | 55.5  | 80.1  | Default config |
|     3     |      3     |      6     |       pairwise      |  67.7   |      83.5       | 55.6  | 79.8  |                |
|     3     |      2     |      5     |       absolute      |  67.1   |      85.1       | 55.1  | 79.0  |                |
|     3     |      5     |      8     |       absolute      |  69.4   |      84.3       | 55.6  | 81.1  |                |
|     5     |      3     |      6     |       absolute      |  68.7   |      85.4       | 54.8  | 79.9  |                |
|     7     |      3     |      6     |       absolute      |  69.6   |      82.8       | 54.7  | 78.4  |                |
|     9     |      3     |      6     |       absolute      |  68.9   |      83.4       | 55.7  | 80.6  |                |
