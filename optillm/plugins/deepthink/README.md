# Deep Think Plugin

## Overview

The Deep Think plugin combines two powerful approaches for enhanced reasoning in large language models:

1. **SELF-DISCOVER Framework**: A method where LLMs self-discover task-intrinsic reasoning structures by selecting, adapting, and implementing atomic reasoning modules into a coherent reasoning plan.

2. **Uncertainty-Routed Chain-of-Thought**: An approach that generates multiple chain-of-thought samples, evaluates confidence through consistency, and routes to either majority voting (high confidence) or greedy decoding (low confidence).

## Key Features

- **Adaptive Reasoning Structure**: Automatically discovers the best reasoning approach for each specific task
- **Confidence-Based Routing**: Uses uncertainty estimation to decide between multiple samples or single greedy output
- **Reasoning Model Support**: Designed for models that produce structured thinking in `<think></think>` tags
- **Multiple Sampling**: Generates multiple reasoning paths and selects the most reliable one

## How It Works

### Stage 1: SELF-DISCOVER Reasoning Structure

1. **SELECT**: From 39 atomic reasoning modules, select those most relevant for the task
2. **ADAPT**: Rephrase selected modules to be task-specific
3. **IMPLEMENT**: Create a structured JSON reasoning plan

### Stage 2: Uncertainty-Routed Generation

1. **Multiple Sampling**: Generate n samples (default: 3) using the discovered structure
2. **Confidence Evaluation**: Assess consistency across samples
3. **Route Decision**: 
   - High confidence → Use majority vote
   - Low confidence → Use greedy sample (temperature=0)

## Usage

```python
# Via optillm model prefix
model = "deepthink-your-model-name"

# Via optillm_approach in request
{
    "model": "your-model-name",
    "optillm_approach": "deepthink",
    "messages": [...],
    "deepthink_samples": 3,           # Number of samples for uncertainty routing
    "confidence_threshold": 0.7,      # Threshold for majority vs greedy routing
    "max_tokens": 16382,             # Extended context for reasoning
    "temperature": 0.7,              # Default temperature for sampling
    "top_p": 0.95                    # Default top_p for sampling
}
```

## Configuration Parameters

- `deepthink_samples` (int, default=3): Number of reasoning samples to generate
- `confidence_threshold` (float, default=0.7): Confidence threshold for routing decision
- `max_tokens` (int, default=16382): Maximum tokens for generation
- `temperature` (float, default=0.7): Sampling temperature
- `top_p` (float, default=0.95): Top-p sampling parameter
- `enable_self_discover` (bool, default=True): Whether to use SELF-DISCOVER structure
- `reasoning_modules_limit` (int, default=5): Max reasoning modules to select

## Atomic Reasoning Modules

The plugin includes 39 reasoning modules covering:
- Critical thinking and analysis
- Creative and innovative approaches  
- Systems thinking and holistic analysis
- Risk assessment and evaluation
- Step-by-step decomposition
- Collaborative and perspective-taking approaches
- Reflective and meta-cognitive strategies

## Examples

### Mathematical Problem Solving
Input: "Solve: If a train travels 120 miles in 2 hours, how long will it take to travel 300 miles?"

The plugin will:
1. Discover a reasoning structure focused on rate calculations
2. Generate multiple solution paths
3. Evaluate consistency and select the most reliable answer

### Complex Reasoning Task
Input: "Analyze the potential long-term economic impacts of remote work adoption"

The plugin will:
1. Select reasoning modules like systems thinking, risk analysis, and critical thinking
2. Create a structured analysis plan
3. Generate multiple perspectives and synthesize the most coherent analysis

## Implementation Details

- **Reasoning Extraction**: Automatically extracts content from `<think></think>` tags
- **Consistency Scoring**: Uses multiple metrics including answer similarity and reasoning coherence
- **Adaptive Thresholds**: Can be fine-tuned based on model performance
- **Token Efficiency**: Optimized to minimize redundant computation while maximizing reasoning quality

## Performance

The Deep Think approach has shown significant improvements on complex reasoning tasks, with particularly strong results on mathematical competition problems.

### AIME 2025 Results

| Model | Approach | Accuracy | Improvement |
|-------|----------|----------|-------------|
| qwen-3-32b | Baseline | 43.33% | - |
| qwen-3-32b | Deep Think | **63.33%** | **+20.00pp** |

*Experimental settings: max_completion_tokens=16382, temperature=0.7, top_p=0.95*

**Key Findings:**
- **46% relative improvement** over baseline on mathematical reasoning
- **Cerebras inference** was crucial for enabling high inference-time compute without latency penalty
- The combination of SELF-DISCOVER structure discovery and uncertainty-routed sampling proved particularly effective for competition mathematics
- Enhanced accuracy on multi-step problems requiring systematic reasoning

### Other Improvements

The Deep Think approach has also demonstrated:
- Enhanced accuracy on multi-step problems
- Better handling of ambiguous or open-ended questions
- Improved consistency across different problem types
- Reduced hallucination through confidence-based routing

## Limitations

- Increased computational cost due to multiple sampling
- Longer response times for complex reasoning tasks
- Requires models capable of structured thinking output
- May over-engineer solutions for simple problems

## References

- Zhou, P. et al. "SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures" (2024)
- Uncertainty-routed chain-of-thought approaches in advanced reasoning systems is from the paper "Gemini: A Family of Highly Capable Multimodal Models" (2023), https://arxiv.org/abs/2312.11805
