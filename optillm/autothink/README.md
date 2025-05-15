# AutoThink

AutoThink is an adaptive thinking approach for Large Language Models that combines query complexity classification with steering vector guidance to enhance model reasoning capabilities.

## Overview

AutoThink combines several advanced techniques to optimize the thinking process of LLMs:

1. **Query Complexity Classification**: Uses an adaptive classifier to determine if a query requires HIGH or LOW complexity reasoning
2. **Token Budget Allocation**: Dynamically allocates thinking tokens based on query complexity
3. **Steering Vector Guidance**: Applies activation-based steering vectors to guide the model's reasoning process
4. **Controlled Thinking Process**: Manages explicit thinking phases with start and end tokens

## How It Works

### 1. Query Classification

AutoThink uses the `adaptive-classifier/llm-router` [model](https://huggingface.co/adaptive-classifier/llm-router) to classify incoming queries:

- **HIGH**: Complex queries requiring deep reasoning, multi-step calculations, or thorough exploration
- **LOW**: Simpler queries requiring less extensive reasoning

### 2. Token Budget

Based on the classification, AutoThink allocates different token budgets for the thinking phase:

- **HIGH**: 70-90% of max tokens allocated for thinking
- **LOW**: 20-40% of max tokens allocated for thinking

### 3. Steering Vectors

AutoThink uses pre-extracted steering vectors from [datasets](https://huggingface.co/datasets?other=pts) like `codelion/Qwen3-0.6B-pts-steering-vectors`. These vectors represent different reasoning patterns:

- **Depth and thoroughness**: Encourages detailed, step-by-step reasoning
- **Numerical accuracy**: Promotes precise calculations and verification
- **Self-correction**: Facilitates error detection and correction
- **Exploration**: Supports considering multiple approaches
- **Organization**: Improves logical structure in responses

During inference, the model's internal activations are modified based on these vectors to enhance specific reasoning capabilities.

### 4. Controlled Thinking Process

The generation process includes:
1. A thinking phase marked by `<think>` and `</think>` tokens
2. Automatic adjustment of thinking time based on query complexity
3. Dynamic application of steering vectors
4. Graceful transition to the final response

## Configuration

AutoThink can be configured with:

```python
{
    "model_name": "your-model-name",
    "classifier_model": "adaptive-classifier/llm-router",
    "steering_dataset": "codelion/Qwen3-0.6B-pts-steering-vectors",
    "target_layer": 19,  # Layer to apply steering vectors
    "high_complexity_min_tokens": 1024, 
    "high_complexity_max_tokens": 4096,
    "low_complexity_min_tokens": 256,
    "low_complexity_max_tokens": 1024,
    "pattern_strengths": {
        "depth_and_thoroughness": 2.5,  # Steering strength for different patterns
        "numerical_accuracy": 2.0,
        "self_correction": 3.0,
        "exploration": 2.0,
        "organization": 1.5
    }
}
```

## Usage

```python
from optillm.autothink import autothink_decode

response = autothink_decode(
    model,
    tokenizer,
    messages,
    {
        "steering_dataset": "codelion/Qwen3-0.6B-pts-steering-vectors",
        "target_layer": 19
    }
)
```

## Benefits

- **Adaptive Resource Usage**: Models think more on complex problems and less on simple ones
- **Enhanced Reasoning**: Steering vectors guide the model toward better reasoning patterns
- **Efficiency**: Better performance without increasing model size
- **Customizability**: Can be tailored for different domains using domain-specific steering vector datasets


## Citation

If you use this approach in your research, please cite:

```bibtex
@article{autothink,
  title={AutoThink: efficient inference for reasoning LLMs},
  author={Sharma, Asankhaya},
  journal={SSRN Artificial Intelligence eJournal},
  year={2025},
  url = {https://dx.doi.org/10.2139/ssrn.5253327}
}
```
