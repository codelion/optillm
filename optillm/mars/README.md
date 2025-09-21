# MARS: Multi-Agent Reasoning System

A sophisticated multi-agent reasoning system designed for challenging mathematical problems, inspired by systems like Gemini 2.5 Pro Deep Think and the successful IMO25 solver.

## Overview

MARS leverages multiple AI agents working collaboratively to solve complex mathematical problems through:
- **Multi-agent exploration** with diverse reasoning approaches
- **Rigorous verification** using a 5-pass consensus threshold
- **Iterative improvement** based on verification feedback
- **OpenRouter reasoning API** for deep mathematical thinking
- **Shared workspace** for agent collaboration

## Key Features

### 1. Multi-Agent Architecture
- **5 parallel agents** with different temperature settings (0.3-1.0)
- **Temperature diversity** ensures varied exploration strategies
- **Independent reasoning** followed by collaborative verification

### 2. OpenRouter Reasoning API Integration
- **Thinking tokens**: Up to 32,768 tokens for deep reasoning
- **Effort levels**: Low (20%), Medium (50%), High (80%) reasoning budgets
- **Adaptive allocation** based on agent temperature and problem complexity

### 3. Verification System
- **5-pass threshold**: Solutions must pass 5 consecutive verifications
- **Cross-agent verification**: Agents verify each other's solutions
- **Mathematical rigor**: Focus on complete proofs, not just correct answers
- **Consensus building**: Multiple verified solutions required

### 4. Iterative Improvement
- **Feedback-driven**: Solutions improved based on verification feedback
- **Error correction**: Automatic identification and fixing of mathematical errors
- **Logical gap filling**: Strengthening incomplete reasoning steps

## Architecture Components

```
optillm/mars/
├── __init__.py           # Package exports
├── mars.py               # Main orchestration logic
├── agent.py              # Individual agent implementation
├── workspace.py          # Shared collaboration workspace
├── verifier.py           # 5-pass verification system
├── prompts.py            # Mathematical reasoning prompts
└── README.md             # This documentation
```

## Configuration

### Default Configuration
```python
DEFAULT_CONFIG = {
    'num_agents': 5,                     # Number of parallel agents
    'max_iterations': 30,                # Maximum improvement iterations
    'verification_passes_required': 5,   # Consecutive passes needed
    'consensus_threshold': 2,            # Verified solutions for consensus
    'min_verified_solutions': 1,         # Minimum to proceed
    'thinking_budget_initial': 10000,    # Initial reasoning tokens
    'thinking_budget_max': 32000,        # Maximum reasoning tokens
    'max_response_tokens': 4096,         # Maximum response length
    'early_termination': True,           # Stop on consensus
    'use_reasoning_api': True            # Enable OpenRouter reasoning
}
```

## Usage

### Via OptiLLM Server
```bash
# Start OptiLLM with MARS support
python optillm.py --model google/gemma-2.5-flash-lite --approach mars

# Make API call
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mars-google/gemma-2.5-flash-lite",
    "messages": [
      {
        "role": "user",
        "content": "Solve this IMO problem: Find all positive integers n such that..."
      }
    ]
  }'
```

### Via extra_body Parameter
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="anything")

response = client.chat.completions.create(
    model="google/gemma-2.5-flash-lite",
    messages=[
        {"role": "user", "content": "Mathematical problem here"}
    ],
    extra_body={"optillm_approach": "mars"}
)
```

### Via Prompt Tags
```python
response = client.chat.completions.create(
    model="google/gemma-2.5-flash-lite",
    messages=[
        {"role": "system", "content": "<optillm_approach>mars</optillm_approach>"},
        {"role": "user", "content": "Mathematical problem here"}
    ]
)
```

## Process Flow

### Phase 1: Multi-Agent Exploration
1. Initialize 5 agents with diverse temperatures
2. Each agent independently analyzes the problem
3. Generate initial solutions using OpenRouter reasoning API
4. Solutions stored in shared workspace

### Phase 2: Verification System
1. Cross-agent verification of all solutions
2. Each solution requires 5 consecutive "CORRECT" assessments
3. Verification feedback captured for improvement
4. Solutions marked as verified/unverified

### Phase 3: Iterative Improvement
1. Unverified solutions improved based on feedback
2. Agents address specific issues identified in verification
3. Re-verification of improved solutions
4. Process continues until consensus or max iterations

### Phase 4: Final Synthesis
1. Best verified solution selected as final answer
2. If no verified solutions, synthesis from all attempts
3. High-effort reasoning applied to synthesis
4. Complete solution with mathematical rigor

## Evaluation

MARS is designed to excel on challenging mathematical benchmarks:

- **IMO (International Mathematical Olympiad)**: Complex proof-based problems
- **AIME (American Invitational Mathematics Examination)**: Numerical competition problems
- **Mathematical reasoning tasks**: General problem-solving capabilities

### Performance Metrics
- **Accuracy**: Percentage of correctly solved problems
- **Verification Rate**: Percentage of solutions passing 5-pass threshold
- **Reasoning Efficiency**: Tokens used per correct solution
- **Consensus Quality**: Agreement between verified solutions

## Implementation Details

### Temperature Diversity Strategy
- **Agent 0**: Temperature 0.3 (Conservative, rigorous)
- **Agent 1**: Temperature 0.5 (Balanced approach)
- **Agent 2**: Temperature 0.7 (Creative exploration)
- **Agent 3**: Temperature 0.9 (High creativity)
- **Agent 4**: Temperature 1.0 (Maximum exploration)

### Reasoning Budget Allocation
- **Low effort (temp ≤ 0.4)**: 20% of reasoning budget
- **Medium effort (0.4 < temp ≤ 0.7)**: 50% of reasoning budget
- **High effort (temp > 0.7)**: 80% of reasoning budget

### Verification Criteria
Solutions are verified based on:
- **Mathematical correctness**: Accurate calculations and logic
- **Completeness**: All problem aspects addressed
- **Rigor**: Proper justification for each step
- **Clarity**: Clear mathematical communication
- **Format compliance**: Proper answer formatting

## Inspired By

- **IMO25 Solver**: 5/6 problems solved with 5-consecutive-pass verification
- **Gemini 2.5 Pro Deep Think**: Native reasoning tokens and thinking budgets
- **OpenRouter Reasoning API**: Standardized interface for deep thinking
- **CEPO Architecture**: Multi-file approach pattern in OptiLLM

## Future Enhancements

- **Multi-model support**: Different models for different agent roles
- **Dynamic temperature adjustment**: Adaptive exploration strategies
- **Specialized agent roles**: Proof-focused, computation-focused, verification-focused
- **Knowledge base integration**: Access to mathematical theorems and techniques
- **Interactive verification**: Human-in-the-loop verification for critical problems