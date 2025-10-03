# MARS: Multi-Agent Reasoning System

A sophisticated multi-agent reasoning system designed for challenging mathematical problems, inspired by systems like Gemini 2.5 Pro Deep Think and the successful IMO25 solver.

## Overview

MARS leverages multiple AI agents working collaboratively to solve complex mathematical problems through:
- **Multi-agent exploration** with diverse reasoning approaches (3 agents by default, configurable)
- **Rigorous verification** using a 2-pass consensus threshold (configurable)
- **Iterative improvement** based on verification feedback
- **OpenRouter reasoning API** for deep mathematical thinking
- **RSA-inspired aggregation** for solution refinement
- **Strategy network** for cross-agent insight sharing
- **Shared workspace** for agent collaboration

## Key Features

### 1. Multi-Agent Architecture
- **3 parallel agents** by default (configurable: 2 for lightweight, 3+ for advanced)
- **Temperature diversity** (0.3, 0.6, 1.0) ensures varied exploration strategies
- **Independent reasoning** followed by collaborative verification

### 2. OpenRouter Reasoning API Integration
- **Effort-based reasoning**: "low", "medium", "high" effort levels via OpenRouter API
- **Adaptive allocation**: Low effort (temp ≤ 0.4), Medium (0.4-0.8), High (> 0.8)
- **Configurable token budgets**: 4K for lightweight coding, 64K for complex reasoning

### 3. Verification System
- **2-pass threshold** by default (configurable: 1 for lightweight, 2+ for advanced)
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
├── __init__.py              # Package exports
├── mars.py                  # Main orchestration with parallel execution
├── agent.py                 # Individual agent implementation
├── workspace.py             # Shared collaboration workspace
├── verifier.py              # Multi-pass verification system
├── aggregator.py            # RSA-inspired solution aggregation
├── strategy_network.py      # Cross-agent insight sharing
├── answer_extraction.py     # Clean answer extraction with thinking tags
├── prompts.py               # Mathematical reasoning prompts
└── README.md                # This documentation
```

## Configuration

### Default Configuration (Mathematical Reasoning)
```python
DEFAULT_CONFIG = {
    'num_agents': 3,                        # Number of parallel agents
    'max_iterations': 5,                    # Maximum improvement iterations
    'verification_passes_required': 2,      # Consecutive passes needed
    'consensus_threshold': 2,               # Verified solutions for consensus
    'min_verified_solutions': 1,            # Minimum to proceed
    'max_tokens': 64000,                    # Token budget for complex reasoning
    'max_verification_attempts': 3,         # Max verification retries
    'early_termination': True,              # Stop on consensus
    'use_reasoning_api': True,              # Enable OpenRouter reasoning
    # RSA-inspired aggregation
    'enable_aggregation': True,             # Enable solution aggregation
    'population_size': 6,                   # Population for diversity
    'aggregation_size': 3,                  # Solutions per aggregation
    'aggregation_loops': 3,                 # Aggregation iterations
    # Strategy Network
    'enable_strategy_network': True,        # Cross-agent insight sharing
    'strategy_extraction_enabled': True,    # Extract reasoning strategies
    'cross_agent_enhancement': True,        # Enhanced solutions via peer strategies
    # Thinking tags and answer extraction
    'use_thinking_tags': True,              # Wrap reasoning in <think> tags
    'answer_extraction_mode': 'auto',       # 'auto', 'code', 'math', or 'none'
}
```

### Lightweight Configuration (Coding Benchmarks)
```python
LIGHTWEIGHT_CONFIG = {
    'num_agents': 2,                        # Reduced agent count
    'max_iterations': 2,                    # Faster iteration limit
    'verification_passes_required': 1,      # Single-pass verification
    'consensus_threshold': 1,               # Lower threshold for 2 agents
    'min_verified_solutions': 1,
    'max_tokens': 4000,                     # Smaller token budget
    'max_verification_attempts': 2,
    'early_termination': True,
    'use_reasoning_api': True,
    # Disable expensive features for speed
    'enable_aggregation': False,            # Skip RSA aggregation
    'enable_strategy_network': False,       # Skip strategy network
    'strategy_extraction_enabled': False,
    'cross_agent_enhancement': False,
    # Thinking tags still enabled
    'use_thinking_tags': True,
    'answer_extraction_mode': 'auto',
}
```

**Note**: MARS automatically uses lightweight config when `max_tokens ≤ 4000` in the request.

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

### Phase 1: Multi-Agent Exploration (Parallel)
1. Initialize 3 agents with diverse temperatures (0.3, 0.6, 1.0)
2. Each agent independently analyzes the problem
3. Generate initial solutions using OpenRouter reasoning API with effort levels
4. All agent API calls executed in parallel via ThreadPoolExecutor
5. Solutions stored in shared workspace

### Phase 2a: RSA-Inspired Aggregation (Optional, Parallel)
1. Maintain population of N=6 solutions for diversity
2. Select K=3 solutions for aggregation
3. Run T=3 aggregation loops to refine solutions
4. Parallel execution of aggregation API calls
5. Enhanced solutions added back to workspace

### Phase 2b: Cross-Agent Strategy Network (Optional, Parallel)
1. Extract reasoning strategies from agent solutions
2. Identify successful patterns and techniques
3. Share strategies across agents
4. Generate enhanced solutions using peer insights
5. Parallel execution of strategy extraction and enhancement

### Phase 3: Verification System (Parallel)
1. Cross-agent verification of all solutions
2. Each solution requires 2 consecutive "CORRECT" assessments (configurable)
3. Verification feedback captured for improvement
4. Solutions marked as verified/unverified
5. Parallel execution of verification calls

### Phase 4: Iterative Improvement (Parallel)
1. Unverified solutions improved based on feedback
2. Agents address specific issues identified in verification
3. Re-verification of improved solutions
4. Process continues until consensus or max iterations (5 default)
5. Parallel execution of improvement and verification

### Phase 5: Final Synthesis
1. **Numerical voting**: If 2+ agents agree on same numerical answer, use that solution
2. **Best verified solution**: Otherwise, select highest-scoring verified solution
3. **Synthesis**: If no verified solution, synthesize from top 3 solutions
4. **Answer extraction**: Apply thinking tags and extract clean answer (if enabled)
5. Complete solution with mathematical rigor

## Evaluation

MARS is designed to excel on challenging mathematical benchmarks:

- **IMO (International Mathematical Olympiad)**: Complex proof-based problems
- **AIME (American Invitational Mathematics Examination)**: Numerical competition problems
- **LiveCodeBench**: Competitive programming challenges
- **Mathematical reasoning tasks**: General problem-solving capabilities

### Performance Metrics
- **Accuracy**: Percentage of correctly solved problems
- **Verification Rate**: Percentage of solutions passing 5-pass threshold
- **Reasoning Efficiency**: Tokens used per correct solution
- **Consensus Quality**: Agreement between verified solutions

## Benchmark Results

### Gemini 2.5 Flash Lite Preview Model

Evaluation results using `google/gemini-2.5-flash-lite-preview-09-2025` via OpenRouter:

| Benchmark | Approach | Problems | Correct | Accuracy | Notes |
|-----------|----------|----------|---------|----------|-------|
| **AIME 2025** | Baseline | 30 | 13 | 43.3% | Pass@1, max_tokens=4000 |
| **AIME 2025** | MARS | 30 | 22 | 73.3% | **+9 problems (+30pp)** |
| **IMO 2025** | Baseline (lite) | 6 | 1 | 16.7% | Problem 4 correct |
| **IMO 2025** | MARS (lite) | 6 | 2 | 33.3% | **+1 problem (+16.6pp)** |
| **LiveCodeBench v5/v6** | Baseline | 105 | 41 | 39.05% | Code generation, pass@1 |
| **LiveCodeBench v5/v6** | MARS + Thinking | 105 | 53 | 50.48% | **+12 problems (+29.3%)** |

### Key Findings

#### AIME 2025: Significant Accuracy Improvement
- **Results**: 22/30 problems solved (73.3%) vs baseline 13/30 (43.3%)
- **Improvement**: +9 problems (+69.2% relative improvement), +30.0 percentage points
- **Key Success Factor**: Multi-agent collaboration with verification effectively solves numerical competition problems
- **Approach**: 3 agents with diverse temperatures, iterative verification and refinement

#### IMO 2025: Proof-Based Competition Problems

- **Results**: 2/6 problems solved (33.3%) vs baseline 1/6 (16.7%)
- **Improvement**: +1 problem (+100% relative improvement), +16.6 percentage points
- **Problems Solved**: Problem 2 (geometry proof) + Problem 4 (number theory)
- **Runtime**: ~10 minutes per problem (vs ~40 seconds baseline)
- **Key Success Factor**: Multi-agent exploration with disabled thinking tags allows full proof visibility
- **Configuration**: `use_thinking_tags=False`, `answer_extraction_mode="none"` for proof problems

#### LiveCodeBench: Strong Performance with Thinking Tags
- **Results**: 53/105 problems solved (50.48%) vs baseline 41/105 (39.05%)
- **Improvement**: +12 problems (+29.3% relative improvement), +11.43 percentage points
- **Code Extraction**: 87/105 (82.9%) vs baseline 54/105 (51.4%) - **+61.1% improvement**
- **Key Success Factor**: Thinking tags beneficial for code generation - allows agents to reason through logic before writing code
- **Multi-agent benefit**: Different temperature agents explore varied solution approaches

#### Lessons Learned
1. **MARS excels at numerical competition problems**: +69.2% relative improvement on AIME 2025 (43.3% → 73.3%)
2. **MARS improves proof-based problems**: +100% relative improvement on IMO 2025 (16.7% → 33.3%)
3. **Thinking tags are problem-type dependent**:
   - ✅ **Enable for code generation**: +29.3% improvement on LiveCodeBench
   - ✅ **Enable for numerical problems**: Multi-agent reasoning effective on AIME
   - ❌ **Disable for proof problems**: IMO proofs need full visibility to evaluators
4. **Multi-agent diversity** provides significant value - different temperature agents explore complementary approaches
5. **Code extraction rate** is a leading indicator - MARS achieved 82.9% vs baseline 51.4% (+61.1%)

### Completed Evaluations

- ✅ **AIME 2025**: Baseline 13/30 (43.3%) → MARS 22/30 (73.3%) **+30pp improvement**
- ✅ **IMO 2025**: Baseline 1/6 (16.7%) → MARS 2/6 (33.3%) **+16.6pp improvement**
- ✅ **LiveCodeBench v5/v6**: Baseline 41/105 (39.05%) → MARS 53/105 (50.48%) **+11.43pp improvement**

*All evaluations use gemini-2.5-flash-lite-preview-09-2025 model via OpenRouter.*

### Configuration for IMO Proof Problems
For proof-based problems like IMO, disable thinking tags to ensure full proof visibility:
```python
extra_body = {
    "optillm_approach": "mars",
    "mars_config": {
        "use_thinking_tags": False,        # Full proof visibility
        "answer_extraction_mode": "none"   # Proofs are the answer
    }
}
```

*All evaluations use pass@1 accuracy metric.*

## Implementation Details

### Temperature Diversity Strategy (3-Agent Default)
- **Agent 0**: Temperature 0.3 (Conservative, rigorous, low effort)
- **Agent 1**: Temperature 0.6 (Balanced approach, medium effort)
- **Agent 2**: Temperature 1.0 (Maximum exploration, high effort)

**Note**: Temperature assignments cycle for configurations with more agents (e.g., 5 agents: 0.3, 0.6, 1.0, 0.3, 0.6)

### Reasoning Effort Allocation (OpenRouter API)
- **Low effort** (temp ≤ 0.4): `{"reasoning": {"effort": "low"}}` - Conservative reasoning
- **Medium effort** (0.4 < temp ≤ 0.8): `{"reasoning": {"effort": "medium"}}` - Balanced reasoning
- **High effort** (temp > 0.8): `{"reasoning": {"effort": "high"}}` - Maximum reasoning depth

**Note**: OpenRouter's reasoning API automatically allocates appropriate thinking tokens based on effort level and model capabilities.

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