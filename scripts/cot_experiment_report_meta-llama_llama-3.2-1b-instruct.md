# Chain of Thought (CoT) Experiment Report

Date: 2025-04-24 14:26:39

## Experiment Overview

This experiment tests whether Chain of Thought (CoT) reasoning provides value through coherent reasoning or simply by giving models more computation time/tokens to produce answers.

Three different prompting strategies were compared:

1. **Standard**: Direct answer with no reasoning steps
2. **Chain of Thought (CoT)**: Structured reasoning before the answer
3. **Gibberish CoT**: Meaningless text of similar length to CoT before the answer

## Accuracy Results

| Prompt Type | Problems | Correct | Accuracy |
|-------------|----------|---------|----------|
| Standard | 500 | 86 | 17.20% |
| Cot | 500 | 115 | 23.00% |
| Gibberish | 500 | 20 | 4.00% |

## Response Characteristics

| Prompt Type | Avg Length | Reasoning Quality | Gibberish Level |
|-------------|------------|-------------------|----------------|
| Standard | 1516.2 | 0.45 | 0.34 |
| Cot | 1793.0 | 0.53 | 0.30 |
| Gibberish | 1766.4 | 0.35 | 0.38 |

## Analysis

- CoT improved accuracy by 33.7% compared to the standard prompt.
- Gibberish CoT improved accuracy by -76.7% compared to the standard prompt.
- CoT was 475.0% more accurate than Gibberish CoT.

## Conclusions

The results strongly suggest that structured reasoning is crucial for improved performance. Since Gibberish CoT did not improve over the standard prompt while CoT did, the benefit appears to come from the reasoning process itself rather than just extra computation time.

## Future Work

- Expand testing to different problem domains beyond mathematics
- Test with different model architectures to see if the pattern holds
- Analyze intermediate activation patterns during reasoning vs gibberish generation
- Investigate whether fine-tuning on gibberish CoT would yield similar benefits to CoT fine-tuning
