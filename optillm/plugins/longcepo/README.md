# The Long-Context Cerebras Planning and Optimization (LongCePO) Method

LongCePO is an inference-time computation method designed to provide LLMs with the capability to work with infinite context such as external knowledge bases that can run into millions of tokens. We achieve this goal through a combination of multiple strategies including planning (query decomposition) and divide-and-conquer long-context processing. This approach enables to use a limited context window (e.g. 8K) and outperform full-context processing with the same base model in many question-answering tasks.

If you have any questions or want to contribute, please reach out to us on [cerebras.ai/discord](https://cerebras.ai/discord).

## Usage

Start the optillm proxy server with directory to plugins specified in the command line:

```bash
python optillm.py --base-url https://api.cerebras.ai/v1 --port <port> --plugins-dir ./optillm/plugins
```

Now, you can send requests to the proxy using model name `longcepo-{model_name}` (e.g. `longcepo-llama-3.3-70b`) using the following format of the user message: `{context}<CONTEXT_END>{query}`. The `<CONTEXT_END>` delimiter string is used to split the user message into the (long) context and the user's query, respectively. This delimiter string can be changed (along with other LongCePO parameters) in the [config file](./config.py).


## LongCePO Results

LongCePO excels at tasks with long context (128K tokens and more) which is demonstrated below on LongBench v2 and HELMET benchmarks in comparison to frontier models. We additionally provide data points for tasks with shorter context that still exceeds the context window of 8K (HotpotQA and MuSiQue samples of 12-16K length). For our evaluations, we report mean and standard deviation of the target metric over 5 runs below.

### LongBench v2

| Model¹                             | Context window | Short samples (up to 32K words) | Medium samples (32–128K words) |
|----------------------------------|----------------|------------------|----------------|
| Llama 3.3 70B Instruct           | 128K           | 36.7 (45.0)               | 27.0 (33.0)            |
| **LongCePO + Llama 3.3 70B Instruct** | **8K**             | **36.8 ± 1.38**        |  **38.7 ± 2.574 (39.735)²**             |
| Mistral-Large-Instruct-2411     | 128K           | 41.7 (46.1)                 | 30.7 (34.9)             |
| o1-mini-2024-09-12               | 128K           | 48.6 (48.9)                | 33.3 (32.9)            |
| Claude-3.5-Sonnet-20241022       | 200K           | 46.1 (53.9)                | 38.6 (41.9)            |
| Llama-4-Maverick-17B-128E-Instruct | 524K         | 32.22 (50.56)                  | 28.84 (41.86)               |

 ¹ Performance numbers reported by LongBench v2 authors, except for LongCePO and Llama-4-Maverick results. Results in parentheses reported in LongBench v2 correspond to Chain-of-Thought prompting.

 ² Results in parentheses for LongCePO indicate accuracy of majority voting from 5 runs.

### HELMET (InfiniteBench En.MC, 128K length)

| Model   | Accuracy (%) |
|---------|---------------|
| Llama 3.3 70B Instruct  (full context)  | 58.0          |
| **LongCePO + Llama 3.3 70B Instruct (8K context)** | **71.6 ± 1.855 (73.0)¹**  |
| o1-mini-2024-09-12 (full context) | 58.0          |
| gpt-4o-2024-08-06 (full context) | 74.0          |

 ¹ Numbers in parentheses for LongCePO indicate accuracy of majority voting from 5 runs.

### LongBench v1 (HotpotQA, 12K+ length - 124 samples)

| Model   | F1 Metric (%) | LLM-as-a-judge accuracy (%) |
|---------|---------------|-----------------------------|
| Llama 3.3 70B Instruct (full context)  |   63.372 ± 0.269         |   77.903 ± 0.822                      |
| **LongCePO + Llama 3.3 70B Instruct (8K context)** |  **64.842 ± 1.295**            |   **79.355 ± 1.66**                  |

### LongBench v1  (MuSiQue, 12K+ length - 191 samples)

| Model   | F1 Metric (%) | LLM-as-a-judge accuracy (%) |
|---------|---------------|-----------------------------|
| Llama 3.3 70B Instruct  (full context) |    48.481 ± 0.641        |     49.424 ± 0.71                     |
| **LongCePO + Llama 3.3 70B Instruct (8K context)** |  **54.076 ± 2.059**     |     **60.628  ±  2.156**                 |


## LongCePO Methodology

LongCePO is based on the [LLM×MapReduce](https://arxiv.org/abs/2410.09342) approach to long document processing, adding a planning layer on top of a map-reduce-based question-answering engine. We also improve upon the map-reduce approach itself by (i) adding query-aware summaries of neighboring document chunks during the map stage of the processing, (ii) reducing the collapse (merging) stage to a minimum required number of collapse iterations by using a sliding window to iteratively merge pairs of summaries, (iii) using a customized system prompt produced with an [OPRO-like](https://arxiv.org/abs/2309.03409) optimization approach to enhance question-anwering performance. Given a user query, a plan consisting of sub-queries is generated from a normalized query; a map-reduce question-answering engine is then run for each sub-query consecutively, conditioned on the answers to previous sub-queries. Finally, the answer to original user's query is produced via map-reduce conditioned on answers to the whole plan. Similarly to [LLM×MapReduce](https://arxiv.org/abs/2410.09342), we retain the structured information protocol for producing document chunk summaries. We find that splitting the document into chunks of size smaller than the available context window (e.g. chunks of 4K size with available context window of 8K) leads to better performance, and use the remaning context budget to incorporate summaries from neighboring chunks into the map stage for each respective chunks, leading to a further boost in overall performance.

Note: the system prompt for Map/Collapse/Reduce stages has been optimized for the Llama3.3-70B-Instruct model, when using other base models with LongCePO, a more general system prompt can be used ([example](https://github.com/DenisSergeevitch/chatgpt-custom-instructions)).


## LongCePO Current Status

This project is a work in progress, and the provided code is in an early experimental stage. While the proposed approach works well across the benchmarks we tested, further improvements can be achieved through a smart organization of the external knowledge base as well as customization of the plan generation to different tasks. For updates on LongCePO, [follow us on X](https://x.com/cerebrassystems) and join our [Discord](https://cerebras.ai/discord)!


## References

1. Zhou, Zihan, et al. *LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models.* arXiv preprint arXiv:2410.09342 (2024).

2. Yang, Chengrun, et al. *Large language models as optimizers.* arXiv preprint arXiv:2309.03409 (2023).

## Citing LongCePO

```bibtex
@misc{
    cerebras2025longcepo,
    author = {Lazarevich, Ivan and Hassanpour, Mohammad and Venkatesh, Ganesh},
    title = {LongCePO: Empowering LLMs to efficiently leverage infinite context},
    month = March,
    year = 2025,
    howpublished = {\url{https://cerebras.ai/blog/longcepo}, }
}
```