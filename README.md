# optillm

optillm is an OpenAI compatible optimizing inference proxy which implements several state-of-the-art techniques that can improve the accuracy and performance of LLMs. The current focus
is on implementing techniques that improve reasoning over code, logical and mathematical queries. It is possible to beat the frontier models using these techniques across diverse tasks 
by doing additional compute at inference time. 

## Installation

Just clone the repository with `git` and use `pip install` to setup the dependencies.

```bash
git clone https://github.com/codelion/optillm.git
cd optillm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can then run the optillm proxy as follows.

```bash
python optillm.py                           
2024-09-06 07:57:14,191 - INFO - Starting server with approach: auto
2024-09-06 07:57:14,191 - INFO - Server configuration: {'approach': 'auto', 'mcts_simulations': 2, 'mcts_exploration': 0.2, 'mcts_depth': 1, 'best_of_n': 3, 'model': 'gpt-4o-mini', 'rstar_max_depth': 3, 'rstar_num_rollouts': 5, 'rstar_c': 1.4, 'base_url': ''}
 * Serving Flask app 'optillm'
 * Debug mode: off
2024-09-06 07:57:14,212 - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://192.168.10.48:8000
2024-09-06 07:57:14,212 - INFO - Press CTRL+C to quit
```

### Usage

Once the proxy is running, you can just use it as a drop in replacement for an OpenAI client by setting the `base_url` as `http://localhost:8000/v1`.

```bash
import os
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "http://localhost:8000/v1"
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

response = client.chat.completions.create(
  model="moa-gpt-4o",
  messages=[
    {
      "role": "user",
      "content": "Write a Python program to build an RL model to recite text from any position that the user provides, using only numpy."
    }
  ],
  temperature=0.2
)

print(response)
```

You can control the technique you use for optimization by prepending the slug to the model name `{slug}-model-name`. E.g. in the above code we are using `moa` or 
mixture of agents as the optimization approach. In the proxy logs you will see the following showing the `moa` is been used with the base model as `gpt-4o-mini`.

```bash
2024-09-06 08:35:32,597 - INFO - Using approach moa, with gpt-4o-mini
2024-09-06 08:35:35,358 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:39,553 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:44,795 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:44,797 - INFO - 127.0.0.1 - - [06/Sep/2024 08:35:44] "POST /v1/chat/completions HTTP/1.1" 200 -
```

## Implemented techniques

| Technique | Slug | Description |
|-----------|----------------|-------------|
| Monte Carlo Tree Search | `mcts` | Uses MCTS for decision-making in chat responses |
| Best of N Sampling | `bon` | Generates multiple responses and selects the best one |
| Mixture of Agents | `moa` | Combines responses from multiple critiques |
| Round Trip Optimization | `rto` | Optimizes responses through a round-trip process |
| Z3 Solver | `z3` | Utilizes the Z3 theorem prover for logical reasoning |
| Self-Consistency | `self_consistency` | Implements an advanced self-consistency method |
| PV Game | `pvg` | Applies a prover-verifier game approach at inference time |
| R* Algorithm | `rstar` | Implements the R* algorithm for problem-solving |

## References

- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)
- [Prover-Verifier Games improve legibility of LLM outputs](https://arxiv.org/abs/2407.13692)
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
- [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness](https://arxiv.org/abs/2402.08699)
- [Patched MOA: optimizing inference for diverse software development tasks](https://arxiv.org/abs/2407.18521)
- [Patched RTC: evaluating LLMs for diverse software development tasks](https://arxiv.org/abs/2407.16557)


