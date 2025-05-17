# System Prompt Learning (SPL) Plugin for OptiLLM

This plugin implements Andrej Karpathy's [proposed](https://x.com/karpathy/status/1921368644069765486) "third paradigm" for LLM learning, enabling LLMs to learn and improve their problem-solving strategies over time.

## Concept

While traditional LLM learning involves either:
- **Pretraining**: Learning factual knowledge from data
- **Finetuning**: Learning behavioral patterns through supervision or reinforcement

System Prompt Learning introduces a third paradigm:
- **Strategy Learning**: The model learns explicit problem-solving strategies and remembers them in a growing database
- These strategies can be selectively applied to new problems based on their type and similarity
- The system tracks which strategies work well and refines them over time

## Usage

### Basic Usage

Use the plugin by prefixing your model name with `spl-`:

```
spl-gpt-4o
```

### Combining with Other Plugins

SPL can be combined with other plugins using the `&` operator:

```
spl&memory-gpt-4o
```

### Learning Mode

By default, the plugin runs in inference-only mode, which uses existing strategies without creating or modifying them. To enable learning mode, which allows the plugin to create and refine strategies based on usage, add the `spl_learning` parameter to the request config:

```python
client.chat.completions.create(
    model="spl-gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ],
    extra_body= {"spl_learning": True},
)
```

## How It Works

1. **Problem Classification**: The plugin analyzes each query to determine its problem type
2. **Strategy Selection**: It selects relevant strategies from its database based on the problem type and content
3. **System Prompt Augmentation**: Selected strategies (up to MAX_STRATEGIES_FOR_INFERENCE) are added to the system prompt

When learning mode is enabled, the plugin also performs:
4. **Effectiveness Evaluation**: After generating a response, the system evaluates how well each strategy worked
5. **Strategy Creation & Refinement**: The system creates new strategies for unseen problem types and periodically refines existing strategies based on usage

The plugin maintains two separate limits:
- **Storage Limit** (MAX_STRATEGIES_PER_TYPE): Controls how many strategies can be stored in the database per problem type
- **Inference Limit** (MAX_STRATEGIES_FOR_INFERENCE): Controls how many strategies are used during inference for system prompt augmentation

## Data Storage

Strategies are stored in JSON format in the `spl_data` directory:
- `strategies.json`: Contains all learned strategies
- `metrics.json`: Contains performance metrics and usage statistics

## Configuration

The SPL plugin maintains these core files:
- **Strategy Database**: `/optillm/plugins/spl/data/strategies.json`
- **Metrics**: `/optillm/plugins/spl/data/metrics.json`

You can:
1. Backup these files to preserve learned strategies
2. Edit the strategies.json file to manually add or modify strategies
3. Reset the learning by deleting these files (they will be recreated)

## Example Strategy

A strategy in the database looks like this:

```json
{
  "strategy_id": "strategy_1",
  "problem_type": "arithmetic_calculation",
  "strategy_text": "When solving arithmetic calculations:\n1. Identify the operations needed (addition, subtraction, multiplication, division)\n2. Follow the order of operations (PEMDAS)\n3. Simplify expressions step by step, showing your work\n4. Double-check your calculations with inverse operations",
  "examples": ["Solve 3x + 5 = 14"],
  "success_count": 8,
  "total_attempts": 10,
  "created_at": "2025-05-12T10:15:30.123456",
  "last_used": "2025-05-12T14:25:10.654321",
  "last_updated": "2025-05-12T12:30:45.987654",
  "confidence": 0.8,
  "tags": ["math", "equations"]
}
```

## Benefits

1. **Cumulative Learning**: The LLM improves on specific problem types over time
2. **Explicit Knowledge**: Strategies are human-readable and provide insight into the LLM's reasoning
3. **Efficiency**: Reuses successful approaches rather than solving each problem from scratch
4. **Adaptability**: Different strategies for different problem types
