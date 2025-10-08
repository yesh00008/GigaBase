# [Benchmark/Evaluation Name]

**Quick Summary**: [One sentence describing what this benchmark evaluates]

## Overview

[Provide a comprehensive 2-3 paragraph description. Include:
- What the benchmark measures
- Why it's important
- What tasks or capabilities it evaluates
- How it compares to other benchmarks]

## Keywords

`#benchmark` `#evaluation` `#metrics` `#performance` `#[task-type]` `#[domain]`

[Add relevant tags from the following categories:
- Task type: #classification, #generation, #reasoning, #QA, #translation, #summarization
- Domain: #general, #code, #math, #science, #commonsense, #multilingual
- Metric: #accuracy, #f1, #bleu, #rouge, #perplexity, #pass-rate
- Benchmark: #MMLU, #HumanEval, #GLUE, #SuperGLUE, #BigBench, #HellaSwag]

## Benchmark Details

### What It Measures

**Primary Metrics:**
- [Metric 1]: [Description, e.g., Accuracy - percentage of correct answers]
- [Metric 2]: [Description, e.g., F1 Score - harmonic mean of precision and recall]
- [Metric 3]: [Description, e.g., Pass@k - percentage passing within k attempts]

**Secondary Metrics:**
- [Metric 4]: [Description]
- [Metric 5]: [Description]

### Dataset Information

- **Size**: [e.g., 1000 examples, 10 tasks, 57 subjects]
- **Split**: [e.g., Dev: 500, Test: 500]
- **Format**: [e.g., Multiple choice, Free form, Code completion]
- **Language**: [e.g., English, Multilingual]
- **Domain**: [e.g., General knowledge, Programming, Mathematics]

### Task Description

[Detailed description of what the model needs to do]

**Example Task:**
```
Question: What is the capital of France?
A) London
B) Berlin
C) Paris
D) Madrid

Correct Answer: C
```

## Requirements

### Dependencies

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets
pip install evaluate
pip install scikit-learn
pip install numpy pandas
```

**Full requirements:**
```bash
pip install -r requirements.txt
```

### Hardware

**Minimum:**
- GPU: [e.g., 1x GPU with 8GB VRAM] (or CPU for small models)
- RAM: [e.g., 16GB]

**Recommended:**
- GPU: [e.g., 1x A100 or V100]
- RAM: [e.g., 32GB]

## Installation

```bash
# Clone benchmark repository (if separate)
git clone https://github.com/your-org/benchmark-repo.git
cd benchmark-repo

# Install dependencies
pip install -r requirements.txt

# Download evaluation data
python download_data.py
```

## Usage

### Quick Start

**Basic Evaluation:**
```bash
python evaluate.py \
  --model_name_or_path "gpt2" \
  --batch_size 8 \
  --output_dir ./results
```

**Custom Configuration:**
```bash
python evaluate.py \
  --model_name_or_path "your-model-path" \
  --batch_size 16 \
  --max_length 2048 \
  --num_fewshot 5 \
  --output_dir ./results \
  --save_predictions
```

### Python API

```python
from evaluate_benchmark import BenchmarkEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize evaluator
evaluator = BenchmarkEvaluator(
    model=model,
    tokenizer=tokenizer,
    batch_size=8,
    num_fewshot=5
)

# Run evaluation
results = evaluator.evaluate()

# Print results
print(f"Overall Accuracy: {results['accuracy']:.2f}%")
print(f"Average Score: {results['avg_score']:.2f}")

# Per-category results
for category, score in results['by_category'].items():
    print(f"{category}: {score:.2f}%")
```

### Advanced Usage

**Few-Shot Evaluation:**
```python
# Evaluate with different few-shot settings
for k in [0, 1, 3, 5]:
    results = evaluator.evaluate(num_fewshot=k)
    print(f"{k}-shot accuracy: {results['accuracy']:.2f}%")
```

**Custom Prompt Template:**
```python
evaluator = BenchmarkEvaluator(
    model=model,
    tokenizer=tokenizer,
    prompt_template="Question: {question}\nAnswer: {answer}",
    few_shot_template="{examples}\n\nQuestion: {question}\nAnswer:"
)
```

**Batch Processing:**
```python
# Evaluate multiple models
models_to_evaluate = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

results_dict = {}
for model_name in models_to_evaluate:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    evaluator.model = model
    results_dict[model_name] = evaluator.evaluate()

# Compare results
import pandas as pd
df = pd.DataFrame(results_dict).T
print(df)
```

## Evaluation Protocol

### Zero-Shot Evaluation

```python
# No examples provided
results = evaluator.evaluate(num_fewshot=0)
```

Prompt format:
```
Question: {question}
Answer:
```

### Few-Shot Evaluation

```python
# k examples provided as context
results = evaluator.evaluate(num_fewshot=5)
```

Prompt format:
```
Question: {example_1_question}
Answer: {example_1_answer}

Question: {example_2_question}
Answer: {example_2_answer}

...

Question: {test_question}
Answer:
```

### Evaluation Steps

1. **Load Data**: Load benchmark dataset
2. **Format Prompts**: Create prompts according to benchmark format
3. **Generate Predictions**: Run model inference
4. **Parse Outputs**: Extract answers from model outputs
5. **Calculate Metrics**: Compute evaluation metrics
6. **Aggregate Results**: Combine results across categories

## Metrics

### Primary Metrics

#### Accuracy
```python
accuracy = correct_predictions / total_predictions
```

**Interpretation:**
- 0-25%: Random guessing (for 4-option multiple choice)
- 25-50%: Below average
- 50-70%: Average
- 70-85%: Good
- 85%+: Excellent

#### F1 Score
```python
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)
```

#### Pass@k
```python
# For code generation: percentage of problems with ≥1 correct solution in k attempts
pass_at_k = problems_solved / total_problems
```

### Secondary Metrics

- **Perplexity**: Measure of model's uncertainty
- **Exact Match**: Percentage of exact string matches
- **BLEU/ROUGE**: For generation tasks
- **Macro/Micro Average**: Different averaging strategies

## Results

### Benchmark Results

[Include results from standard models]

| Model | Parameters | Accuracy | F1 Score | Notes |
|-------|-----------|----------|----------|-------|
| Random Baseline | - | 25.0% | - | Random guessing |
| GPT-2 | 125M | 35.2% | 0.42 | Base model |
| GPT-2 Medium | 345M | 42.1% | 0.51 | |
| GPT-2 Large | 774M | 48.7% | 0.58 | |
| LLaMA-2 7B | 7B | 65.3% | 0.73 | |
| LLaMA-2 13B | 13B | 72.4% | 0.79 | |
| GPT-3.5 | - | 85.5% | 0.88 | SOTA |

### Results by Category

[If applicable, break down by task categories]

| Category | GPT-2 | LLaMA-2 7B | GPT-3.5 |
|----------|-------|------------|---------|
| STEM | 32.1% | 58.3% | 78.9% |
| Humanities | 38.5% | 67.2% | 87.3% |
| Social Science | 35.8% | 63.5% | 84.1% |
| Other | 36.2% | 68.9% | 88.7% |

### Performance Analysis

**Key Findings:**
- [e.g., Larger models generally perform better]
- [e.g., Performance varies significantly across categories]
- [e.g., Few-shot learning improves results by 10-15%]

**Error Analysis:**
- [Common error types]
- [Challenging question categories]
- [Model failure modes]

## Interpretation Guide

### Score Ranges

- **0-40%**: Model struggles with the task
- **40-60%**: Basic understanding, significant room for improvement
- **60-75%**: Competent performance
- **75-85%**: Strong performance
- **85%+**: Excellent, approaching human level

### What the Scores Mean

**High Accuracy:**
- Model has good understanding of the domain
- Suitable for practical applications
- May still make errors on edge cases

**Low Accuracy:**
- May need fine-tuning on domain-specific data
- Consider using a larger model
- Check prompt formatting

### Comparing Models

When comparing models:
1. Use same evaluation settings (few-shot, prompts, etc.)
2. Report confidence intervals if possible
3. Consider speed vs accuracy tradeoffs
4. Check performance on relevant subcategories

## Reproducing Results

### Exact Reproduction

```bash
# Use exact configuration
python evaluate.py \
  --model_name_or_path "gpt2" \
  --batch_size 8 \
  --num_fewshot 5 \
  --seed 42 \
  --max_length 2048 \
  --temperature 0.0  # Greedy decoding
```

### Configuration File

```yaml
# eval_config.yaml
model:
  name_or_path: "gpt2"
  dtype: "float16"
  device_map: "auto"

evaluation:
  batch_size: 8
  num_fewshot: 5
  max_length: 2048
  temperature: 0.0
  seed: 42

output:
  save_predictions: true
  output_dir: "./results"
```

Run with config:
```bash
python evaluate.py --config eval_config.yaml
```

## Customization

### Custom Metrics

```python
def custom_metric(predictions, references):
    """
    Implement your custom metric
    """
    score = 0.0
    for pred, ref in zip(predictions, references):
        # Your metric logic
        if pred.lower().strip() == ref.lower().strip():
            score += 1
    return score / len(predictions)

# Add to evaluator
evaluator.add_metric("custom_metric", custom_metric)
```

### Custom Prompts

```python
# Define custom prompt template
CUSTOM_TEMPLATE = """
Context: {context}
Question: {question}
Choices:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
Answer (A/B/C/D):
"""

evaluator.set_prompt_template(CUSTOM_TEMPLATE)
```

### Subset Evaluation

```python
# Evaluate on specific categories only
results = evaluator.evaluate(
    categories=["STEM", "Humanities"],
    max_samples_per_category=100
)
```

## Troubleshooting

### Common Issues

**Issue: Low scores on all benchmarks**
- Check prompt formatting
- Verify model is loaded correctly
- Try few-shot evaluation
- Check for tokenization issues

**Issue: Out of memory**
- Reduce batch size
- Use smaller max_length
- Enable gradient checkpointing (if training)
- Use model quantization

**Issue: Slow evaluation**
- Increase batch size
- Use GPU acceleration
- Enable mixed precision
- Use batched generation

### Validation

```python
# Verify evaluation is working correctly
# Run on small subset first
results = evaluator.evaluate(max_samples=10)
print(results)

# Check predictions manually
predictions = evaluator.get_predictions(max_samples=5)
for i, pred in enumerate(predictions):
    print(f"Sample {i}:")
    print(f"  Question: {pred['question']}")
    print(f"  Predicted: {pred['prediction']}")
    print(f"  Correct: {pred['reference']}")
    print()
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{benchmark_name,
  title={Benchmark Name: Description},
  author={Author Names},
  journal={Conference/Journal},
  year={2024}
}
```

**Original Benchmark:**
```bibtex
@article{original_benchmark,
  title={Original Benchmark Paper},
  author={Original Authors},
  journal={Original Venue},
  year={2023}
}
```

## License

This evaluation code is released under the [LICENSE_NAME] license.

The benchmark dataset may have a different license. Please check the original source.

## Contact

**Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

## Acknowledgments

- Original benchmark creators
- Contributors
- Computing resources

## Related Resources

- **Benchmark Website**: [Link]
- **Leaderboard**: [Link to public leaderboard]
- **Paper**: [Link to paper]
- **Discussion**: [Link to forum/discussion]

---

**Last Updated**: [YYYY-MM-DD]

**Keywords**: #benchmark #evaluation #metrics #performance #LLM #NLP #assessment #open-source
