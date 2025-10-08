# Evaluation

This directory contains benchmarks, evaluation scripts, and performance metrics for LLMs.

## 📈 Available Benchmarks

Browse the folders below to find evaluation tools. Each includes:
- Evaluation script implementation
- Benchmark data or download links
- Documentation (`.md` file)
- Metrics calculation
- Usage examples
- Result interpretation guides

## 🔍 Finding Evaluation Tools

**By Task:**
- Language Understanding (MMLU, HellaSwag, ARC)
- Code Generation (HumanEval, MBPP)
- Mathematical Reasoning (GSM8K, MATH)
- Common Sense (PIQA, WinoGrande)
- Question Answering (TriviaQA, NaturalQuestions)
- Summarization (CNN/DailyMail, XSum)
- Translation (WMT, FLORES)

**By Domain:**
- General knowledge
- Code and programming
- Mathematics
- Science
- Multilingual
- Domain-specific (medical, legal, etc.)

**By Metric:**
- Accuracy
- F1 Score
- BLEU/ROUGE
- Perplexity
- Pass@k
- Human evaluation

## 📝 Contributing Evaluations

To contribute a benchmark or evaluation:

1. Create a new folder: `evaluation/your_benchmark_name/`
2. Add evaluation code and data
3. Create documentation: `your_benchmark_name.md` (use [TEMPLATE.md](TEMPLATE.md))
4. Include:
   - Overview and what it measures
   - Dataset information
   - Metrics and interpretation
   - Usage examples
   - Baseline results
   - Comparison tables

See [TEMPLATE.md](TEMPLATE.md) for the full documentation template.

## 🏷️ Common Tags

Use these tags in your evaluation documentation:
- `#benchmark` `#evaluation` `#metrics` `#performance`
- `#MMLU` `#HumanEval` `#GLUE` `#SuperGLUE` `#GSM8K`
- `#accuracy` `#f1` `#bleu` `#rouge` `#perplexity`
- `#classification` `#generation` `#reasoning` `#QA`
- `#general` `#code` `#math` `#science` `#multilingual`

## 📖 Documentation Template

See [TEMPLATE.md](TEMPLATE.md) for a comprehensive template with all required sections.

## 🎯 Best Practices

When creating evaluation documentation:
- Clearly describe what is being measured
- Provide baseline results for comparison
- Explain metrics and their interpretation
- Include few-shot and zero-shot settings
- Document exact evaluation protocol
- Show result breakdowns by category
- Add reproduction instructions
- Link to leaderboards if available

## 📊 Standard Benchmarks

Common benchmarks you might find:

**Language Understanding:**
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Common sense reasoning)
- ARC (AI2 Reasoning Challenge)
- TruthfulQA (Truthfulness evaluation)

**Code Generation:**
- HumanEval (Python code generation)
- MBPP (Mostly Basic Python Problems)

**Mathematics:**
- GSM8K (Grade school math)
- MATH (Competition-level math)

**Multilingual:**
- XNLI (Cross-lingual NLI)
- FLORES (Translation)

## 📄 License

Each evaluation tool may have its own license. Check the individual documentation for licensing information.

---

**Keywords**: #evaluation #benchmark #metrics #performance #assessment #LLM #NLP
