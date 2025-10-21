# LLM Development Project

This project aims to build and train large language models (LLMs) similar to GPT-5 and Gemini 2.5, incorporating datasets collected from the internet.

## Project Structure

- `src/`: Source code for model training and inference
- `data/`: Datasets and data processing scripts
- `models/`: Saved model checkpoints and configurations
- `notebooks/`: Jupyter notebooks for experimentation
- `scripts/`: Utility scripts for data collection and training

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure Python environment
3. Run the complete pipeline: `python main.py --all`

## Pipeline Components

### 1. Data Collection
Automatically collects data from various internet sources:
- Wikipedia articles
- Project Gutenberg books
- Stack Exchange Q&A
- News articles from RSS feeds
- Code samples from GitHub
- Academic papers from arXiv
- Reddit posts from AI/tech communities
- Technical documentation from popular projects
- Custom URLs

### 2. Data Preprocessing
Cleans and prepares data for training:
- Text normalization
- Artifact removal
- Dataset creation in training format

### 3. Model Training
Fine-tunes pre-trained models using the collected data:
- Standard training with quality focus
- Fast training with optimized speed
- Ultra-fast training for rapid iteration
- Super-fast training for quick prototyping

### 4. Model Testing Frontend
Web interface to test your trained models:
- Select from different trained models
- Input custom prompts
- Generate and evaluate text
- Compare model outputs
- Supports various Hugging Face models
- Configurable training parameters
- Automatic model saving
- Fast training options

## Usage

Run the complete pipeline:
```bash
python main.py --all
```

Or run individual steps:
```bash
# Collect data only
python main.py --collect-data

# Preprocess data only
python main.py --preprocess-data

# Train model only
python main.py --train-model

# Train model specifically with collected data
python main.py --train-with-collected-data

# Fast train model with collected data
python main.py --fast-train

# Ultra-fast train model with collected data
python main.py --ultra-fast-train

# Super-fast train model with collected data
python main.py --super-fast-train

# Launch the web frontend for testing models
python main.py --launch-frontend
```

### Web Frontend for Testing Models

Access the web interface to test your trained models:

1. Launch the frontend: `python main.py --launch-frontend`
2. Open your browser and navigate to `http://localhost:5000`
3. Select a model (including multiple pretrained models), enter a prompt, and generate text
4. Compare outputs from different models

**Note:** Multiple pretrained models from Hugging Face are available by default and will be automatically downloaded when first used:
- DistilGPT2 (default, 82M parameters) - Fast and lightweight
- GPT-2 (117M parameters) - Original GPT-2 model
- GPT-2 Medium (345M parameters) - Better performance
- GPT-2 Large (762M parameters) - Even more capable
- GPT-2 XL (1.5B parameters) - Most powerful GPT-2 variant
- BLOOM 560M - Multilingual model
- OPT 350M - Meta's efficient transformer

For detailed instructions, see [Frontend Guide](docs/FRONTEND_GUIDE.md).

### Speed-Optimized Training

For faster training, use the direct script interfaces with additional options:

```bash
# Fast training with basic optimizations
python scripts/fast_train.py --model gpt2 --batch-size 8 --fp16 --generate-sample

# Ultra-fast training with advanced optimizations
python scripts/ultra_fast_train.py --model gpt2 --generate-sample

# Super-fast training with maximum optimizations
python scripts/super_fast_train.py --model distilgpt2 --generate-sample

# Fine-tune larger models with extreme optimizations
python scripts/ultra_fast_train.py --model facebook/opt-1.3b --epochs 0.5
```

Collect specific types of data:
```bash
# Collect news articles
python scripts/data_collection.py --source news

# Collect code samples
python scripts/data_collection.py --source code

# Collect academic papers
python scripts/data_collection.py --source academic

# Collect Reddit posts
python scripts/data_collection.py --source reddit

# Collect technical documentation
python scripts/data_collection.py --source documentation

# Collect AI/ML/DL resources
python scripts/data_collection.py --source ai_ml_dl

# Collect Hugging Face datasets
python scripts/data_collection.py --source huggingface

# Collect Kaggle datasets
python scripts/data_collection.py --source kaggle
```

Train with collected data specifically:
```bash
# Use the dedicated training script
python scripts/train_with_collected_data.py --generate-sample

# Use the fast training script
python scripts/fast_train.py --generate-sample

# Use the ultra-fast training script
python scripts/ultra_fast_train.py --generate-sample

# Use the super-fast training script
python scripts/super_fast_train.py --generate-sample

# Or use the main script
python main.py --train-with-collected-data
python main.py --fast-train
python main.py --ultra-fast-train
python main.py --super-fast-train
```

## Supported Data Sources

1. **Wikipedia** - General knowledge from random articles
2. **Project Gutenberg** - Classic literature and books
3. **Stack Exchange** - Technical Q&A from community platforms
4. **News RSS Feeds** - Current events and news articles
5. **Code Repositories** - Programming examples and documentation
6. **Academic Papers** - Research abstracts from arXiv
7. **Reddit Posts** - Discussions from AI/tech communities
8. **Technical Documentation** - Official docs from popular projects
9. **Technical Documentation** - Official docs from popular projects
10. **AI/ML/DL Resources** - Specialized resources for artificial intelligence, machine learning, and deep learning
11. **Hugging Face Datasets** - Curated datasets from the Hugging Face platform
12. **Kaggle Datasets** - Datasets from the Kaggle platform

## Speed Optimization Options

1. **Standard Training** (`--train-model`) - Full training with all features
2. **Fast Training** (`--fast-train`) - Optimized for speed with reduced sequence length and epochs
3. **Ultra-Fast Training** (`--ultra-fast-train`) - Maximum speed optimizations with minimal training
4. **Super-Fast Training** (`--super-fast-train`) - Maximum speed optimizations with minimal training on a smaller subset

## About Trillion-Parameter Models

Training models with trillions of parameters requires:
- Thousands of high-end GPUs (A100/H100 or similar)
- Several weeks to months of training time
- Specialized infrastructure and expertise
- Massive electricity costs

While this pipeline helps collect extensive training data, training trillion-parameter models is typically only feasible for large tech companies with substantial resources. For individual researchers and developers, fine-tuning smaller pre-trained models (like those with billions of parameters) is more practical and cost-effective.

## Important Notes

Building LLMs from scratch requires massive computational resources and is not feasible for individual developers. This project focuses on fine-tuning existing pre-trained models using open-source tools.

Always ensure compliance with data usage laws and ethical guidelines when collecting internet data.

Training large models requires significant GPU resources. For best results, use a machine with at least 16GB of GPU memory.