# Pretrained Model Setup Guide

## Overview

The LLM project now supports using multiple pretrained models directly from Hugging Face for text generation. By default, the **DistilGPT2** model is available as the pretrained option, but we've added several more powerful models.

## What Changed?

### 1. **Model Loading** (`app/model_utils.py`)
- Added support for loading multiple pretrained models from Hugging Face
- When selecting "pretrained" models, the system automatically downloads and uses the selected model
- No manual model file downloads required - everything is handled automatically

### 2. **Available Models** (`app/model_utils.py`)
The pretrained models now available include:
- **Pretrained DistilGPT2 (Default)** - Fast, lightweight model (82M parameters)
- **Pretrained GPT-2** - Original GPT-2 model (117M parameters)
- **Pretrained GPT-2 Medium** - Larger GPT-2 model (345M parameters)
- **Pretrained GPT-2 Large** - Even larger GPT-2 model (762M parameters)
- **Pretrained GPT-2 XL** - Largest GPT-2 model (1.5B parameters)
- **Pretrained BLOOM 560M** - Multilingual model (560M parameters)
- **Pretrained OPT 350M** - Meta's Open Pretrained Transformer (350M parameters)

All pretrained models now appear first in the model selection list, ordered by size and capability.

### 3. **Web Frontend** (`app/app.py`)
- Updated to properly handle all pretrained model generation
- Displays token count and generation time for all models
- Simplified error handling and response formatting

## How to Use the Pretrained Models

### Option 1: Web Interface
```bash
# Launch the web frontend
python main.py --launch-frontend

# Then:
# 1. Open http://localhost:5000 in your browser
# 2. Select any pretrained model from the dropdown (they appear first)
# 3. Enter your prompt
# 4. Click "Generate Text"
```

### Option 2: Python Script
```python
from app.model_utils import load_model, generate_text

# Load any of the pretrained models
model, tokenizer = load_model("pretrained/gpt2-medium")  # or any other model ID

# Generate text
prompt = "Artificial intelligence is"
generated_texts = generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_length=50,
    temperature=0.8
)

print(generated_texts[0])
```

### Option 3: Test Script
```bash
# Run the test script to test multiple models
python test_pretrained.py
```

## Model Details

### GPT-2 Series
- **GPT-2** (117M parameters) - The original GPT-2 model
- **GPT-2 Medium** (345M parameters) - Better performance than base GPT-2
- **GPT-2 Large** (762M parameters) - Even more capable with larger parameter count
- **GPT-2 XL** (1.5B parameters) - Most powerful GPT-2 variant

### Multilingual Models
- **BLOOM 560M** - Trained on 46 languages, good for multilingual tasks

### Efficient Models
- **DistilGPT2** (82M parameters) - Distilled version of GPT-2, 2x faster with 97% performance
- **OPT 350M** - Meta's efficient transformer model

## Recommendations

1. **For quick testing**: Use DistilGPT2 (fastest, least resource intensive)
2. **For better quality**: Use GPT-2 Medium or Large
3. **For maximum capability**: Use GPT-2 XL (requires more memory)
4. **For multilingual tasks**: Use BLOOM 560M
5. **For efficiency**: Use OPT 350M

## Advantages

1. **No Setup Required**: All models download automatically on first use
2. **Multiple Options**: Choose the right model for your needs
3. **Low Resource Usage**: DistilGPT2 works well on CPU, larger models work on GPU
4. **Good Baseline**: Great for comparing with your fine-tuned models
5. **Always Available**: No need to train a model first to test the system

## Custom Models

You can still use your own fine-tuned models:
- Place them in the `models/` directory (e.g., `models/fine_tuned/`, `models/fast_trained/`)
- They will appear in the model selection alongside the pretrained options
- Must contain model weights (`.safetensors` or `.bin` files) and `config.json`

## Troubleshooting

### Model Download Issues
If any model fails to download:
```bash
# Try downloading manually using transformers CLI
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('MODEL_NAME'); AutoTokenizer.from_pretrained('MODEL_NAME')"
```

Replace `MODEL_NAME` with the actual model name (e.g., `gpt2-medium`, `bigscience/bloom-560m`).

### Memory Issues
If you encounter memory errors:
- The system automatically uses CPU if CUDA is not available
- On GPU, it uses FP16 for memory efficiency
- Close other applications to free up RAM
- Use smaller models like DistilGPT2 or GPT-2 for low-memory systems

### Generation Quality
To improve generation quality:
- Adjust temperature (0.7-0.9 for balanced output)
- Increase max_length for longer responses
- Use different prompts with more context

## Next Steps

1. **Test different pretrained models** to understand their capabilities
2. **Fine-tune on your data** using the training scripts
3. **Compare outputs** between pretrained and fine-tuned models
4. **Iterate and improve** based on the comparison

## Technical Implementation

The implementation handles pretrained models by:
1. Detecting "pretrained/*" in the model_id
2. Loading from Hugging Face instead of local directory
3. Caching the model for subsequent uses
4. Using the same generation interface as custom models

This ensures a seamless experience whether using pretrained or custom models.