# Using Pretrained Models

This project now supports multiple pretrained models from Hugging Face that can be used directly without any training.

## Available Pretrained Models

We currently support 7 pretrained models from Hugging Face:

1. **DistilGPT2 (Default)** - `pretrained/distilgpt2`
   - Fast, lightweight distilled version of GPT-2
   - 82M parameters
   - Good for quick testing and prototyping

2. **GPT-2** - `pretrained/gpt2`
   - Original GPT-2 model
   - 117M parameters
   - Better quality than DistilGPT2

3. **GPT-2 Medium** - `pretrained/gpt2-medium`
   - Larger GPT-2 model
   - 345M parameters
   - Better performance than base GPT-2

4. **GPT-2 Large** - `pretrained/gpt2-large`
   - Even larger GPT-2 model
   - 762M parameters
   - High-quality text generation

5. **GPT-2 XL** - `pretrained/gpt2-xl`
   - Largest GPT-2 model
   - 1.5B parameters
   - Best quality but requires more resources

6. **BLOOM 560M** - `pretrained/bloom-560m`
   - Multilingual model trained on 46 languages
   - 560M parameters
   - Good for multilingual tasks

7. **OPT 350M** - `pretrained/opt-350m`
   - Meta's Open Pretrained Transformer
   - 350M parameters
   - Efficient model with good performance

## How to Use

### Web Interface

1. Launch the web frontend:
   ```bash
   python main.py --launch-frontend
   ```

2. Open your browser and go to `http://localhost:5000`

3. Select any pretrained model from the list (they appear first)

4. Enter your prompt and click "Generate Text"

### Python Script

```python
from app.model_utils import load_model, generate_text

# Load any pretrained model
model, tokenizer = load_model("pretrained/gpt2-medium")  # or any other model ID

# Generate text
prompt = "Artificial intelligence is"
generated_texts = generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_length=100,
    temperature=0.8
)

print(generated_texts[0])
```

### Command Line

```bash
# List all available models
python list_pretrained_models.py

# Test multiple pretrained models
python test_pretrained.py
```

## Model Selection Guide

| Model | Parameters | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| DistilGPT2 | 82M | Fast | Good | Quick testing, low-resource environments |
| GPT-2 | 117M | Medium | Better | General purpose, balanced performance |
| GPT-2 Medium | 345M | Slower | Good | Better quality text generation |
| GPT-2 Large | 762M | Slow | Very Good | High-quality generation |
| GPT-2 XL | 1.5B | Very Slow | Excellent | Best quality, resource intensive |
| BLOOM 560M | 560M | Slow | Good | Multilingual tasks |
| OPT 350M | 350M | Medium | Good | Efficient, balanced performance |

## First-Time Usage

When you first use any pretrained model, it will be automatically downloaded from Hugging Face. This requires:
- Internet connection
- Sufficient disk space (100MB - 3GB depending on model)
- Time for download (varies by model size and connection speed)

## Resource Requirements

| Model | Disk Space | RAM | GPU Memory | Notes |
|-------|------------|-----|------------|-------|
| DistilGPT2 | ~300MB | 1GB | 2GB | Works on CPU |
| GPT-2 | ~500MB | 2GB | 2GB | Works on CPU |
| GPT-2 Medium | ~1.3GB | 3GB | 4GB | Better with GPU |
| GPT-2 Large | ~3GB | 5GB | 8GB | Needs GPU |
| GPT-2 XL | ~6GB | 8GB | 12GB | Needs powerful GPU |
| BLOOM 560M | ~2.2GB | 3GB | 4GB | Better with GPU |
| OPT 350M | ~1.3GB | 3GB | 4GB | Better with GPU |

## Troubleshooting

### Download Issues
If models fail to download:
1. Check your internet connection
2. Try again (temporary network issues)
3. Check disk space availability
4. Use a VPN if Hugging Face is blocked in your region

### Memory Issues
If you encounter memory errors:
1. Use a smaller model (DistilGPT2 or GPT-2)
2. Close other applications to free up RAM
3. Use CPU instead of GPU (slower but works with less memory)

### Performance Issues
To improve generation speed:
1. Use smaller models
2. Reduce max_length parameter
3. Use GPU if available
4. Lower the temperature parameter