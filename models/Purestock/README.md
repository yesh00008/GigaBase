# Purestock Model

**The One and Only Model for Your LLM Application**

## Model Information

- **Name**: Purestock
- **Version**: 1.0  
- **Base Model**: DistilGPT2
- **Training Samples**: 2,603 unique samples
- **Training Epochs**: 2
- **Training Time**: ~26 minutes
- **Training Date**: October 11, 2025

## Training Data Sources

Purestock was trained on diverse, high-quality data from 9 different datasets:

1. **C4 (Common Crawl)**: 449 samples - Web text from Common Crawl
2. **FineWeb**: 360 samples - High-quality web content
3. **TinyStories**: 751 samples - Simple, coherent stories
4. **Wikipedia**: 55 samples - Encyclopedia content
5. **FineWeb-Edu**: 202 samples - Educational web content
6. **DeepMath**: 800 samples - Mathematical reasoning and problems
7. **Nemotron-Personas**: Conversational personas data
8. **WildChat**: Diverse chat conversations
9. **Medical-O1**: Medical reasoning data

**Total**: 2,603 unique, high-quality training samples

## Model Capabilities

Purestock is a text continuation model that can:
- Generate coherent text continuations
- Complete partial sentences and paragraphs
- Produce creative writing
- Generate technical and educational content
- Handle diverse topics from math to general knowledge

## Usage

### In the Web Application

1. Start your Flask web server:
   ```bash
   python app/app.py
   ```

2. Open your browser to `http://localhost:5000`

3. Select "Purestock" from the model dropdown

4. Start chatting!

### Programmatically

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Purestock
model = AutoModelForCausalLM.from_pretrained("E:/LLM/models/Purestock")
tokenizer = AutoTokenizer.from_pretrained("E:/LLM/models/Purestock")

# Generate text
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.8, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
```

## Model Architecture

- **Type**: Causal Language Model (GPT-2 architecture)
- **Parameters**: ~82M (DistilGPT2 base)
- **Vocabulary Size**: 50,257 tokens
- **Maximum Sequence Length**: 96 tokens (training)
- **Precision**: FP32 (CPU) / FP16 (GPU)

## Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 5e-4 (with cosine decay)
- **Batch Size**: 12 per device
- **Gradient Accumulation**: Enabled
- **Warmup Steps**: 50
- **Max Gradient Norm**: 1.0

## Performance

**Sample Outputs:**

*Prompt: "Artificial intelligence is"*
> Artificial intelligence is an important field of study where knowledge and algorithms are needed to make a lasting impression on our world...

*Prompt: "Machine learning can"*
> Machine learning can be an exciting opportunity for students who develop an understanding of the nature of the world...

## Location

- **Model Path**: `E:\LLM\models\Purestock\`
- **Model Files**: 
  - `model.safetensors` - Model weights
  - `config.json` - Model configuration
  - `tokenizer.json` - Tokenizer
  - `vocab.json` - Vocabulary
  - `merges.txt` - BPE merges

## Notes

- This is a **text continuation model**, not a question-answering model
- Best used with prompts that invite text completion
- For Q&A, consider fine-tuning on instruction/conversation data
- Model performs best on topics similar to training data

## License

Based on DistilGPT2, which is licensed under Apache 2.0

---

**Purestock - Your Single, Unified Language Model**
