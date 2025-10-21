# ChatGPT Interface - Usage Guide

## How to Use

The interface is designed to be simple and intuitive - just like ChatGPT!

### Getting Started

1. **Open the Interface**: Navigate to http://localhost:5000 in your browser
2. **Select a Model**: Choose any model from the sidebar (Pretrained DistilGPT2 is selected by default)
3. **Ask Your Question**: Type anything in the input box at the bottom
4. **Get Your Answer**: Press Enter or click the send button

### Available Models

The interface now supports multiple pretrained models from Hugging Face:

1. **Pretrained DistilGPT2 (Default)** - Fast, lightweight model (82M parameters)
2. **Pretrained GPT-2** - Original GPT-2 model (117M parameters)
3. **Pretrained GPT-2 Medium** - Better performance model (345M parameters)
4. **Pretrained GPT-2 Large** - High-quality model (762M parameters)
5. **Pretrained GPT-2 XL** - Most powerful GPT-2 variant (1.5B parameters)
6. **Pretrained BLOOM 560M** - Multilingual model (560M parameters)
7. **Pretrained OPT 350M** - Meta's efficient transformer (350M parameters)

You can also use any custom trained models you've placed in the `models/` directory.

### Example Questions You Can Ask

#### General Questions
- "What is artificial intelligence?"
- "Explain machine learning to me"
- "Tell me about neural networks"
- "What is the difference between AI and ML?"

#### Technical Questions
- "How does backpropagation work?"
- "What are transformers in deep learning?"
- "Explain gradient descent"
- "What is a convolutional neural network?"

#### Code-Related Questions
- "Write a Python function to calculate factorial"
- "Show me how to create a list in Python"
- "What is object-oriented programming?"
- "Explain recursion with an example"

#### Creative Prompts
- "Write a short story about robots"
- "Create a poem about technology"
- "Describe the future of AI"
- "What would happen if..."

#### Open-Ended
- "Tell me something interesting"
- "What should I know about programming?"
- "Explain quantum computing"
- "What are the benefits of learning Python?"

### Tips for Better Responses

1. **Be Specific**: Instead of "Tell me about AI", try "Explain how AI is used in healthcare"
2. **Ask Follow-ups**: Continue the conversation by asking related questions
3. **Provide Context**: Give background information if needed
4. **Keep It Clear**: Simple, clear questions get better responses
5. **Choose the Right Model**: 
   - Use DistilGPT2 for fast responses
   - Use GPT-2 Medium/Large for better quality
   - Use GPT-2 XL for highest quality (requires more resources)
   - Use BLOOM for multilingual tasks

### Keyboard Shortcuts

- **Enter**: Send your message
- **Shift + Enter**: Add a new line without sending

### Features

✅ **Conversation Flow**: Your questions and the AI's answers appear in a chat format
✅ **Auto-Scroll**: The chat automatically scrolls to show the latest message
✅ **Loading Indicator**: See animated dots while the AI is thinking
✅ **Model Selection**: Choose different models from the sidebar
✅ **Responsive Design**: Works on desktop and mobile browsers
✅ **Multiple Model Support**: Access to 7 different pretrained models

### What Happens Behind the Scenes

1. You type your question
2. The question is sent to the selected model
3. The model processes your question and generates a response
4. The response appears in the chat as an answer

### Model Information

#### DistilGPT2 (Default)
- **Type**: Causal Language Model
- **Size**: 82M parameters
- **Strength**: Fast text generation, good for general knowledge
- **Best For**: General questions, creative writing, basic explanations

#### GPT-2 Series
- **GPT-2**: 117M parameters - Better quality than DistilGPT2
- **GPT-2 Medium**: 345M parameters - Good balance of quality and speed
- **GPT-2 Large**: 762M parameters - High-quality text generation
- **GPT-2 XL**: 1.5B parameters - Best quality, requires more resources

#### Multilingual Models
- **BLOOM 560M**: Trained on 46 languages, good for multilingual tasks

#### Efficient Models
- **OPT 350M**: Meta's efficient transformer model

### Troubleshooting

**Nothing happens when I click send:**
- Make sure you've typed something in the input box
- Check that a model is selected in the sidebar

**The response seems slow:**
- Larger models take longer to generate responses
- The model runs on CPU by default, which can take a few seconds
- First-time model loading may take longer (model downloads from Hugging Face)

**The response doesn't make sense:**
- Try rephrasing your question
- Be more specific with your request
- Remember: the model generates text based on patterns, not true understanding
- Try a different model for variety

**Model download fails:**
- Check your internet connection
- Ensure you have sufficient disk space
- Try again (temporary network issues)

### Example Conversation

**You**: What is Python?

**ChatGPT**: Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991...

**You**: Can you show me a simple Python program?

**ChatGPT**: Here's a simple Python program that prints "Hello, World!" to the console:
```python
print("Hello, World!")
```

---

## Ready to Start?

Just open http://localhost:5000 and start asking questions! 🚀

Try different models to see how they compare!