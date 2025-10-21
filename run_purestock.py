#!/usr/bin/env python3
"""
Purestock AI - Simplified Web Interface
Runs without complex imports by loading model on first use
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Simple Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'purestock-ai-2025'

# Global model cache
_model = None
_tokenizer = None

def load_purestock():
    """Load Purestock model (lazy loading to avoid startup issues)"""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print("Loading Purestock model...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_path = r"E:\LLM\models\Purestock"
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        _model.to(device)
        
        print(f"✅ Purestock loaded successfully on {device.upper()}!")
        return _model, _tokenizer
        
    except Exception as e:
        print(f"❌ Error loading Purestock: {e}")
        return None, None

def generate_with_purestock(prompt, max_tokens=150, temperature=0.75):
    """Generate text using Purestock"""
    model, tokenizer = load_purestock()
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded", 0
    
    try:
        import torch
        
        # Get device
        device = next(model.parameters()).device
        
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = len(inputs['input_ids'][0])
        
        # Generate with quality settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=min(prompt_length + max_tokens, 512),
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the text
        generated_text = clean_text(generated_text, prompt)
        
        # Count tokens
        token_count = len(outputs[0])
        
        return generated_text, token_count
        
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating text: {str(e)}", 0

def clean_text(text, original_prompt):
    """Clean generated text for neat display"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Find complete sentences
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    
    # Use complete sentences if we have them
    if sentences:
        text = ' '.join(sentences)
    
    # Remove incomplete trailing sentence
    if text and text[-1] not in '.!?':
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(original_prompt):
            text = text[:last_period + 1]
    
    return text.strip()

@app.route('/')
def home():
    """Render the Purestock AI interface"""
    return render_template('index.html')

@app.route('/models')
def list_models():
    """Return Purestock model info"""
    return jsonify([{
        'id': 'Purestock',
        'name': 'Purestock (Your Custom Model)',
        'path': r'E:\LLM\models\Purestock'
    }])

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text endpoint"""
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 150))
    temperature = float(data.get('temperature', 0.75))
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Generate text
    generated_text, token_count = generate_with_purestock(
        prompt, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    
    return jsonify({
        'prompt': prompt,
        'generated_texts': [generated_text],
        'token_count': token_count,
        'model_id': 'Purestock'
    })

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache"""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    return jsonify({'status': 'success', 'message': 'Cache cleared'})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎯 PURESTOCK AI - WEB INTERFACE")
    print("=" * 60)
    print("\n✅ Starting server...")
    print("📍 URL: http://localhost:5000")
    print("🎯 Model: Purestock (2,603 samples trained)")
    print("\n💡 Model will load on first text generation")
    print("⚡ Press Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        debug=False,  # Disable debug to avoid import issues
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
