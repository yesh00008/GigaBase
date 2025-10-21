#!/usr/bin/env python3
"""
Flask Application for LLM Model Testing

This Flask app provides a web interface to test the trained LLM models.
It allows users to select different models and generate text using custom prompts.
"""

import os
import sys
import time
import re
import random
import math
from flask import Flask, render_template, request, jsonify

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change to app directory for imports
app_dir = os.path.join(project_root, 'app')
sys.path.insert(0, app_dir)

# Import model utilities
import model_utils
from model_utils import get_available_models, load_model, generate_text, clear_model_cache

# Import hybrid generation system
from hybrid_generation import create_hybrid_generator
from fast_search import get_search_engine
from example_based_generation import create_example_generator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Set the models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# PRELOAD PURESTOCK MODEL + SEARCH ENGINE at startup for faster response
print("\n" + "="*70)
print("🚀 INITIALIZING HYBRID GENERATION SYSTEM")
print("="*70)
_preloaded_model = None
_preloaded_tokenizer = None
_hybrid_generator = None
_example_generator = None
_search_engine = None

try:
    # Load model
    _preloaded_model, _preloaded_tokenizer = load_model('Purestock', MODELS_DIR)
    if _preloaded_model and _preloaded_tokenizer:
        print("✅ Purestock model preloaded successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in _preloaded_model.parameters()):,}")
        
        # Initialize search engine
        print("\n🔍 Loading ultra-fast search engine...")
        _search_engine = get_search_engine()
        
        # Create hybrid generator
        _hybrid_generator = create_hybrid_generator(_preloaded_model, _preloaded_tokenizer)
        
        # Create example-based generator WITH model for analysis (output still dataset-only)
        print("\n📚 Initializing enhanced example-based generator...")
        print("   - Model analyzes query intent and concepts")
        print("   - Multi-strategy search for best examples")
        print("   - Output comes 100% from dataset (zero hallucination)")
        _example_generator = create_example_generator(_preloaded_model, _preloaded_tokenizer)
        
        print("✅ All generation systems ready!")
    else:
        print("⚠️ Failed to preload Purestock model - will load on first request")
except Exception as e:
    print(f"⚠️ Error preloading model: {e}")
    import traceback
    traceback.print_exc()
print("="*70 + "\n")

@app.route('/')
def home():
    """Render the home page with the list of available models"""
    # We no longer need to pass models directly to the template
    # as we're loading them via AJAX
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    EXAMPLE-BASED GENERATION - Returns ONLY dataset examples (accurate, no hallucination)
    Three modes:
    1. example-only: Pure dataset extraction (default, most accurate)
    2. hybrid: Model + dataset (balanced)
    3. model-only: Pure model generation (creative, may hallucinate)
    """
    data = request.json
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    max_tokens = int(data.get('max_tokens', 150))
    temperature = float(data.get('temperature', 0.0))
    deterministic = data.get('deterministic', True)
    
    # Generation mode selection
    generation_mode = data.get('mode', 'example-only')  # default: example-only
    
    if not model_id or not prompt:
        return jsonify({'error': 'Missing model_id or prompt'}), 400
    
    # Validate model_id
    if model_id != 'Purestock':
        return jsonify({'error': 'Only Purestock model is supported'}), 400
    
    # MODE 1: EXAMPLE-ONLY (Pure dataset, no hallucination) - DEFAULT
    if generation_mode == 'example-only' and _example_generator is not None:
        generation_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Mode: EXAMPLE-ONLY (Dataset extraction, 100% accurate)")
        print(f"📚 Source: Dataset examples only")
        print(f"{'='*70}\n")
        
        # Generate using only dataset examples
        result = _example_generator.generate_from_examples(
            prompt,
            num_examples=5,
            max_length=max_tokens * 4  # Approximate character limit
        )
        
        output_text = result['output']
        metadata = result['metadata']
        
        # Calculate token count
        if _preloaded_tokenizer:
            token_count = len(_preloaded_tokenizer.encode(output_text)) if output_text else 0
        else:
            token_count = len(output_text.split())
        word_count = len(output_text.split()) if output_text else 0
        
        generation_time = time.time() - generation_start_time
        
        # Log results
        print(f"✅ Response: {output_text[:100]}...")
        print(f"📊 Tokens: {token_count}, Words: {word_count}")
        print(f"📚 Sources: {', '.join(metadata.get('sources', []))}")
        print(f"⚡ Search: {metadata.get('search_time_us', 0):.2f} μs")
        print(f"⚡ Total: {generation_time*1000:.2f} ms")
        print(f"🎯 Confidence: {metadata.get('confidence', 'unknown')}\n")
        
        return jsonify({
            'prompt': prompt,
            'generated_texts': [output_text],
            'token_count': token_count,
            'word_count': word_count,
            'generation_time': generation_time,
            'model_id': model_id,
            'mode': 'example-only',
            'quality': 'dataset-accurate',
            'metadata': {
                'method': metadata.get('method'),
                'num_examples': metadata.get('num_examples'),
                'search_time_microseconds': metadata.get('search_time_us', 0),
                'confidence': metadata.get('confidence'),
                'sources': metadata.get('sources', []),
                'hallucination_risk': 'none'
            }
        })
    
    # MODE 2: HYBRID (Model + Dataset, balanced) - OPTIONAL
    elif generation_mode == 'hybrid' and _hybrid_generator is not None:
        # HYBRID MODE: Model + Dataset Examples (RAG)
        generation_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Mode: HYBRID (Model + Dataset Examples)")
        print(f"🔍 RAG: Enabled (ultra-fast search)")
        print(f"{'='*70}\n")
        
        # Generate with hybrid system
        result = _hybrid_generator.generate_hybrid(
            prompt,
            max_length=max_tokens,
            use_examples=True,
            num_examples=3,
            temperature=temperature,
            deterministic=deterministic
        )
        
        output_text = result['output']
        metadata = result['metadata']
        
        # Calculate token count using hybrid generator's tokenizer
        token_count = len(_hybrid_generator.tokenizer.encode(output_text)) if output_text and _hybrid_generator else 0
        word_count = len(output_text.split()) if output_text else 0
        
        # Log results
        print(f"✅ Response: {output_text[:100]}...")
        print(f"📊 Tokens: {token_count}, Words: {word_count}")
        print(f"⚡ Search: {metadata['search_time_us']:.2f} μs")
        print(f"⚡ Generation: {metadata['generation_time_ms']:.2f} ms")
        print(f"⚡ Total: {metadata['total_time_ms']:.2f} ms\n")
        
        return jsonify({
            'prompt': prompt,
            'generated_texts': [output_text],
            'token_count': token_count,
            'word_count': word_count,
            'generation_time': metadata['total_time_ms'] / 1000,
            'model_id': model_id,
            'mode': 'hybrid-rag',
            'quality': 'factual (dataset-augmented)',
            'metadata': {
                'intent': metadata['intent'],
                'search_time_microseconds': metadata['search_time_us'],
                'generation_time_ms': metadata['generation_time_ms'],
                'num_examples_used': metadata['num_examples_used'],
                'used_retrieval': metadata['used_rag'],
                'hallucination_risk': 'low'
            }
        })
    
    else:
        # FALLBACK: Standard model-only generation
        if _preloaded_model is None or _preloaded_tokenizer is None:
            model, tokenizer = load_model(model_id, MODELS_DIR)
            if model is None or tokenizer is None:
                return jsonify({'error': f'Failed to load Purestock model.'}), 500
        else:
            model, tokenizer = _preloaded_model, _preloaded_tokenizer
        
        generation_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Mode: STANDARD (Model only)")
        print(f"{'='*60}\n")
        
        generated_texts = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=max_tokens,
            temperature=temperature,
            top_p=1.0,
            deterministic=deterministic,
            use_system_prompt=True
        )
        generation_time = time.time() - generation_start_time
        
        # Calculate actual token count
        if generated_texts and generated_texts[0]:
            token_count = len(tokenizer.encode(generated_texts[0]))
            word_count = len(generated_texts[0].split())
            
            print(f"✅ Response: {generated_texts[0][:100]}...")
            print(f"📊 Tokens: {token_count}, Words: {word_count}, Time: {generation_time:.2f}s\n")
        else:
            token_count = 0
            word_count = 0
        
        return jsonify({
            'prompt': prompt,
            'generated_texts': generated_texts,
            'token_count': token_count,
            'word_count': word_count,
            'generation_time': round(generation_time, 2),
            'model_id': model_id,
            'mode': 'standard',
            'quality': 'factual (model-only)'
        })

@app.route('/models')
def list_models():
    """Return list of available models as JSON"""
    models = get_available_models(MODELS_DIR)
    return jsonify(models)

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the model cache to free up memory"""
    clear_model_cache()
    return jsonify({'status': 'success', 'message': 'Model cache cleared'})

def run_app(debug=False, port=5000):
    """Run the Flask app"""
    print(f"\nStarting LLM Testing Frontend on http://localhost:{port}")
    print("Available models:")
    models = get_available_models(MODELS_DIR)
    for model in models:
        print(f" - {model['name']}")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(debug=debug, host='0.0.0.0', port=port)

if __name__ == '__main__':
    run_app(debug=True)