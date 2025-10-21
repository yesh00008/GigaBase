#!/usr/bin/env python3
"""
Test script for the pretrained model

This script tests that the pretrained model loads and generates text correctly.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_utils import load_model, generate_text, get_available_models

def test_model(model_id, model_name):
    """Test a specific model"""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name}")
    print(f"{'=' * 60}")
    
    model, tokenizer = load_model(model_id)
    
    if model is None or tokenizer is None:
        print(f"❌ Failed to load {model_name}!")
        return False
    
    print(f"✅ {model_name} loaded successfully!")
    
    # Test text generation
    print(f"\n{'-' * 60}")
    print("Testing Text Generation")
    print(f"{'-' * 60}")
    
    test_prompts = [
        "Artificial intelligence is",
        "The future of technology",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        generated_texts = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=30,
            temperature=0.8
        )
        
        if generated_texts:
            print(f"Generated: {generated_texts[0][:100]}...")
        else:
            print("❌ Failed to generate text")
    
    return True

def main():
    print("=" * 60)
    print("Testing Multiple Pretrained Models")
    print("=" * 60)
    
    # Show available models
    print("\nAvailable Models:")
    models = get_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} (id: {model['id']})")
    
    # Test a few key pretrained models
    pretrained_models_to_test = [
        ("pretrained/distilgpt2", "Pretrained DistilGPT2"),
        ("pretrained/gpt2", "Pretrained GPT-2"),
        ("pretrained/bloom-560m", "Pretrained BLOOM 560M"),
    ]
    
    # Test the models
    print(f"\n{'=' * 60}")
    print("Running Tests")
    print(f"{'=' * 60}")
    
    for model_id, model_name in pretrained_models_to_test:
        try:
            success = test_model(model_id, model_name)
            if not success:
                print(f"⚠️  Skipping remaining tests due to failure")
                break
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()