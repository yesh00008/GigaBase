#!/usr/bin/env python3
"""
Test script to verify the generation fix

This script tests that the generated text doesn't include the original prompt.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_utils import load_model, generate_text

def main():
    print("=" * 60)
    print("Testing Text Generation Fix")
    print("=" * 60)
    
    # Load the pretrained model
    print("\nLoading pretrained model...")
    model, tokenizer = load_model("pretrained")
    
    if model is None or tokenizer is None:
        print("❌ Failed to load pretrained model!")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "What is Python?",
        "Explain machine learning",
        "Write a hello world program",
    ]
    
    print("\n" + "=" * 60)
    print("Testing that responses don't include the prompt")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        generated_texts = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=100,
            temperature=0.7
        )
        
        if generated_texts:
            response = generated_texts[0]
            print(f"🤖 Response: {response[:200]}...")
            
            # Check if the response starts with the prompt
            if response.strip().lower().startswith(prompt.lower()):
                print("⚠️  WARNING: Response still includes the prompt!")
            else:
                print("✅ Response correctly excludes the prompt")
        else:
            print("❌ Failed to generate text")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
