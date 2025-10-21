#!/usr/bin/env python3
"""
Test script with CORRECT prompts for DistilGPT2

This demonstrates how to use text continuation models properly.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_utils import load_model, generate_text

def main():
    print("=" * 60)
    print("Testing DistilGPT2 with CORRECT Prompts")
    print("(Text Continuation, NOT Question-Answering)")
    print("=" * 60)
    
    # Load the pretrained model
    print("\nLoading pretrained model...")
    model, tokenizer = load_model("pretrained")
    
    if model is None or tokenizer is None:
        print("❌ Failed to load pretrained model!")
        return
    
    print("✅ Model loaded successfully!")
    
    # CORRECT prompts for text continuation
    test_prompts = [
        "Python is a programming language that",
        "The benefits of machine learning include",
        "Once upon a time, there was a robot who",
        "In the future, artificial intelligence will",
        "The programmer sat down and began to",
    ]
    
    print("\n" * 60)
    print("Testing with Proper Text Continuation Prompts")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 Starting Text: '{prompt}'")
        print("-" * 60)
        
        generated_texts = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=100,
            temperature=0.8
        )
        
        if generated_texts:
            response = generated_texts[0]
            print(f"✨ Generated Text:\n{response}\n")
        else:
            print("❌ Failed to generate text")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\n💡 Notice how the model CONTINUES the text you started,")
    print("rather than trying to answer a question!")

if __name__ == "__main__":
    main()
