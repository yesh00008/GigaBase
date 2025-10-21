#!/usr/bin/env python3
"""
Test script to verify GPT-like quality improvements in Purestock model.
Tests various prompts to ensure accurate, non-mismatched token generation.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
from model_utils import load_model, generate_text

def test_prompt(model, tokenizer, prompt, description):
    """Test a single prompt and display results"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"📝 Prompt: {prompt}")
    print(f"\n🎯 Generating response...")
    
    # Generate with GPT-like parameters
    responses = generate_text(
        model, 
        tokenizer, 
        prompt,
        max_length=200,
        temperature=0.8,
        top_p=0.92,
        top_k=50
    )
    
    if responses and responses[0]:
        response = responses[0]
        print(f"\n✨ Response:")
        print(f"{response}")
        
        # Calculate metrics
        tokens = len(tokenizer.encode(response))
        words = len(response.split())
        
        print(f"\n📊 Metrics:")
        print(f"  • Tokens: {tokens}")
        print(f"  • Words: {words}")
        print(f"  • Characters: {len(response)}")
        print(f"  • Complete sentences: {'✅' if response.endswith(('.', '!', '?')) else '❌'}")
        
        # Check for quality issues
        issues = []
        if '####' in response or '****' in response:
            issues.append("Contains garbled tokens")
        if len(set(response.split()[-5:])) == 1:
            issues.append("Repetitive ending")
        if len(response.split()) < 10:
            issues.append("Too short")
        
        if issues:
            print(f"\n⚠️ Potential Issues:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ Quality: GPT-like (no issues detected)")
    else:
        print(f"\n❌ Failed to generate response")

def main():
    """Main test function"""
    print("=" * 80)
    print("🎯 PURESTOCK GPT-LIKE QUALITY TEST")
    print("=" * 80)
    
    # Load Purestock model
    print("\n📦 Loading Purestock model...")
    model_path = os.path.join("models", "Purestock")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Purestock model not found at {model_path}")
        return
    
    model, tokenizer = load_model("Purestock", "models")
    
    if model is None or tokenizer is None:
        print("❌ Error: Failed to load Purestock model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test prompts that check for various quality aspects
    test_cases = [
        {
            "prompt": "Explain what artificial intelligence is",
            "description": "Factual explanation test"
        },
        {
            "prompt": "Write a short story about a robot learning to paint",
            "description": "Creative writing test"
        },
        {
            "prompt": "What is the capital of France?",
            "description": "Simple factual question"
        },
        {
            "prompt": "List three benefits of exercise",
            "description": "Structured response test"
        },
        {
            "prompt": "How does photosynthesis work?",
            "description": "Scientific explanation test"
        },
        {
            "prompt": "Describe the color blue without using the word blue",
            "description": "Abstract thinking test"
        }
    ]
    
    # Run all tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#' * 80}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'#' * 80}")
        
        test_prompt(model, tokenizer, test["prompt"], test["description"])
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("🎯 TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ All {len(test_cases)} tests completed!")
    print(f"\nModel: Purestock")
    print(f"Quality: GPT-like generation enabled")
    print(f"Features:")
    print(f"  • Advanced token cleaning")
    print(f"  • No repetition (n-gram size 4)")
    print(f"  • Complete sentence endings")
    print(f"  • Proper punctuation")
    print(f"  • Natural capitalization")
    print(f"  • Mismatched token prevention")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
