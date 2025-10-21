#!/usr/bin/env python3
"""Test if Purestock model can be loaded"""

import sys
import os
sys.path.insert(0, 'e:/LLM')

try:
    print("Testing model load...")
    from app import model_utils
    
    model, tokenizer = model_utils.load_model('Purestock', 'e:/LLM/models')
    
    if model is None or tokenizer is None:
        print("❌ Model or tokenizer is None!")
    else:
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Try a simple generation
        print("\nTesting generation...")
        result = model_utils.generate_text(
            model, 
            tokenizer, 
            "What is Python?",
            max_length=50,
            temperature=0.0,
            deterministic=True
        )
        print(f"Result: {result}")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
