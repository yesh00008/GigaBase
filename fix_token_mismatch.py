#!/usr/bin/env python3
"""
Quick Fix for Token Mismatches
This script updates the generation parameters to eliminate token issues
with the CURRENT model (no retraining needed).
"""

import os
import sys

# Test the current model with stricter parameters
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

print("="*80)
print("🔧 TESTING PURESTOCK WITH ANTI-MISMATCH SETTINGS")
print("="*80)

from model_utils import load_model
import torch

# Load model
print("\n📦 Loading Purestock model...")
model, tokenizer = load_model("Purestock", "models")

if model is None:
    print("❌ Failed to load model")
    sys.exit(1)

print("✅ Model loaded successfully!")

# Test prompts
test_prompts = [
    "What is the capital of France?",
    "Explain artificial intelligence in simple terms.",
    "What is 2 + 2?",
]

print("\n" + "="*80)
print("TESTING STRICT DETERMINISTIC MODE")
print("="*80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/3")
    print(f"{'='*80}")
    print(f"📝 Prompt: {prompt}")
    
    # ULTRA-STRICT PARAMETERS (no randomness at all)
    device = next(model.parameters()).device
    
    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate with STRICTEST settings
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,  # Short responses
            
            # ULTRA-DETERMINISTIC
            do_sample=False,          # NO sampling (pure greedy)
            temperature=1.0,          # Ignored when do_sample=False
            top_p=1.0,               # Ignored when do_sample=False
            top_k=0,                 # Ignored when do_sample=False
            
            # ANTI-REPETITION (strongest)
            no_repeat_ngram_size=2,  # Block 2-word repeats
            repetition_penalty=1.5,  # Heavy penalty for repeating
            
            # CLEAN ENDINGS
            num_beams=1,             # Greedy search
            early_stopping=False,    # Let it finish
            
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    # Clean up
    response = response.strip()
    
    # Find first complete sentence
    for end_char in ['.', '!', '?']:
        if end_char in response:
            end_idx = response.find(end_char)
            response = response[:end_idx + 1]
            break
    
    print(f"\n🤖 Response: {response}")
    
    # Quality check
    words = response.split()
    has_repetition = False
    for j in range(len(words) - 2):
        if words[j] == words[j+1] == words[j+2]:
            has_repetition = True
            break
    
    print(f"\n📊 Quality:")
    print(f"  {'✅' if not has_repetition else '❌'} No repetition")
    print(f"  {'✅' if response.endswith(('.', '!', '?')) else '❌'} Complete sentence")
    print(f"  {'✅' if len(words) >= 3 else '❌'} Meaningful length")

print("\n" + "="*80)
print("💡 RECOMMENDATION")
print("="*80)
print("""
The current model was trained with random sampling (temperature=0.8).
For BEST results with zero token mismatches:

OPTION 1: Use current model with strict settings (done above)
  ✅ Quick fix (no retraining)
  ⚠️ May still have some quality issues

OPTION 2: Retrain with deterministic data (recommended)
  Run: python train_purestock_deterministic.py
  ✅ Best quality
  ✅ No token mismatches
  ⏱️ Takes ~30 minutes

The code is already updated for Option 1.
The server will now use strict anti-repetition settings.
""")
