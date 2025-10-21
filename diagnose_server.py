#!/usr/bin/env python3
"""Diagnose server issues"""

import requests
import json
import time

print("="*70)
print("🔍 DIAGNOSING PURESTOCK SERVER")
print("="*70)

# Test 1: Check if server is running
print("\n1️⃣ Testing server connectivity...")
try:
    response = requests.get("http://localhost:5000/models", timeout=5)
    print(f"   ✅ Server is running (Status: {response.status_code})")
    models = response.json()
    print(f"   📊 Available models: {len(models)}")
    for model in models:
        print(f"      - {model['name']}")
except Exception as e:
    print(f"   ❌ Server not responding: {e}")
    exit(1)

# Test 2: Test generation endpoint
print("\n2️⃣ Testing text generation...")
try:
    data = {
        "model_id": "Purestock",
        "prompt": "What is Python?",
        "max_tokens": 50,
        "temperature": 0.0,
        "deterministic": True
    }
    
    print(f"   📝 Prompt: {data['prompt']}")
    print("   ⏳ Generating... (this may take 10-30 seconds)")
    
    start_time = time.time()
    response = requests.post(
        "http://localhost:5000/generate",
        json=data,
        timeout=60  # Longer timeout for generation
    )
    duration = time.time() - start_time
    
    print(f"   ⏱️ Request took {duration:.2f} seconds")
    print(f"   📡 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✅ SUCCESS!")
        print(f"   📄 Generated Text:")
        print(f"      {result['generated_texts'][0]}")
        print(f"\n   📊 Stats:")
        print(f"      - Tokens: {result.get('token_count', 'N/A')}")
        print(f"      - Words: {result.get('word_count', 'N/A')}")
        print(f"      - Generation Time: {result.get('generation_time', 'N/A')}s")
        print(f"      - Mode: {result.get('mode', 'N/A')}")
    else:
        print(f"\n   ❌ ERROR:")
        print(f"      Status: {response.status_code}")
        print(f"      Response: {response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("   ❌ Request timed out (>60s)")
except requests.exceptions.ConnectionError as e:
    print(f"   ❌ Connection error: {e}")
except Exception as e:
    print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("🏁 DIAGNOSIS COMPLETE")
print("="*70)
