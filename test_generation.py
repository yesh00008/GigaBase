#!/usr/bin/env python3
"""Test server generation endpoint"""

import requests
import json

url = "http://localhost:5000/generate"
data = {
    "model_id": "Purestock",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.0,
    "deterministic": True
}

print("Testing generation endpoint...")
print(f"Prompt: {data['prompt']}")
print("="*60)

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Generated Text:")
        print(result['generated_texts'][0])
        print(f"\n📊 Stats:")
        print(f"  - Tokens: {result.get('token_count', 'N/A')}")
        print(f"  - Words: {result.get('word_count', 'N/A')}")
        print(f"  - Time: {result.get('generation_time', 'N/A')}s")
        print(f"  - Mode: {result.get('mode', 'N/A')}")
        print(f"  - Quality: {result.get('quality', 'N/A')}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {type(e).__name__}: {e}")
