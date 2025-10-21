#!/usr/bin/env python3
"""Quick test to verify Flask server is responding"""

import requests
import json

def test_server():
    base_url = "http://localhost:5000"
    
    print("="*60)
    print("🧪 TESTING FLASK SERVER")
    print("="*60)
    
    # Test 1: Check if server is running
    print("\n1️⃣ Testing server connection...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Check models endpoint
    print("\n2️⃣ Testing /models endpoint...")
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint working! Found {len(models)} model(s)")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print(f"❌ Models endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Test generation (simple prompt)
    print("\n3️⃣ Testing /generate endpoint...")
    try:
        payload = {
            "model_id": "Purestock",
            "prompt": "Test",
            "max_tokens": 10,
            "temperature": 0.0,
            "deterministic": True
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generation endpoint working!")
            print(f"   Prompt: {data.get('prompt')}")
            print(f"   Response: {data.get('generated_texts', [''])[0][:50]}...")
            print(f"   Tokens: {data.get('token_count')}")
            print(f"   Time: {data.get('generation_time')}s")
        else:
            print(f"❌ Generation failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model might be loading...)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nServer is working correctly!")
    print("You can now use the web interface at: http://localhost:5000")
    return True

if __name__ == "__main__":
    import time
    
    print("\nWaiting for server to fully start...")
    time.sleep(3)
    
    test_server()
