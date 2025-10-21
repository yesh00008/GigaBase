#!/usr/bin/env python3
"""
Purestock Deterministic Quality Test Suite
Tests factual accuracy, anti-hallucination, and deterministic behavior.

This script validates:
1. Deterministic outputs (same input = same output)
2. No hallucinations (factual responses only)
3. Quality validation (complete sentences, no garbage)
4. Few-shot learning capability
"""

import os
import sys
import json

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
from model_utils import load_model, generate_text

def test_deterministic_behavior(model, tokenizer):
    """Test that same input produces same output (deterministic)"""
    print("\n" + "="*80)
    print("TEST 1: DETERMINISTIC BEHAVIOR")
    print("="*80)
    
    prompt = "What is the capital of France?"
    
    print(f"📝 Prompt: {prompt}")
    print(f"🔄 Running 3 times (should be identical)...\n")
    
    responses = []
    for i in range(3):
        response = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_length=50,
            temperature=0.0,  # Deterministic
            deterministic=True
        )[0]
        responses.append(response)
        print(f"Run {i+1}: {response}")
    
    # Check if all identical
    if len(set(responses)) == 1:
        print(f"\n✅ PASS: All responses identical (deterministic)")
        return True
    else:
        print(f"\n❌ FAIL: Responses differ (non-deterministic)")
        return False

def test_no_hallucination(model, tokenizer):
    """Test that model refuses to answer when unsure"""
    print("\n" + "="*80)
    print("TEST 2: NO HALLUCINATION")
    print("="*80)
    
    # Impossible/nonsense questions
    impossible_prompts = [
        "What is the color of number 7?",
        "How many stars are in the word 'hello'?",
        "What did Abraham Lincoln tweet yesterday?",
    ]
    
    results = []
    for prompt in impossible_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        response = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=100,
            temperature=0.0,
            deterministic=True,
            use_system_prompt=True
        )[0]
        
        print(f"🤖 Response: {response}")
        
        # Check if model refuses or says "I don't know"
        refusal_phrases = ["don't know", "unsure", "unclear", "cannot", "unable", "not sure"]
        has_refusal = any(phrase in response.lower() for phrase in refusal_phrases)
        
        if has_refusal:
            print(f"✅ Correctly refused to hallucinate")
            results.append(True)
        else:
            print(f"⚠️ May have hallucinated")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Refusal rate: {success_rate:.0f}%")
    return success_rate >= 50  # At least 50% should refuse

def test_quality_validation(model, tokenizer):
    """Test output quality (no garbage, complete sentences)"""
    print("\n" + "="*80)
    print("TEST 3: OUTPUT QUALITY")
    print("="*80)
    
    prompts = [
        "Explain photosynthesis",
        "What is artificial intelligence?",
        "Describe the solar system",
    ]
    
    results = []
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        response = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=150,
            temperature=0.0,
            deterministic=True
        )[0]
        
        print(f"🤖 Response: {response}")
        
        # Quality checks
        checks = {
            "No garbage tokens": not any(x in response for x in ['####', '****', '[[[[', ']]]]']),
            "Complete sentence": response.rstrip().endswith(('.', '!', '?')),
            "Minimum length": len(response.split()) >= 10,
            "No excessive repetition": check_no_repetition(response),
            "Printable characters": sum(c.isprintable() or c.isspace() for c in response) / len(response) > 0.95,
        }
        
        print(f"\n📊 Quality Checks:")
        for check, passed in checks.items():
            print(f"  {'✅' if passed else '❌'} {check}")
        
        all_passed = all(checks.values())
        results.append(all_passed)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Quality pass rate: {success_rate:.0f}%")
    return success_rate >= 80

def check_no_repetition(text):
    """Check for excessive word repetition"""
    words = text.split()
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return False
    return True

def test_few_shot_learning(model, tokenizer):
    """Test few-shot learning with examples in prompt"""
    print("\n" + "="*80)
    print("TEST 4: FEW-SHOT LEARNING")
    print("="*80)
    
    # Few-shot prompt with examples
    few_shot_prompt = """Answer the following questions concisely and factually.

Example 1:
Q: What is the capital of England?
A: London.

Example 2:
Q: What is 2 + 2?
A: 4.

Example 3:
Q: What color is the sky?
A: Blue.

Now answer this question:
Q: What is the capital of Japan?
A:"""
    
    print(f"📝 Few-shot prompt (with 3 examples)")
    print(f"🎯 Expected: Short factual answer about Tokyo")
    
    response = generate_text(
        model,
        tokenizer,
        few_shot_prompt,
        max_length=50,
        temperature=0.0,
        deterministic=True,
        use_system_prompt=False  # Already has structure
    )[0]
    
    print(f"\n🤖 Response: {response}")
    
    # Check if follows pattern (short, factual)
    is_short = len(response.split()) <= 10
    mentions_tokyo = 'tokyo' in response.lower()
    
    print(f"\n📊 Few-shot checks:")
    print(f"  {'✅' if is_short else '❌'} Short answer (<=10 words)")
    print(f"  {'✅' if mentions_tokyo else '⚠️'} Mentions Tokyo")
    
    return is_short

def test_factual_accuracy(model, tokenizer):
    """Test factual accuracy with known facts"""
    print("\n" + "="*80)
    print("TEST 5: FACTUAL ACCURACY")
    print("="*80)
    
    # Known facts
    factual_tests = [
        {
            "prompt": "What is the capital of France?",
            "expected_keywords": ["paris"],
            "description": "Geography fact"
        },
        {
            "prompt": "How many days are in a week?",
            "expected_keywords": ["7", "seven"],
            "description": "Basic knowledge"
        },
        {
            "prompt": "What is water made of?",
            "expected_keywords": ["hydrogen", "oxygen", "h2o"],
            "description": "Science fact"
        },
    ]
    
    results = []
    for test in factual_tests:
        print(f"\n📝 {test['description']}")
        print(f"   Prompt: {test['prompt']}")
        
        response = generate_text(
            model,
            tokenizer,
            test['prompt'],
            max_length=100,
            temperature=0.0,
            deterministic=True
        )[0]
        
        print(f"🤖 Response: {response}")
        
        # Check for expected keywords
        response_lower = response.lower()
        has_keyword = any(keyword in response_lower for keyword in test['expected_keywords'])
        
        print(f"   {'✅' if has_keyword else '❌'} Contains expected keyword")
        results.append(has_keyword)
    
    accuracy = sum(results) / len(results) * 100
    print(f"\n📊 Factual accuracy: {accuracy:.0f}%")
    return accuracy >= 60  # At least 60% correct

def run_all_tests():
    """Run complete test suite"""
    print("="*80)
    print("🎯 PURESTOCK DETERMINISTIC QUALITY TEST SUITE")
    print("="*80)
    
    # Load model
    print("\n📦 Loading Purestock model...")
    model_path = os.path.join("models", "Purestock")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Purestock model not found at {model_path}")
        print(f"   Please train the model first: python train_purestock_deterministic.py")
        return
    
    model, tokenizer = load_model("Purestock", "models")
    
    if model is None or tokenizer is None:
        print("❌ Error: Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Run tests
    test_results = {}
    
    try:
        test_results['deterministic'] = test_deterministic_behavior(model, tokenizer)
        test_results['no_hallucination'] = test_no_hallucination(model, tokenizer)
        test_results['quality'] = test_quality_validation(model, tokenizer)
        test_results['few_shot'] = test_few_shot_learning(model, tokenizer)
        test_results['factual'] = test_factual_accuracy(model, tokenizer)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} | {status}")
    
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = total_passed / total_tests * 100
    
    print("="*80)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({success_rate:.0f}%)")
    print("="*80)
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT: Model meets quality standards!")
    elif success_rate >= 60:
        print("\n✅ GOOD: Model performs adequately")
    else:
        print("\n⚠️ NEEDS IMPROVEMENT: Consider retraining with:")
        print("   - More epochs")
        print("   - Lower learning rate")
        print("   - More balanced data")
        print("   - Better few-shot examples")
    
    # Save results
    results_file = "test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "results": test_results,
            "success_rate": success_rate,
            "total_passed": total_passed,
            "total_tests": total_tests
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
