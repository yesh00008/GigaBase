#!/usr/bin/env python3
"""Test the hybrid generation system"""

import sys
sys.path.insert(0, 'e:/LLM')

print("=" * 70)
print("TESTING HYBRID GENERATION SYSTEM")
print("=" * 70)

# Test 1: Import modules
print("\n1️⃣ Testing imports...")
try:
    from app.fast_search import FastDatasetSearch
    from app.hybrid_generation import HybridGenerator
    from app.model_utils import load_model
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load search engine
print("\n2️⃣ Testing fast search engine...")
try:
    search_engine = FastDatasetSearch(data_dir="e:/LLM/data/processed")
    search_engine.load_datasets()
    print(f"✅ Loaded {len(search_engine.examples)} examples")
except Exception as e:
    print(f"❌ Search engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test search
print("\n3️⃣ Testing ultra-fast search...")
try:
    test_queries = [
        "machine learning",
        "python programming",
        "neural networks"
    ]
    
    for query in test_queries:
        result = search_engine.search(query, max_results=3)
        print(f"\nQuery: '{query}'")
        print(f"  ⚡ Search time: {result['search_time_us']:.2f} μs")
        print(f"  📊 Results: {result['count']}")
        if result['results']:
            print(f"  📄 Top result: {result['results'][0]['text'][:80]}...")
except Exception as e:
    print(f"❌ Search failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Load model
print("\n4️⃣ Testing model loading...")
try:
    model, tokenizer = load_model('Purestock', 'e:/LLM/models')
    if model and tokenizer:
        print("✅ Model loaded successfully")
        print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("❌ Model loading returned None")
        sys.exit(1)
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test hybrid generation
print("\n5️⃣ Testing hybrid generation...")
try:
    from app.hybrid_generation import create_hybrid_generator
    
    hybrid_gen = create_hybrid_generator(model, tokenizer)
    
    test_prompt = "What is machine learning?"
    print(f"\nPrompt: {test_prompt}")
    
    result = hybrid_gen.generate_hybrid(
        test_prompt,
        max_length=100,
        use_examples=True,
        num_examples=3,
        deterministic=True
    )
    
    print(f"\n✅ HYBRID GENERATION RESULT:")
    print(f"Output: {result['output']}")
    print(f"\n📊 Metadata:")
    print(f"  - Intent: {result['metadata']['intent']}")
    print(f"  - Concepts: {', '.join(result['metadata']['concepts'])}")
    print(f"  - Examples used: {result['metadata']['num_examples_used']}")
    print(f"  - Search time: {result['metadata']['search_time_us']:.2f} μs")
    print(f"  - Generation time: {result['metadata']['generation_time_ms']:.2f} ms")
    print(f"  - Total time: {result['metadata']['total_time_ms']:.2f} ms")
    print(f"  - Used RAG: {result['metadata']['used_rag']}")
    
except Exception as e:
    print(f"❌ Hybrid generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
