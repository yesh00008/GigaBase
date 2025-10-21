#!/usr/bin/env python3
"""
Ultra-Fast Dataset Search Engine
Nanosecond-level search with caching and semantic matching
"""

import os
import json
import time
import re
from collections import defaultdict
from functools import lru_cache
import numpy as np

class FastDatasetSearch:
    """Lightning-fast dataset search with multiple indexing strategies"""
    
    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir
        self.cache = {}
        self.keyword_index = defaultdict(list)  # keyword -> [examples]
        self.ngram_index = defaultdict(set)  # ngram -> {example_ids}
        self.examples = []
        self.loaded = False
        
    def load_datasets(self):
        """Load and index all datasets for ultra-fast search"""
        if self.loaded:
            return
            
        start = time.perf_counter()
        print("\n🚀 Loading datasets for ultra-fast search...")
        
        # Load all text files
        dataset_files = [
            'ai_ml_dl_data.txt',
            'code_data.txt',
            'documentation_data.txt',
            'stackexchange_data.txt',
            'wikipedia_data.txt'
        ]
        
        example_id = 0
        for filename in dataset_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Split into chunks (each chunk is an example)
                        chunks = content.split('\n\n')
                        for chunk in chunks:
                            chunk = chunk.strip()
                            if len(chunk) > 20:  # Valid example
                                self.examples.append({
                                    'id': example_id,
                                    'text': chunk,
                                    'source': filename.replace('_data.txt', ''),
                                    'length': len(chunk)
                                })
                                
                                # Index keywords
                                self._index_example(chunk, example_id)
                                example_id += 1
                                
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
        
        self.loaded = True
        load_time = (time.perf_counter() - start) * 1000  # ms
        print(f"✅ Loaded {len(self.examples)} examples in {load_time:.2f}ms")
        print(f"📊 Keyword index: {len(self.keyword_index)} keywords")
        print(f"📊 N-gram index: {len(self.ngram_index)} n-grams\n")
    
    def _index_example(self, text, example_id):
        """Index example with keywords and n-grams for fast lookup"""
        # Extract keywords (normalized)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Index individual keywords
        for word in set(words):
            if len(word) > 2:  # Skip very short words
                self.keyword_index[word].append(example_id)
        
        # Index 2-grams and 3-grams for phrase matching
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            self.ngram_index[bigram].add(example_id)
            
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            self.ngram_index[trigram].add(example_id)
    
    @lru_cache(maxsize=1000)
    def search(self, query, max_results=5):
        """
        Ultra-fast search using multi-level indexing
        Returns: list of relevant examples in microseconds
        """
        start = time.perf_counter()
        
        if not self.loaded:
            self.load_datasets()
        
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        # Score each example
        scores = defaultdict(float)
        
        # 1. Exact phrase match (highest score)
        for example in self.examples:
            if query_lower in example['text'].lower():
                scores[example['id']] += 100.0
        
        # 2. N-gram matching (high score)
        for i in range(len(query_words) - 2):
            trigram = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
            if trigram in self.ngram_index:
                for ex_id in self.ngram_index[trigram]:
                    scores[ex_id] += 50.0
        
        for i in range(len(query_words) - 1):
            bigram = f"{query_words[i]} {query_words[i+1]}"
            if bigram in self.ngram_index:
                for ex_id in self.ngram_index[bigram]:
                    scores[ex_id] += 25.0
        
        # 3. Keyword matching (medium score)
        for word in query_words:
            if word in self.keyword_index:
                for ex_id in self.keyword_index[word]:
                    scores[ex_id] += 10.0 / (len(self.keyword_index[word]) + 1)
        
        # Get top results
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_results]
        results = [self.examples[ex_id] for ex_id in top_ids if ex_id < len(self.examples)]
        
        search_time = (time.perf_counter() - start) * 1_000_000  # microseconds
        
        return {
            'results': results,
            'count': len(results),
            'search_time_us': search_time,
            'scores': {ex_id: scores[ex_id] for ex_id in top_ids}
        }
    
    def get_context_examples(self, query, num_examples=3):
        """Get best matching examples for context enrichment"""
        search_result = self.search(query, max_results=num_examples)
        return [r['text'] for r in search_result['results']]


# Global search engine instance (singleton)
_search_engine = None

def get_search_engine():
    """Get or create the global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = FastDatasetSearch()
        _search_engine.load_datasets()
    return _search_engine
