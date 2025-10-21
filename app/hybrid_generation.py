#!/usr/bin/env python3
"""
Hybrid Generation System
Combines model predictions with dataset examples for accurate, context-rich outputs
"""

import time
import re
from fast_search import get_search_engine

class HybridGenerator:
    """
    Advanced generation combining:
    1. Model's learned knowledge
    2. Relevant dataset examples (RAG - Retrieval Augmented Generation)
    3. Context matching for token consistency
    """
    
    def __init__(self, model, tokenizer, search_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.search_engine = search_engine or get_search_engine()
        
    def analyze_query(self, query):
        """
        Analyze user query to extract:
        - Intent
        - Key concepts
        - Expected answer type
        """
        query_lower = query.lower()
        
        # Detect question type
        intent = 'general'
        if any(q in query_lower for q in ['what is', 'what are', 'define', 'explain']):
            intent = 'definition'
        elif any(q in query_lower for q in ['how to', 'how do', 'how can']):
            intent = 'instruction'
        elif any(q in query_lower for q in ['why', 'reason']):
            intent = 'explanation'
        elif any(q in query_lower for q in ['example', 'demonstrate', 'show']):
            intent = 'example'
        
        # Extract key concepts (simple keyword extraction)
        concepts = []
        important_words = re.findall(r'\b[a-z]{3,}\b', query_lower)
        # Filter out common words
        stop_words = {'what', 'how', 'why', 'the', 'and', 'for', 'with', 'that', 'this', 'are', 'can'}
        concepts = [w for w in important_words if w not in stop_words]
        
        return {
            'intent': intent,
            'concepts': concepts[:5],  # Top 5 concepts
            'length': len(query.split())
        }
    
    def generate_hybrid(self, prompt, max_length=150, use_examples=True, 
                       num_examples=3, temperature=0.0, deterministic=True):
        """
        Generate output using hybrid approach:
        1. Search relevant examples from dataset (ultra-fast)
        2. Create enriched prompt with examples
        3. Generate with model
        4. Validate and refine output
        """
        start_time = time.perf_counter()
        
        # Step 1: Analyze query
        analysis = self.analyze_query(prompt)
        
        # Step 2: Fast dataset search (microseconds)
        context_examples = []
        search_time = 0
        
        if use_examples:
            search_start = time.perf_counter()
            search_result = self.search_engine.search(prompt, max_results=num_examples)
            search_time = search_result['search_time_us']
            
            # Get top examples
            for result in search_result['results'][:num_examples]:
                example_text = result['text'][:200]  # Limit length
                context_examples.append(example_text)
        
        # Step 3: Create enriched prompt
        if context_examples and use_examples:
            # RAG approach: Add context before question
            context_block = "\n\n".join([f"Reference {i+1}: {ex}" 
                                        for i, ex in enumerate(context_examples)])
            
            enriched_prompt = f"""Based on the following references, answer the question accurately and concisely.

{context_block}

Question: {prompt}

Answer:"""
        else:
            enriched_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Step 4: Generate with model
        from app.model_utils import generate_text
        
        gen_start = time.perf_counter()
        generated_texts = generate_text(
            self.model,
            self.tokenizer,
            enriched_prompt,
            max_length=max_length,
            temperature=temperature,
            deterministic=deterministic,
            use_system_prompt=False  # Already have context
        )
        gen_time = (time.perf_counter() - gen_start) * 1000  # ms
        
        # Step 5: Clean and validate output
        final_output = generated_texts[0] if generated_texts else ""
        
        # Remove any reference markers that might have leaked
        final_output = re.sub(r'Reference \d+:', '', final_output)
        final_output = re.sub(r'Based on the following references?', '', final_output)
        final_output = re.sub(r'Question:.*?Answer:', '', final_output, flags=re.DOTALL)
        final_output = final_output.strip()
        
        # Ensure completeness
        if final_output and not final_output[-1] in '.!?':
            # Try to find last complete sentence
            sentences = re.split(r'[.!?]+', final_output)
            if len(sentences) > 1:
                final_output = '.'.join(sentences[:-1]) + '.'
        
        total_time = (time.perf_counter() - start_time) * 1000  # ms
        
        return {
            'output': final_output,
            'metadata': {
                'intent': analysis['intent'],
                'concepts': analysis['concepts'],
                'num_examples_used': len(context_examples),
                'search_time_us': search_time,
                'generation_time_ms': gen_time,
                'total_time_ms': total_time,
                'used_rag': use_examples and len(context_examples) > 0,
                'context_examples': context_examples if use_examples else []
            }
        }
    
    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts efficiently"""
        results = []
        for prompt in prompts:
            result = self.generate_hybrid(prompt, **kwargs)
            results.append(result)
        return results


def create_hybrid_generator(model, tokenizer):
    """Factory function to create hybrid generator"""
    return HybridGenerator(model, tokenizer)
