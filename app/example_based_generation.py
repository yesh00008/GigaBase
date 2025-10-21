#!/usr/bin/env python3
"""
Enhanced Example-Based Generator - Model-Analyzed, Dataset-Only Output
- Model analyzes query to understand intent and extract key concepts
- Intelligent refinement of example selection
- Output comes ONLY from dataset (zero hallucination)
"""

import re
import time
from fast_search import get_search_engine

class ExampleBasedGenerator:
    """
    Advanced example-based generation:
    1. Model analyzes the query (intent, keywords, answer type)
    2. Intelligent multi-strategy search
    3. Smart ranking and filtering of examples
    4. Refined extraction and combination
    5. Output is STILL 100% from dataset (no model generation)
    """
    
    def __init__(self, model=None, tokenizer=None, search_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.search_engine = search_engine or get_search_engine()
        self.use_model_analysis = (model is not None and tokenizer is not None)
    
    def analyze_query_with_model(self, query):
        """
        Use model to analyze query and extract:
        - Intent (question type, expected answer format)
        - Key concepts and entities
        - Search strategy recommendations
        """
        if not self.use_model_analysis:
            return self.analyze_query_simple(query)
        
        # Use model to understand the query better
        analysis_prompt = f"""Analyze this query:
Query: {query}

Extract:
1. Intent (definition/howto/example/explanation):
2. Main concepts:
3. Answer format needed:"""
        
        try:
            inputs = self.tokenizer(analysis_prompt, return_tensors="pt", truncation=True, max_length=150)
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                temperature=0.3,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the analysis
            analysis = self._parse_analysis(analysis_text, query)
            print(f"🧠 Model Analysis: Intent={analysis['intent']}, Concepts={analysis['concepts'][:3]}")
            return analysis
        except Exception as e:
            print(f"⚠️  Model analysis failed, using rule-based: {e}")
            return self.analyze_query_simple(query)
    
    def analyze_query_simple(self, query):
        """
        Simple rule-based query analysis (fallback when model not available)
        """
        query_lower = query.lower()
        
        # Detect intent
        intent = 'general'
        if any(q in query_lower for q in ['what is', 'what are', 'define', 'meaning of']):
            intent = 'definition'
        elif any(q in query_lower for q in ['how to', 'how do', 'how can', 'steps to']):
            intent = 'howto'
        elif any(q in query_lower for q in ['why', 'reason', 'because']):
            intent = 'explanation'
        elif any(q in query_lower for q in ['example', 'demonstrate', 'show me', 'sample']):
            intent = 'example'
        elif any(q in query_lower for q in ['compare', 'difference', 'vs', 'versus']):
            intent = 'comparison'
        elif any(q in query_lower for q in ['list', 'types of', 'kinds of']):
            intent = 'list'
        
        # Extract key concepts (important words)
        concepts = []
        words = re.findall(r'\b[a-z]{3,}\b', query_lower)
        stop_words = {'what', 'how', 'why', 'the', 'and', 'for', 'with', 'that', 'this', 
                     'are', 'can', 'will', 'should', 'would', 'could', 'does', 'from'}
        concepts = [w for w in words if w not in stop_words]
        
        # Determine answer type
        answer_type = 'detailed'
        if len(query.split()) <= 5:
            answer_type = 'short'
        elif 'code' in query_lower or 'function' in query_lower or 'program' in query_lower:
            answer_type = 'code'
        elif intent == 'list':
            answer_type = 'list'
        
        print(f"🔍 Rule-based Analysis: Intent={intent}, Concepts={concepts[:3]}")
        
        return {
            'intent': intent,
            'concepts': concepts[:5],
            'answer_type': answer_type,
            'query_complexity': 'simple' if len(query.split()) <= 7 else 'complex'
        }
    
    def _parse_analysis(self, analysis_text, original_query):
        """Parse model's analysis output into structured format"""
        lines = analysis_text.split('\n')
        
        intent = 'general'
        concepts = []
        answer_type = 'detailed'
        
        for line in lines:
            line_lower = line.lower()
            if 'intent' in line_lower or '1.' in line:
                if 'definition' in line_lower:
                    intent = 'definition'
                elif 'howto' in line_lower or 'how' in line_lower:
                    intent = 'howto'
                elif 'example' in line_lower:
                    intent = 'example'
                elif 'explanation' in line_lower:
                    intent = 'explanation'
            elif 'concept' in line_lower or '2.' in line:
                concepts_text = line.split(':', 1)[-1] if ':' in line else line
                concepts = re.findall(r'\b[a-z]{3,}\b', concepts_text.lower())
            elif 'format' in line_lower or '3.' in line:
                if 'short' in line_lower or 'brief' in line_lower:
                    answer_type = 'short'
                elif 'code' in line_lower:
                    answer_type = 'code'
                elif 'list' in line_lower:
                    answer_type = 'list'
        
        # Fallback: extract from query if model didn't help
        if not concepts:
            concepts = re.findall(r'\b[a-z]{4,}\b', original_query.lower())
        
        return {
            'intent': intent,
            'concepts': concepts[:5],
            'answer_type': answer_type,
            'query_complexity': 'complex'
        }
    
    def multi_strategy_search(self, query, analysis, num_examples=5):
        """
        Use multiple search strategies based on query analysis
        """
        all_results = []
        scores = {}
        
        # Strategy 1: Direct query search (highest weight)
        print(f"🔎 Strategy 1: Direct search for '{query[:50]}...'")
        result1 = self.search_engine.search(query, max_results=num_examples)
        for r in result1['results']:
            all_results.append(r)
            scores[r['id']] = result1['scores'].get(r['id'], 0) * 1.2  # Boost primary results
        
        # Strategy 2: Search using key concepts
        if analysis['concepts']:
            concept_query = ' '.join(analysis['concepts'][:3])
            print(f"🔎 Strategy 2: Concept search '{concept_query}'")
            result2 = self.search_engine.search(concept_query, max_results=num_examples)
            for r in result2['results']:
                if r['id'] not in scores:
                    all_results.append(r)
                    scores[r['id']] = result2['scores'].get(r['id'], 0) * 0.9
                else:
                    # Boost score if found in multiple strategies
                    scores[r['id']] += result2['scores'].get(r['id'], 0) * 0.6
        
        # Strategy 3: Intent-based keyword expansion
        intent_keywords = {
            'definition': ['definition', 'meaning', 'refers to', 'is a'],
            'howto': ['steps', 'how to', 'process', 'method'],
            'example': ['example', 'instance', 'sample'],
            'explanation': ['because', 'reason', 'explanation'],
            'comparison': ['difference', 'compare', 'versus'],
            'list': ['types', 'kinds', 'categories', 'includes']
        }
        
        if analysis['intent'] in intent_keywords:
            intent_terms = ' '.join(intent_keywords[analysis['intent']][:2])
            intent_query = query + ' ' + intent_terms
            print(f"🔎 Strategy 3: Intent search '{intent_terms}'")
            result3 = self.search_engine.search(intent_query, max_results=num_examples)
            for r in result3['results']:
                if r['id'] not in scores:
                    all_results.append(r)
                    scores[r['id']] = result3['scores'].get(r['id'], 0) * 0.85
                else:
                    scores[r['id']] += result3['scores'].get(r['id'], 0) * 0.4
        
        # Remove duplicates and sort by combined score
        unique_results = {r['id']: r for r in all_results}.values()
        sorted_results = sorted(unique_results, key=lambda x: scores.get(x['id'], 0), reverse=True)
        
        print(f"✅ Found {len(sorted_results)} unique examples (top score: {max(scores.values()) if scores else 0:.2f})")
        
        return {
            'results': sorted_results[:num_examples * 2],  # Get more for selection
            'scores': scores,
            'search_time_us': result1['search_time_us']
        }
    
    def extract_answer_refined(self, example_text, query, analysis):
        """
        Enhanced extraction with intent-aware processing
        """
        text = example_text.strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if not sentences:
            return text[:300]
        
        # Score sentences based on multiple factors
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        concept_words = set(analysis['concepts'])
        
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
            
            # Calculate multiple scores
            query_overlap = len(query_words & sentence_words)
            concept_overlap = len(concept_words & sentence_words)
            
            # Intent-based scoring
            intent_bonus = 0
            if analysis['intent'] == 'definition' and any(w in sentence_lower for w in ['is a', 'refers to', 'defined as', 'means']):
                intent_bonus = 3
            elif analysis['intent'] == 'howto' and any(w in sentence_lower for w in ['step', 'first', 'then', 'process', 'method']):
                intent_bonus = 2
            elif analysis['intent'] == 'example' and any(w in sentence_lower for w in ['example', 'instance', 'such as', 'like']):
                intent_bonus = 2
            
            # Length penalty for very short or very long sentences
            length_score = 1.0
            if len(sentence) < 30:
                length_score = 0.7
            elif len(sentence) > 200:
                length_score = 0.8
            
            total_score = (query_overlap * 2 + concept_overlap + intent_bonus) * length_score
            
            if total_score > 0:
                scored_sentences.append((total_score, sentence))
        
        # Return top sentences
        if scored_sentences:
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            # Number of sentences based on answer type
            num_sentences = 2
            if analysis['answer_type'] == 'detailed':
                num_sentences = 3
            elif analysis['answer_type'] == 'short':
                num_sentences = 1
            
            top_sentences = [s[1] for s in scored_sentences[:num_sentences]]
            return '. '.join(top_sentences) + '.'
        
        return '. '.join(sentences[:2]) + '.'
    
    def generate_from_examples(self, prompt, num_examples=5, max_length=400):
        """
        Generate answer using:
        1. Model analysis of the query (intent, concepts)
        2. Multi-strategy search for best examples
        3. Refined extraction from examples
        4. Output is STILL 100% from dataset (no model hallucination)
        """
        start_time = time.perf_counter()
        
        print(f"\n{'='*60}")
        print(f"📝 Query: {prompt}")
        print(f"{'='*60}")
        
        # Step 1: Analyze query (model-powered if available)
        analysis = self.analyze_query_with_model(prompt)
        
        # Step 2: Multi-strategy search
        search_result = self.multi_strategy_search(prompt, analysis, num_examples)
        search_time = search_result['search_time_us']
        
        if not search_result['results']:
            return {
                'output': "I don't have enough information in my dataset to answer this question accurately.",
                'metadata': {
                    'method': 'example-based-enhanced',
                    'num_examples': 0,
                    'search_time_us': search_time,
                    'confidence': 'none',
                    'sources': [],
                    'analysis': analysis
                }
            }
        
        # Step 3: Extract and refine from each example
        extract_start = time.perf_counter()
        extracted_parts = []
        
        for result in search_result['results']:
            answer_part = self.extract_answer_refined(result['text'], prompt, analysis)
            if answer_part:
                extracted_parts.append({
                    'text': answer_part,
                    'source': result['source'],
                    'score': search_result['scores'].get(result['id'], 0)
                })
        
        extract_time = (time.perf_counter() - extract_start) * 1_000_000  # μs
        
        # Step 4: Combine and refine the best parts
        if extracted_parts:
            # Sort by score
            extracted_parts.sort(reverse=True, key=lambda x: x['score'])
            
            print(f"📊 Top 3 Example Scores: {[f'{p['score']:.2f}' for p in extracted_parts[:3]]}")
            
            # Build answer from top examples
            answer_parts = []
            total_length = 0
            sources = []
            seen_content = set()  # Avoid duplicates
            
            for part in extracted_parts:
                part_text = part['text']
                
                # Skip very similar content
                part_normalized = re.sub(r'\W+', '', part_text.lower())
                if part_normalized in seen_content:
                    continue
                
                if total_length + len(part_text) <= max_length:
                    answer_parts.append(part_text)
                    sources.append(part['source'])
                    seen_content.add(part_normalized[:100])  # Track first 100 chars
                    total_length += len(part_text)
                
                # Stop if we have enough quality content
                if len(answer_parts) >= 3 and total_length >= max_length * 0.6:
                    break
            
            # Combine into final answer
            final_answer = ' '.join(answer_parts)
            
            # Clean up formatting
            final_answer = re.sub(r'\s+', ' ', final_answer)  # Remove extra spaces
            final_answer = re.sub(r'(\. \.)+', '.', final_answer)  # Fix double periods
            final_answer = final_answer.strip()
            
            # Ensure proper ending
            if final_answer and final_answer[-1] not in '.!?':
                final_answer += '.'
            
            total_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Determine confidence
            confidence = 'low'
            if len(extracted_parts) >= 5 and extracted_parts[0]['score'] > 10:
                confidence = 'high'
            elif len(extracted_parts) >= 3:
                confidence = 'medium'
            
            print(f"✅ Generated {len(final_answer)} chars from {len(answer_parts)} examples")
            print(f"🎯 Confidence: {confidence}")
            print(f"⚡ Total time: {total_time:.2f} ms\n")
            
            return {
                'output': final_answer,
                'metadata': {
                    'method': 'example-based-enhanced',
                    'model_analysis_used': self.use_model_analysis,
                    'intent': analysis['intent'],
                    'answer_type': analysis['answer_type'],
                    'num_examples': len(extracted_parts),
                    'num_parts_used': len(answer_parts),
                    'search_time_us': search_time,
                    'extraction_time_us': extract_time,
                    'total_time_ms': total_time,
                    'confidence': confidence,
                    'sources': list(set(sources)),
                    'top_scores': [p['score'] for p in extracted_parts[:3]],
                    'strategies_used': 3
                }
            }
        
        # Fallback
        return {
            'output': "I couldn't extract a clear answer from the available examples.",
            'metadata': {
                'method': 'example-based-enhanced',
                'num_examples': 0,
                'confidence': 'none',
                'analysis': analysis
            }
        }


def create_example_generator(model=None, tokenizer=None):
    """Factory function to create enhanced example-based generator"""
    return ExampleBasedGenerator(model=model, tokenizer=tokenizer)
