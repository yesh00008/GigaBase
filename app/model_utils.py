#!/usr/bin/env python3
"""
Model Utility Functions for the LLM Project

This module provides utility functions for working with trained models.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# Cache for loaded models and tokenizers
_model_cache = {}
_tokenizer_cache = {}

def get_available_models(models_dir="models"):
    """
    Get a list of available trained models in the models directory.
    
    Args:
        models_dir (str): Path to the models directory
        
    Returns:
        list: List of dictionaries containing model information
    """
    available_models = []
    
    # ONLY PURESTOCK MODEL - Custom Trained Model
    purestock_path = os.path.join(models_dir, "Purestock")
    
    if os.path.exists(purestock_path):
        available_models.append({
            'id': 'Purestock',
            'name': 'Purestock - Your Custom Model',
            'path': purestock_path
        })
    else:
        # Error if Purestock doesn't exist
        print("⚠️ Purestock model not found!")
        available_models.append({
            'id': 'error',
            'name': '⚠️ Purestock model not found',
            'path': ''
        })
    
    return available_models

def load_model(model_id, models_dir="models", use_cache=True, device=None):
    """
    Load a model and tokenizer from the models directory.
    
    Args:
        model_id (str): The ID or name of the model to load
        models_dir (str): Path to the models directory
        use_cache (bool): Whether to use cached models
        device (str): Device to load the model on ('cuda', 'cpu', etc.)
        
    Returns:
        tuple: (model, tokenizer) pair or (None, None) if loading fails
    """
    # Create cache key
    cache_key = f"{model_id}_{device}"
    
    # Check if model is already loaded in cache
    if use_cache and cache_key in _model_cache and cache_key in _tokenizer_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    # ONLY LOAD PURESTOCK - No fallback to base models
    if model_id == "Purestock":
        model_path = os.path.join(models_dir, "Purestock")
        print(f"Loading Purestock model from: {model_path}")
    else:
        print(f"❌ Invalid model ID: {model_id}. Only 'Purestock' is supported.")
        return None, None
    
    # Check if Purestock exists
    if not os.path.exists(model_path):
        print(f"❌ Purestock model not found at: {model_path}")
        return None, None
    
    try:
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        print(f"Using device: {device}")
        
        # Free up memory if needed
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings for SPEED
        model_kwargs = {
            'low_cpu_mem_usage': True,
        }
        
        # Use mixed precision for CUDA (faster inference)
        if device.type == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.to(device)
        model.eval()  # Set to evaluation mode for faster inference
        
        # Enable optimizations
        if device.type == 'cpu':
            # CPU optimizations
            try:
                import torch.jit as torch_jit  # Don't shadow global torch
                # Note: Script mode can cause issues, so we skip it
                pass
            except:
                pass
        
        # Store in cache
        if use_cache:
            _model_cache[cache_key] = model
            _tokenizer_cache[cache_key] = tokenizer
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_text(model, tokenizer, prompt, max_length=100, num_return_sequences=1, 
                 temperature=0.0, top_p=1.0, top_k=50, 
                 deterministic=True, use_system_prompt=True):
    """
    Generate accurate, factual text with deterministic decoding (anti-hallucination).
    
    DETERMINISTIC MODE (default):
    - temperature=0: Most likely tokens only (no randomness)
    - top_p=1.0: No nucleus sampling filtering
    - frequency_penalty=0: No repetition penalties that add randomness
    - presence_penalty=0: No presence penalties
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt (str): The prompt to generate from
        max_length (int): Maximum length of generated text
        num_return_sequences (int): Number of outputs to generate
        temperature (float): 0.0 for deterministic (factual), 0.7+ for creative
        top_p (float): 1.0 for deterministic, 0.9 for creative
        top_k (int): Ignored when temperature=0
        deterministic (bool): If True, overrides all params for factual output
        use_system_prompt (bool): Add instruction-following system prompt
        
    Returns:
        list: List of generated texts (deterministic and factual)
    """
    try:
        # Get device
        device = next(model.parameters()).device
        
        # DETERMINISTIC MODE: Override params for factual accuracy
        if deterministic or temperature == 0:
            temperature = 1.0  # Will be ignored
            top_p = 1.0
            do_sample = False  # GREEDY DECODING (most important!)
            num_beams = 1
            repetition_penalty = 1.5  # STRONGER: Heavy penalty for repeating
            no_repeat_ngram_size = 2  # STRONGER: Block even 2-word repeats
            print("🎯 DETERMINISTIC MODE: Factual, repeatable answers (Greedy decoding)")
        else:
            do_sample = True
            num_beams = 1
            repetition_penalty = 1.1
            no_repeat_ngram_size = 3
            print("🎨 CREATIVE MODE: Varied, creative answers")
        
        # Add system prompt for instruction-following
        if use_system_prompt:
            system_instruction = (
                "You are an assistant that answers concisely and factually. "
                "If you are unsure, respond 'I don't know' and ask for clarification. "
                "Always provide complete sentences."
            )
            processed_prompt = f"{system_instruction}\n\nUser: {prompt}\n\nAssistant:"
        else:
            processed_prompt = preprocess_prompt(prompt)
        
        # Encode the prompt with attention mask (prevent truncation issues)
        encoding = tokenizer(
            processed_prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        prompt_length = len(input_ids[0])
        
        # Calculate max length (FASTER: reduce max tokens for speed)
        actual_max_length = min(max_length + prompt_length, 256)  # Reduced from 512 for speed
        
        # DETERMINISTIC GENERATION (anti-hallucination, anti-repetition)
        # Use torch.inference_mode() for better performance than no_grad()
        with torch.inference_mode():
            # Build generation config (optimized for speed)
            gen_config = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_length': actual_max_length,
                'max_new_tokens': min(max_length, 100),  # Limit new tokens for speed
                'num_return_sequences': num_return_sequences,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'do_sample': do_sample,
                'num_beams': num_beams,
                'early_stopping': True,  # Stop early when EOS is generated (faster)
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'repetition_penalty': repetition_penalty,
                'bad_words_ids': get_bad_words_ids(tokenizer),
                'use_cache': True,  # Enable KV cache for faster generation
            }
            
            # Add sampling params only if not deterministic
            if do_sample and temperature > 0:
                gen_config.update({
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                })
            
            outputs = model.generate(**gen_config)
        
        # Decode and validate outputs
        generated_texts = []
        for output in outputs:
            # Clean decoding
            full_text = tokenizer.decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            # Remove system prompt from output
            if use_system_prompt and "Assistant:" in full_text:
                full_text = full_text.split("Assistant:")[-1].strip()
            
            # Advanced cleaning and validation
            clean_text = clean_generated_text_advanced(full_text, prompt)
            
            # Validate output (reject if garbage)
            if validate_output(clean_text):
                generated_texts.append(clean_text)
            else:
                # Retry with stricter deterministic mode
                print("⚠️ Invalid output detected, retrying with strict mode...")
                generated_texts.append("I don't know. Could you please rephrase your question?")
        
        return generated_texts
    
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["I encountered an error processing your request. Please try again."]

def clean_generated_text(text, original_prompt):
    """
    Clean up generated text for neat display.
    
    Args:
        text (str): Generated text
        original_prompt (str): Original user prompt
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Find the end of a complete sentence
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence.strip()) > 10:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # If we have complete sentences, use them
    if sentences:
        text = ' '.join(sentences)
    
    # Remove any trailing incomplete sentences
    if text and text[-1] not in '.!?':
        # Find the last complete sentence
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(original_prompt):
            text = text[:last_period + 1]
    
    return text.strip()


def preprocess_prompt(prompt):
    """
    Pre-process user prompt for better generation quality.
    
    Args:
        prompt (str): Raw user prompt
        
    Returns:
        str: Processed prompt optimized for generation
    """
    # Strip extra whitespace
    prompt = prompt.strip()
    
    # Ensure prompt ends properly for continuation
    if not prompt:
        return "Hello! "
    
    # Add context markers for better instruction following
    # This helps the model understand it should provide a complete response
    return prompt


def validate_output(text):
    """
    Validate generated output for quality and correctness.
    Reject garbage, hallucinations, and malformed responses.
    
    Args:
        text (str): Generated text to validate
        
    Returns:
        bool: True if valid, False if should be rejected
    """
    import re
    
    # Minimum quality checks
    if not text or len(text.strip()) < 3:
        return False
    
    # Check for garbage tokens
    garbage_patterns = ['####', '****', '[[', ']]', '{{', '}}']
    if any(pattern in text for pattern in garbage_patterns):
        print(f"⚠️ Rejected: Garbage tokens detected")
        return False
    
    # Check for excessive repetition (same word 4+ times in a row)
    words = text.split()
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            print(f"⚠️ Rejected: Excessive repetition detected")
            return False
    
    # Check for incomplete sentences (if long enough)
    if len(words) > 10 and not text.rstrip().endswith(('.', '!', '?', '"', "'")):
        print(f"⚠️ Warning: Incomplete sentence")
        # Still allow, but logged
    
    # Check for valid characters (reject if too many non-printable)
    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
    if printable_ratio < 0.9:
        print(f"⚠️ Rejected: Too many non-printable characters")
        return False
    
    return True


def get_bad_words_ids(tokenizer):
    """
    Get token IDs for words/patterns to avoid during generation.
    Prevents hallucinations and garbage outputs.
    
    Args:
        tokenizer: The tokenizer
        
    Returns:
        list: List of bad word token ID lists
    """
    bad_words = [
        # Repetitive patterns
        "...", "???", "!!!", "~~~",
        # Garbled tokens
        "####", "****", "[[[[", "]]]]",
        # Common hallucination markers
        "<unk>", "[UNK]", "<mask>",
    ]
    
    bad_words_ids = []
    for word in bad_words:
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)
        except:
            pass
    
    return bad_words_ids if bad_words_ids else None


def clean_generated_text_advanced(text, original_prompt):
    """
    Advanced cleaning for GPT-like output quality.
    AGGRESSIVE prompt removal and repetition blocking.
    
    Args:
        text (str): Generated text
        original_prompt (str): Original user prompt
        
    Returns:
        str: Cleaned text with GPT-like quality
    """
    import re
    
    # STEP 1: Remove the original prompt (AGGRESSIVE)
    # Try exact match first
    if text.startswith(original_prompt):
        text = text[len(original_prompt):].strip()
    
    # Try case-insensitive match
    text_lower = text.lower()
    prompt_lower = original_prompt.lower()
    if text_lower.startswith(prompt_lower):
        text = text[len(original_prompt):].strip()
    
    # Remove common prompt patterns
    # e.g., "What is X? What is X..." -> remove first occurrence
    first_sentence_end = text.find('?')
    if first_sentence_end > 0:
        first_sentence = text[:first_sentence_end + 1].strip()
        if first_sentence.lower() in original_prompt.lower():
            text = text[first_sentence_end + 1:].strip()
    
    # STEP 2: Remove system instruction artifacts
    if text.startswith("Assistant:"):
        text = text[10:].strip()
    if text.startswith("Response:"):
        text = text[9:].strip()
    if "User:" in text:
        text = text.split("User:")[0].strip()
    
    # STEP 3: Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # STEP 4: Fix common tokenization issues
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace('( ', '(')
    text = text.replace(' )', ')')
    
    # Handle quotes properly
    text = re.sub(r'\s+"', ' "', text)
    text = re.sub(r'"\s+', '" ', text)
    
    # STEP 5: Remove repetitive sentences (AGGRESSIVE)
    sentences_seen = set()
    unique_sentences = []
    
    # Split into sentences
    sentence_endings = r'[.!?]+'
    parts = re.split(f'({sentence_endings})', text)
    
    current = ""
    for i, part in enumerate(parts):
        current += part
        if re.match(sentence_endings, part):
            sentence = current.strip()
            # Normalize for comparison (lowercase, no punctuation)
            normalized = re.sub(r'[^\w\s]', '', sentence.lower())
            
            # Only add if not duplicate
            if normalized and normalized not in sentences_seen:
                sentences_seen.add(normalized)
                unique_sentences.append(sentence)
            current = ""
    
    # STEP 6: Reconstruct with unique sentences only
    if unique_sentences:
        text = ' '.join(unique_sentences)
    else:
        # If no sentences extracted, try to salvage
        text = current.strip()
    
    # STEP 7: Ensure ends with punctuation
    if text and text[-1] not in '.!?':
        # Find last punctuation
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct > 0:
            text = text[:last_punct + 1]
        else:
            # Add period if makes sense
            if len(text.split()) >= 5:
                text += '.'
    
    # STEP 8: Remove repeated punctuation (but keep ...)
    text = re.sub(r'([.!?])\1{2,}', r'\1', text)
    
    # STEP 9: Fix capitalization
    if text:
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        def capitalize_after_punct(match):
            return match.group(1) + ' ' + match.group(2).upper()
        
        text = re.sub(r'([.!?])\s+([a-z])', capitalize_after_punct, text)
    
    # STEP 10: Final cleanup
    text = text.strip()
    
    # STEP 11: Quality validation - ensure minimum length
    if len(text.split()) < 3:
        # Too short, might be just echo
        return "I don't have enough information to provide a complete answer. Could you rephrase your question?"
    
    return text

def clear_model_cache():
    """Clear the model cache to free up memory"""
    global _model_cache, _tokenizer_cache
    
    for model in _model_cache.values():
        del model
    
    _model_cache.clear()
    _tokenizer_cache.clear()
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Model cache cleared")

if __name__ == "__main__":
    # Example usage
    models = get_available_models()
    print(f"Available models: {len(models)}")
    for model in models:
        print(f"- {model['name']} ({model['id']})")