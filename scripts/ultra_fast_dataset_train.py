#!/usr/bin/env python3
"""
Ultra-Fast Training Script Using Downloaded Datasets

This script trains a model EXTREMELY FAST using:
- GPU acceleration (if available)
- Mixed precision training (FP16)
- Gradient accumulation
- Optimized data loading
- Small batch processing for speed
- Downloaded datasets from E:\LLM\data\downloaded_datasets
"""

import os
import sys
import torch
import json
import glob
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets, Dataset
import warnings
warnings.filterwarnings('ignore')

# Configuration for ULTRA FAST training
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\ultra_fast_trained_v2"
MODEL_NAME = "distilgpt2"  # Small and fast

# Ultra-fast training settings
MAX_SAMPLES_PER_DATASET = 100  # Very small for speed
MAX_LENGTH = 128  # Short sequences for speed
BATCH_SIZE = 8  # Larger batches if GPU available
GRADIENT_ACCUMULATION = 2  # Effective batch size: 16
EPOCHS = 1  # Single epoch for speed
LEARNING_RATE = 5e-4  # Higher learning rate for faster convergence

def find_dataset_files():
    """Find all downloaded dataset files"""
    print("🔍 Searching for downloaded datasets...")
    
    dataset_files = []
    
    # Look for JSON, JSONL, CSV, and TXT files
    patterns = ['*.json', '*.jsonl', '*.csv', '*.txt', '*.parquet']
    
    for pattern in patterns:
        files = glob.glob(os.path.join(DATASETS_DIR, '**', pattern), recursive=True)
        dataset_files.extend(files)
    
    print(f"✅ Found {len(dataset_files)} dataset files")
    return dataset_files

def load_and_prepare_data(dataset_files):
    """Load and prepare data from all datasets"""
    print("\n📂 Loading datasets...")
    
    all_texts = []
    
    for file_path in dataset_files[:10]:  # Limit to 10 files for speed
        try:
            print(f"  Loading: {os.path.basename(file_path)}")
            
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[:MAX_SAMPLES_PER_DATASET]:
                            # Try to extract text from different formats
                            text = extract_text(item)
                            if text:
                                all_texts.append(text)
                    elif isinstance(data, dict):
                        text = extract_text(data)
                        if text:
                            all_texts.append(text)
            
            elif file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= MAX_SAMPLES_PER_DATASET:
                            break
                        try:
                            item = json.loads(line)
                            text = extract_text(item)
                            if text:
                                all_texts.append(text)
                        except:
                            continue
            
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split into chunks
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                    all_texts.extend(chunks[:MAX_SAMPLES_PER_DATASET])
            
            elif file_path.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path, nrows=MAX_SAMPLES_PER_DATASET)
                    # Get text from all string columns
                    for col in df.select_dtypes(include=['object']).columns:
                        all_texts.extend(df[col].dropna().astype(str).tolist())
                except:
                    pass
            
            elif file_path.endswith('.parquet'):
                try:
                    import pandas as pd
                    df = pd.read_parquet(file_path)
                    df = df.head(MAX_SAMPLES_PER_DATASET)
                    for col in df.select_dtypes(include=['object']).columns:
                        all_texts.extend(df[col].dropna().astype(str).tolist())
                except:
                    pass
                    
        except Exception as e:
            print(f"  ⚠️  Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    print(f"\n✅ Loaded {len(all_texts)} text samples")
    
    # Create dataset
    if all_texts:
        dataset = Dataset.from_dict({"text": all_texts})
        return dataset
    else:
        print("❌ No data loaded!")
        return None

def extract_text(item):
    """Extract text from various data formats"""
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        # Try common text field names
        for key in ['text', 'content', 'prompt', 'response', 'question', 'answer', 
                    'instruction', 'output', 'message', 'body', 'description']:
            if key in item and item[key]:
                return str(item[key])
        
        # Try to concatenate multiple fields
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 10:
                text_parts.append(value)
        
        if text_parts:
            return ' '.join(text_parts[:3])  # Max 3 fields
    
    return None

def tokenize_function(examples, tokenizer):
    """Tokenize the texts"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

def main():
    print("=" * 70)
    print("🚀 ULTRA-FAST MODEL TRAINING WITH DOWNLOADED DATASETS")
    print("=" * 70)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Find dataset files
    dataset_files = find_dataset_files()
    
    if not dataset_files:
        print("❌ No dataset files found! Please download datasets first.")
        print(f"   Expected location: {DATASETS_DIR}")
        return
    
    # Load and prepare data
    dataset = load_and_prepare_data(dataset_files)
    
    if dataset is None or len(dataset) == 0:
        print("❌ No data could be loaded from the files!")
        return
    
    print(f"\n📊 Dataset size: {len(dataset)} samples")
    
    # Load tokenizer and model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    print("\n🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments - OPTIMIZED FOR SPEED
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=10,  # Minimal warmup
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        fp16=device == "cuda",  # Mixed precision for GPU
        dataloader_num_workers=0,  # Single worker for speed
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",  # No logging overhead
        gradient_checkpointing=False,  # Faster but more memory
        optim="adamw_torch",  # Fast optimizer
        max_grad_norm=1.0,
        dataloader_pin_memory=True if device == "cuda" else False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING...")
    print("=" * 70)
    print(f"📈 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"📏 Max sequence length: {MAX_LENGTH}")
    print(f"📚 Training samples: {len(tokenized_dataset)}")
    print(f"🔥 Mixed precision (FP16): {device == 'cuda'}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"💾 Model saved to: {OUTPUT_DIR}")
        
        # Save tokenizer
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"💾 Tokenizer saved to: {OUTPUT_DIR}")
        
        # Test generation
        print("\n" + "=" * 70)
        print("🧪 TESTING MODEL...")
        print("=" * 70)
        
        test_prompt = "Artificial intelligence is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"🤖 Generated: {generated_text}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL DONE!")
        print("=" * 70)
        print(f"\n✅ Your trained model is ready at: {OUTPUT_DIR}")
        print(f"⚡ Training completed in {duration:.2f} seconds!")
        print("\n💡 To use this model in your app, update model_utils.py to load from:")
        print(f"   '{OUTPUT_DIR}'")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
