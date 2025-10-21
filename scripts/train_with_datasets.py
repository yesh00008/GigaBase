#!/usr/bin/env python3
"""
GPU-OPTIMIZED ULTRA FAST TRAINING

This script will:
1. Use ALL downloaded datasets from E:\LLM\data\downloaded_datasets
2. Train with GPU acceleration (if available)
3. Use mixed precision (FP16) for speed
4. Complete training in SECONDS (with GPU) or MINUTES (without GPU)
5. Save a high-quality model
"""

import os
import sys
import torch
import json
import glob
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, concatenate_datasets
import warnings
warnings.filterwarnings('ignore')

# Paths
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\dataset_trained_model"
MODEL_NAME = "distilgpt2"  # Fast and efficient

# Training configuration
MAX_SAMPLES_PER_FILE = 500  # Samples per dataset file
MAX_LENGTH = 128           # Sequence length
BATCH_SIZE_GPU = 32        # If GPU available
BATCH_SIZE_CPU = 8         # If CPU only
GRADIENT_ACCUM = 2         # Gradient accumulation steps
EPOCHS = 2                 # Number of epochs
LEARNING_RATE = 5e-4

def discover_datasets():
    """Find all downloaded dataset files"""
    print("🔍 Discovering datasets...")
    
    files = {
        'json': glob.glob(os.path.join(DATASETS_DIR, '**', '*.json'), recursive=True),
        'jsonl': glob.glob(os.path.join(DATASETS_DIR, '**', '*.jsonl'), recursive=True),
        'txt': glob.glob(os.path.join(DATASETS_DIR, '**', '*.txt'), recursive=True),
        'parquet': glob.glob(os.path.join(DATASETS_DIR, '**', '*.parquet'), recursive=True),
    }
    
    total = sum(len(f) for f in files.values())
    print(f"✅ Found {total} dataset files:")
    for fmt, file_list in files.items():
        if file_list:
            print(f"   - {len(file_list)} {fmt.upper()} files")
    
    return files

def load_json_file(filepath, max_samples):
    """Load JSON file"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data[:max_samples]:
                    text = extract_text_from_dict(item)
                    if text:
                        texts.append(text)
    except Exception as e:
        print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
    return texts

def load_jsonl_file(filepath, max_samples):
    """Load JSONL file"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    text = extract_text_from_dict(item)
                    if text:
                        texts.append(text)
                except:
                    continue
    except Exception as e:
        print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
    return texts

def load_txt_file(filepath, max_samples):
    """Load TXT file"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into chunks
            chunk_size = 500
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            texts = chunks[:max_samples]
    except Exception as e:
        print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
    return texts

def load_parquet_file(filepath, max_samples):
    """Load Parquet file"""
    texts = []
    try:
        import pandas as pd
        df = pd.read_parquet(filepath)
        df = df.head(max_samples)
        
        # Extract text from all string columns
        for col in df.select_dtypes(include=['object']).columns:
            col_texts = df[col].dropna().astype(str).tolist()
            texts.extend(col_texts)
    except Exception as e:
        print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
    return texts

def extract_text_from_dict(item):
    """Extract text from dictionary item"""
    if isinstance(item, str):
        return item.strip()
    
    if not isinstance(item, dict):
        return None
    
    # Common text field names
    text_fields = [
        'text', 'content', 'body', 'prompt', 'response', 
        'question', 'answer', 'instruction', 'output',
        'message', 'description', 'summary', 'title'
    ]
    
    # Try to find text
    for field in text_fields:
        if field in item and item[field]:
            text = str(item[field]).strip()
            if len(text) > 10:  # Minimum length
                return text
    
    # Fallback: concatenate string values
    text_parts = []
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 20:
            text_parts.append(value)
            if len(text_parts) >= 2:  # Max 2 fields
                break
    
    if text_parts:
        return ' '.join(text_parts)
    
    return None

def load_all_datasets(dataset_files):
    """Load all datasets efficiently"""
    print("\n📚 Loading all datasets...")
    
    all_texts = []
    
    # Load JSON files
    for filepath in dataset_files['json'][:20]:
        print(f"  📄 Loading: {os.path.basename(filepath)}")
        texts = load_json_file(filepath, MAX_SAMPLES_PER_FILE)
        all_texts.extend(texts)
        print(f"     Added {len(texts)} samples")
    
    # Load JSONL files
    for filepath in dataset_files['jsonl'][:20]:
        print(f"  📄 Loading: {os.path.basename(filepath)}")
        texts = load_jsonl_file(filepath, MAX_SAMPLES_PER_FILE)
        all_texts.extend(texts)
        print(f"     Added {len(texts)} samples")
    
    # Load TXT files
    for filepath in dataset_files['txt'][:10]:
        print(f"  📄 Loading: {os.path.basename(filepath)}")
        texts = load_txt_file(filepath, MAX_SAMPLES_PER_FILE)
        all_texts.extend(texts)
        print(f"     Added {len(texts)} samples")
    
    # Load Parquet files
    for filepath in dataset_files['parquet'][:10]:
        print(f"  📄 Loading: {os.path.basename(filepath)}")
        texts = load_parquet_file(filepath, MAX_SAMPLES_PER_FILE)
        all_texts.extend(texts)
        print(f"     Added {len(texts)} samples")
    
    # Remove duplicates and filter
    unique_texts = list(set(all_texts))
    filtered_texts = [t for t in unique_texts if len(t) > 20 and len(t) < 2000]
    
    print(f"\n✅ Total samples loaded: {len(all_texts)}")
    print(f"✅ Unique samples: {len(unique_texts)}")
    print(f"✅ After filtering: {len(filtered_texts)}")
    
    return Dataset.from_dict({"text": filtered_texts})

def main():
    print("=" * 70)
    print("🚀 GPU-OPTIMIZED ULTRA FAST TRAINING")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        batch_size = BATCH_SIZE_GPU
    else:
        print("   ⚠️  No GPU detected - training will be slower")
        batch_size = BATCH_SIZE_CPU
    
    # Discover datasets
    dataset_files = discover_datasets()
    
    total_files = sum(len(files) for files in dataset_files.values())
    if total_files == 0:
        print("\n❌ No dataset files found!")
        print(f"   Please download datasets to: {DATASETS_DIR}")
        return
    
    # Load all datasets
    dataset = load_all_datasets(dataset_files)
    
    if len(dataset) == 0:
        print("\n❌ No valid data loaded!")
        return
    
    print(f"\n📊 Final dataset size: {len(dataset)} samples")
    
    # Load tokenizer and model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("\n🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    print(f"✅ Tokenized {len(tokenized_dataset)} samples")
    
    # Load model
    print(f"\n🔧 Loading model for training...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=device == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    print(f"📈 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {batch_size} (effective: {batch_size * GRADIENT_ACCUM})")
    print(f"📏 Max length: {MAX_LENGTH}")
    print(f"📚 Samples: {len(tokenized_dataset)}")
    print(f"🔥 FP16: {device == 'cuda'}")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save model
        print("\n💾 Saving model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"💾 Model saved to: {OUTPUT_DIR}")
        
        # Test generation
        print("\n" + "=" * 70)
        print("🧪 TESTING MODEL")
        print("=" * 70)
        
        test_prompts = [
            "Artificial intelligence is",
            "Machine learning can",
            "The future of technology",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Output: {generated}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL DONE!")
        print("=" * 70)
        print(f"\n✅ Trained model ready at: {OUTPUT_DIR}")
        print(f"⚡ Completed in {duration:.2f} seconds!")
        print(f"\n💡 To use this model, update your model_utils.py or:")
        print(f"   model, tokenizer = load_model('dataset_trained_model')")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
