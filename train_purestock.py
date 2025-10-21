#!/usr/bin/env python3
"""
PURESTOCK MODEL - Single Unified Training

Trains ONE powerful model with ALL downloaded datasets.
Model Name: Purestock
Speed: Optimized for fast training
Quality: Maximum (uses all available data)
"""

import os
import sys
import torch
import json
import glob
import shutil
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\Purestock"
MODEL_NAME = "distilgpt2"  # Base model to start from

# Training settings - OPTIMIZED FOR SPEED AND QUALITY
MAX_SAMPLES_PER_FILE = 1000  # Load more data per file
MAX_LENGTH = 128             # Sequence length
BATCH_SIZE = 16              # Batch size (adjust based on RAM/GPU)
GRADIENT_ACCUMULATION = 2    # Effective batch: 32
EPOCHS = 3                   # Train for 3 epochs for better quality
LEARNING_RATE = 5e-4
WARMUP_STEPS = 100

def clean_old_models():
    """Remove all old trained models except Purestock"""
    print("🧹 Cleaning up old models...")
    
    models_dir = r"E:\LLM\models"
    if not os.path.exists(models_dir):
        return
    
    # Models to remove
    old_models = [
        'lightning_fast_model',
        'ultra_fast_trained_v2',
        'dataset_trained_model',
        'fast_trained',
        'collected_data_trained',
    ]
    
    for model_name in old_models:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                print(f"  ✅ Removed: {model_name}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {model_name}: {e}")
    
    print("✅ Cleanup complete - Only Purestock will remain\n")

def discover_all_datasets():
    """Find ALL dataset files"""
    print("🔍 Discovering ALL datasets...")
    
    all_files = []
    
    # Search for all common data formats
    patterns = ['*.json', '*.jsonl', '*.txt', '*.parquet', '*.csv']
    
    for pattern in patterns:
        files = glob.glob(os.path.join(DATASETS_DIR, '**', pattern), recursive=True)
        all_files.extend(files)
    
    print(f"✅ Found {len(all_files)} dataset files\n")
    return all_files

def extract_text_from_item(item):
    """Extract text from any data structure"""
    if isinstance(item, str):
        return item.strip()
    
    if not isinstance(item, dict):
        return None
    
    # Try all possible text fields
    text_fields = [
        'text', 'content', 'body', 'prompt', 'response', 'completion',
        'question', 'answer', 'instruction', 'output', 'input',
        'message', 'description', 'summary', 'title', 'article',
        'paragraph', 'sentence', 'data', 'value', 'story'
    ]
    
    for field in text_fields:
        if field in item and item[field]:
            text = str(item[field]).strip()
            if len(text) > 15:  # Minimum length
                return text
    
    # Fallback: combine multiple fields
    text_parts = []
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 20:
            text_parts.append(value)
            if len(text_parts) >= 3:  # Max 3 fields
                break
    
    if text_parts:
        return ' '.join(text_parts)
    
    return None

def load_all_data(dataset_files):
    """Load ALL data from ALL files"""
    print("📚 Loading ALL data from ALL datasets...")
    print(f"📊 Processing {len(dataset_files)} files...\n")
    
    all_texts = []
    files_processed = 0
    
    for filepath in dataset_files:
        try:
            filename = os.path.basename(filepath)
            print(f"  📄 Loading: {filename[:50]}...")
            
            file_texts = []
            
            # Load based on file type
            if filepath.endswith('.jsonl'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= MAX_SAMPLES_PER_FILE:
                            break
                        try:
                            item = json.loads(line.strip())
                            text = extract_text_from_item(item)
                            if text:
                                file_texts.append(text)
                        except:
                            continue
            
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[:MAX_SAMPLES_PER_FILE]:
                            text = extract_text_from_item(item)
                            if text:
                                file_texts.append(text)
                    elif isinstance(data, dict):
                        text = extract_text_from_item(data)
                        if text:
                            file_texts.append(text)
            
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split into chunks
                    chunk_size = 800
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    file_texts.extend(chunks[:MAX_SAMPLES_PER_FILE])
            
            elif filepath.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(filepath)
                df = df.head(MAX_SAMPLES_PER_FILE)
                for col in df.select_dtypes(include=['object']).columns:
                    texts = df[col].dropna().astype(str).tolist()
                    file_texts.extend(texts)
            
            elif filepath.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(filepath, nrows=MAX_SAMPLES_PER_FILE)
                for col in df.select_dtypes(include=['object']).columns:
                    texts = df[col].dropna().astype(str).tolist()
                    file_texts.extend(texts)
            
            all_texts.extend(file_texts)
            files_processed += 1
            print(f"     ✅ Added {len(file_texts)} samples (Total: {len(all_texts)})")
            
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
            continue
    
    # Remove duplicates and filter
    print(f"\n📊 Processing data...")
    print(f"   - Raw samples: {len(all_texts)}")
    
    unique_texts = list(set(all_texts))
    print(f"   - Unique samples: {len(unique_texts)}")
    
    # Filter: keep text between 20-2000 characters
    filtered_texts = [t for t in unique_texts if 20 <= len(t) <= 2000]
    print(f"   - After filtering: {len(filtered_texts)}")
    
    print(f"\n✅ Data loading complete!")
    print(f"📈 Total training samples: {len(filtered_texts)}\n")
    
    return Dataset.from_dict({"text": filtered_texts})

def main():
    print("=" * 70)
    print("🚀 PURESTOCK MODEL - UNIFIED TRAINING")
    print("=" * 70)
    print("Training ONE powerful model with ALL your data")
    print("Model Name: Purestock")
    print("=" * 70 + "\n")
    
    # Clean old models
    clean_old_models()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device.upper()}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  Training on CPU (slower but works!)")
    print()
    
    # Discover datasets
    dataset_files = discover_all_datasets()
    
    if not dataset_files:
        print("❌ No datasets found!")
        print(f"   Please download datasets to: {DATASETS_DIR}")
        return
    
    # Load ALL data
    dataset = load_all_data(dataset_files)
    
    if len(dataset) == 0:
        print("❌ No valid data loaded!")
        return
    
    # Load tokenizer and model
    print(f"🤖 Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("\n🔤 Tokenizing ALL data...")
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
    
    print(f"✅ Tokenization complete: {len(tokenized_dataset)} samples\n")
    
    # Load model
    print("🔧 Preparing Purestock model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,  # Only keep the best model
        fp16=device == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        weight_decay=0.01,
        max_grad_norm=1.0,
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
    
    # TRAIN!
    print("=" * 70)
    print("🎯 TRAINING PURESTOCK MODEL")
    print("=" * 70)
    print(f"📈 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"📏 Max length: {MAX_LENGTH}")
    print(f"📚 Training samples: {len(tokenized_dataset)}")
    print(f"🔥 FP16: {device == 'cuda'}")
    print(f"💾 Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    try:
        # Train
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save
        print("\n💾 Saving Purestock model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save model info
        model_info = {
            "name": "Purestock",
            "version": "1.0",
            "base_model": MODEL_NAME,
            "training_samples": len(tokenized_dataset),
            "epochs": EPOCHS,
            "training_time_seconds": duration,
            "training_date": datetime.now().isoformat(),
            "datasets_used": len(dataset_files),
        }
        
        with open(os.path.join(OUTPUT_DIR, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✅ PURESTOCK MODEL TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"📊 Trained on: {len(tokenized_dataset)} samples")
        print(f"📁 Datasets used: {len(dataset_files)} files")
        print(f"💾 Saved to: {OUTPUT_DIR}")
        
        # Test generation
        print("\n" + "=" * 70)
        print("🧪 TESTING PURESTOCK")
        print("=" * 70)
        
        test_prompts = [
            "Artificial intelligence is",
            "The future of technology will",
            "Machine learning algorithms can",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=60,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Purestock: {generated}")
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print("\n✅ Purestock model is ready!")
        print(f"📁 Location: {OUTPUT_DIR}")
        print(f"⚡ Trained in: {duration/60:.2f} minutes")
        print("\n💡 To use Purestock in your app:")
        print("   1. Update model_utils.py to load 'Purestock'")
        print("   2. Restart your web server")
        print("   3. Select 'Purestock' from the model dropdown")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
