#!/usr/bin/env python3
"""
EXTREME SPEED Training - Trains in SECONDS!

This script is optimized for MAXIMUM SPEED:
- Tiny dataset (100-500 samples)
- Very short sequences (64 tokens)
- Large batch size
- Single epoch
- Minimal logging
- GPU optimized
"""

import os
import sys
import torch
import json
import glob
from datetime import datetime
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

# EXTREME SPEED SETTINGS
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\lightning_fast_model"
MODEL_NAME = "distilgpt2"

# ULTRA MINIMAL SETTINGS FOR SPEED
MAX_SAMPLES = 200  # Tiny dataset
MAX_LENGTH = 64    # Very short sequences
BATCH_SIZE = 16    # Large batches
EPOCHS = 1         # Single pass
SAVE_STEPS = 1000  # Rarely save

def load_quick_data():
    """Load data as fast as possible"""
    print("⚡ Quick-loading data...")
    
    texts = []
    files = glob.glob(os.path.join(DATASETS_DIR, '**', '*.json*'), recursive=True)
    
    for file_path in files[:3]:  # Only first 3 files
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= MAX_SAMPLES // 3:
                            break
                        try:
                            item = json.loads(line)
                            # Quick text extraction
                            for key in ['text', 'content', 'prompt']:
                                if key in item:
                                    texts.append(str(item[key])[:500])
                                    break
                        except:
                            continue
            if len(texts) >= MAX_SAMPLES:
                break
        except:
            continue
    
    # Add some fallback data if needed
    if len(texts) < 50:
        texts.extend([
            "Artificial intelligence is transforming the world.",
            "Machine learning models can learn from data.",
            "Python is a popular programming language.",
            "Deep learning uses neural networks.",
            "Natural language processing enables text understanding.",
        ] * 10)
    
    return Dataset.from_dict({"text": texts[:MAX_SAMPLES]})

def main():
    print("=" * 60)
    print("⚡ LIGHTNING FAST TRAINING - COMPLETES IN SECONDS!")
    print("=" * 60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    
    # Load data
    dataset = load_quick_data()
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("🔤 Tokenizing...")
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Training args - EXTREME SPEED
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        warmup_steps=0,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        fp16=device == "cuda",
        dataloader_num_workers=0,
        report_to="none",
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # TRAIN!
    print("\n⚡ TRAINING NOW...")
    start = datetime.now()
    
    trainer.train()
    
    duration = (datetime.now() - start).total_seconds()
    
    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"⏱️  Time: {duration:.1f} seconds")
    print(f"💾 Saved: {OUTPUT_DIR}")
    
    # Quick test
    print("\n🧪 Quick test:")
    inputs = tokenizer("AI is", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=30)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
