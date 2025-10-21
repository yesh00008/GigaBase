#!/usr/bin/env python3
"""PURESTOCK - Ultra-Fast Training"""

import os
import json
import glob
import shutil
import torch
from datetime import datetime
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# CONFIG - OPTIMIZED FOR SPEED
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\Purestock"
MODEL_NAME = "distilgpt2"
MAX_TOTAL_SAMPLES = 500  # Limit total samples for speed
MAX_LENGTH = 64          # Shorter sequences = faster
BATCH_SIZE = 8          # Smaller batch = less memory
EPOCHS = 1              # Single epoch for speed
LEARNING_RATE = 5e-4

def clean_old_models():
    """Remove old models"""
    print("🧹 Cleaning...")
    models_dir = r"E:\LLM\models"
    
    for name in ['lightning_fast_model', 'ultra_fast_trained_v2', 'dataset_trained_model', 'fast_trained', 'collected_data_trained']:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✅ {name}")
            except:
                pass
    print()

def load_fast():
    """Load data FAST"""
    print("📚 Loading data...")
    
    files = glob.glob(os.path.join(DATASETS_DIR, "*.jsonl"))
    print(f"Found {len(files)} files\n")
    
    texts = []
    
    for filepath in files:
        name = os.path.basename(filepath)[:40]
        print(f"  {name}...", end=" ")
        
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if len(texts) >= MAX_TOTAL_SAMPLES:
                        break
                    if i >= 100:  # Max 100 per file
                        break
                    
                    try:
                        item = json.loads(line)
                        text = None
                        
                        if isinstance(item, dict):
                            for field in ['text', 'content', 'prompt', 'response']:
                                if field in item and item[field]:
                                    text = str(item[field]).strip()
                                    if len(text) > 20:
                                        break
                        
                        if text and 20 <= len(text) <= 1000:
                            texts.append(text)
                            count += 1
                    except:
                        continue
        except:
            pass
        
        print(f"{count}")
        
        if len(texts) >= MAX_TOTAL_SAMPLES:
            break
    
    # Remove duplicates
    unique = list(set(texts))
    print(f"\n✅ {len(unique)} unique samples\n")
    
    return Dataset.from_dict({"text": unique[:MAX_TOTAL_SAMPLES]})

def main():
    print("\n" + "=" * 60)
    print("🚀 PURESTOCK - ULTRA-FAST TRAINING")
    print("=" * 60 + "\n")
    
    clean_old_models()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}\n")
    
    # Load data
    dataset = load_fast()
    print(f"Training samples: {len(dataset)}\n")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("Tokenizing...")
    def tokenize(x):
        return tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH)
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"✅ {len(tokenized)} tokenized\n")
    
    # Model
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        fp16=False,  # Disable FP16 for stability
        dataloader_num_workers=0,
        push_to_hub=False,
        report_to="none",
    )
    
    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    
    # TRAIN
    print("\n" + "=" * 60)
    print(f"🎯 TRAINING PURESTOCK")
    print(f"Samples: {len(tokenized)} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    start = datetime.now()
    
    try:
        trainer.train()
        
        duration = (datetime.now() - start).total_seconds()
        
        # Save
        print("\n💾 Saving...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        info = {
            "name": "Purestock",
            "samples": len(tokenized),
            "training_time": f"{duration:.0f}s",
        }
        with open(os.path.join(OUTPUT_DIR, "model_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ PURESTOCK COMPLETE!")
        print(f"⏱️  {duration:.0f}s ({duration/60:.1f} min)")
        print(f"📊 {len(tokenized)} samples")
        print(f"💾 {OUTPUT_DIR}")
        print("=" * 60)
        
        # Test
        print("\n🧪 Testing...\n")
        for prompt in ["Artificial intelligence is", "The future of"]:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.8)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"'{prompt}' → {text[:80]}...\n")
        
        print("=" * 60)
        print("🎉 PURESTOCK IS READY!")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
