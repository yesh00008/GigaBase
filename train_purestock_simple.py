#!/usr/bin/env python3
"""
PURESTOCK MODEL - Fast Training (JSONL only)
Trains ONE model with ALL JSONL datasets
"""

import os
import json
import glob
import shutil
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# CONFIGURATION
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\Purestock"
MODEL_NAME = "distilgpt2"
MAX_SAMPLES_PER_FILE = 1000
MAX_LENGTH = 128
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
EPOCHS = 3
LEARNING_RATE = 5e-4
WARMUP_STEPS = 100

def clean_old_models():
    """Remove old models"""
    print("🧹 Cleaning old models...")
    models_dir = r"E:\LLM\models"
    
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
    
    print("✅ Cleanup complete!\n")

def load_jsonl_datasets():
    """Load all JSONL files"""
    print("📚 Loading JSONL datasets...\n")
    
    jsonl_files = glob.glob(os.path.join(DATASETS_DIR, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files\n")
    
    all_texts = []
    
    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        print(f"  📄 {filename[:60]}...")
        
        file_texts = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= MAX_SAMPLES_PER_FILE:
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract text
                        text = None
                        if isinstance(item, str):
                            text = item
                        elif isinstance(item, dict):
                            # Try common fields
                            for field in ['text', 'content', 'prompt', 'response', 'question', 'answer']:
                                if field in item and item[field]:
                                    text = str(item[field]).strip()
                                    if len(text) > 20:
                                        break
                        
                        if text and 20 <= len(text) <= 2000:
                            file_texts.append(text)
                    
                    except:
                        continue
            
            all_texts.extend(file_texts)
            print(f"     ✅ {len(file_texts)} samples (Total: {len(all_texts)})")
        
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
    
    # Remove duplicates
    print(f"\n📊 Raw: {len(all_texts)} samples")
    unique_texts = list(set(all_texts))
    print(f"📊 Unique: {len(unique_texts)} samples\n")
    
    return Dataset.from_dict({"text": unique_texts})

def main():
    print("=" * 70)
    print("🚀 PURESTOCK MODEL TRAINING")
    print("=" * 70)
    print("Training ONE model with ALL data")
    print("=" * 70 + "\n")
    
    # Clean old models
    clean_old_models()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device.upper()}\n")
    
    # Load data
    dataset = load_jsonl_datasets()
    
    if len(dataset) == 0:
        print("❌ No data loaded!")
        return
    
    print(f"✅ Total samples: {len(dataset)}\n")
    
    # Load tokenizer
    print(f"🤖 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("🔤 Tokenizing data...")
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
    
    print(f"✅ Tokenized: {len(tokenized_dataset)} samples\n")
    
    # Load model
    print(f"🔧 Loading model: {MODEL_NAME}")
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
        save_total_limit=1,
        fp16=device == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print("\n" + "=" * 70)
    print("🎯 TRAINING PURESTOCK")
    print("=" * 70)
    print(f"📈 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"📚 Samples: {len(tokenized_dataset)}")
    print(f"💾 Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    start = datetime.now()
    
    try:
        trainer.train()
        
        duration = (datetime.now() - start).total_seconds()
        
        # Save
        print("\n💾 Saving Purestock...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save info
        model_info = {
            "name": "Purestock",
            "version": "1.0",
            "base_model": MODEL_NAME,
            "samples": len(tokenized_dataset),
            "epochs": EPOCHS,
            "training_time_seconds": duration,
            "training_date": datetime.now().isoformat(),
        }
        
        with open(os.path.join(OUTPUT_DIR, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✅ PURESTOCK TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Time: {duration:.0f}s ({duration/60:.1f} min)")
        print(f"📊 Samples: {len(tokenized_dataset)}")
        print(f"💾 Saved: {OUTPUT_DIR}")
        print("=" * 70)
        
        # Test
        print("\n🧪 Testing Purestock...\n")
        
        test_prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "Machine learning can",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=60,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"📝 {prompt}")
            print(f"🤖 {text}\n")
        
        print("=" * 70)
        print("🎉 PURESTOCK IS READY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
