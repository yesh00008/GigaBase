#!/usr/bin/env python3
"""
PURESTOCK MODEL - Single Unified Training
Trains ALL downloaded data into ONE model named "Purestock"
"""

import os
import sys
import torch
import json
import glob
import shutil
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

# CONFIGURATION
DATASETS_DIR = r"E:\LLM\data\downloaded_datasets"
OUTPUT_DIR = r"E:\LLM\models\Purestock"
MODEL_NAME = "distilgpt2"

# SPEED + QUALITY BALANCE
MAX_SAMPLES_PER_FILE = 800  # More samples per file
MAX_LENGTH = 96             # Medium sequence length
BATCH_SIZE = 12             # Balanced batch size
EPOCHS = 2                  # 2 epochs for better quality
LEARNING_RATE = 5e-4

def clean_old_models():
    """Remove ALL old models"""
    print("🧹 Removing old models...")
    models_dir = r"E:\LLM\models"
    
    old_models = [
        'lightning_fast_model',
        'ultra_fast_trained_v2', 
        'dataset_trained_model',
        'fast_trained',
        'collected_data_trained',
        'fine_tuned',
    ]
    
    for model_name in old_models:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                print(f"  ✅ Removed: {model_name}")
            except:
                pass
    
    print("✅ Cleanup done!\n")

def load_all_data():
    """Load ALL data from downloaded datasets"""
    print("📚 Loading ALL downloaded data...")
    
    texts = []
    files = glob.glob(os.path.join(DATASETS_DIR, '**', '*.json*'), recursive=True)
    
    print(f"Found {len(files)} files\n")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"  📄 {filename[:55]}...", end=" ")
        
        file_texts = []
        
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= MAX_SAMPLES_PER_FILE:
                            break
                        try:
                            item = json.loads(line.strip())
                            
                            # Extract text from any structure
                            text = None
                            if isinstance(item, str):
                                text = item
                            elif isinstance(item, dict):
                                # Try all common fields
                                for field in ['text', 'content', 'prompt', 'response', 'completion', 
                                             'question', 'answer', 'instruction', 'output']:
                                    if field in item and item[field]:
                                        text = str(item[field]).strip()
                                        if len(text) > 20:
                                            break
                            
                            if text and 20 <= len(text) <= 1500:
                                file_texts.append(text)
                        
                        except:
                            continue
            
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[:MAX_SAMPLES_PER_FILE]:
                            if isinstance(item, dict):
                                for field in ['text', 'content', 'prompt', 'response']:
                                    if field in item and item[field]:
                                        text = str(item[field]).strip()
                                        if 20 <= len(text) <= 1500:
                                            file_texts.append(text)
                                            break
            
            texts.extend(file_texts)
            print(f"{len(file_texts)} samples (Total: {len(texts)})")
        
        except Exception as e:
            print(f"⚠️  {str(e)[:30]}")
    
    # Remove duplicates
    print(f"\n📊 Total raw samples: {len(texts)}")
    unique_texts = list(set(texts))
    print(f"📊 Unique samples: {len(unique_texts)}")
    
    return Dataset.from_dict({"text": unique_texts})

def main():
    print("\n" + "=" * 70)
    print("🚀 PURESTOCK MODEL - TRAINING")
    print("=" * 70)
    print("ONE MODEL with ALL DATA")
    print("=" * 70 + "\n")
    
    # Clean old models first
    clean_old_models()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load ALL data
    dataset = load_all_data()
    
    if len(dataset) == 0:
        print("\n❌ No data found!")
        return
    
    print(f"\n✅ Total training samples: {len(dataset)}\n")
    
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
        desc="Tokenizing"
    )
    
    print(f"✅ Tokenized {len(tokenized_dataset)} samples\n")
    
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
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # TRAIN!
    print("\n" + "=" * 70)
    print("🎯 TRAINING PURESTOCK NOW")
    print("=" * 70)
    print(f"📈 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"📏 Max length: {MAX_LENGTH}")
    print(f"📚 Samples: {len(tokenized_dataset)}")
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
        
        # Save info
        model_info = {
            "name": "Purestock",
            "version": "1.0",
            "base_model": MODEL_NAME,
            "training_samples": len(tokenized_dataset),
            "epochs": EPOCHS,
            "training_time_seconds": duration,
            "training_date": datetime.now().isoformat(),
        }
        
        with open(os.path.join(OUTPUT_DIR, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✅ PURESTOCK TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration:.0f}s ({duration/60:.1f} min)")
        print(f"📊 Trained on: {len(tokenized_dataset)} samples")
        print(f"💾 Saved to: {OUTPUT_DIR}")
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
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"📝 '{prompt}'")
            print(f"🤖 {generated}\n")
        
        print("=" * 70)
        print("🎉 PURESTOCK IS READY!")
        print("=" * 70)
        print("\n💡 To use Purestock:")
        print("   1. Restart your web app")
        print("   2. Select 'Purestock' from the model dropdown")
        print("   3. Start chatting!")
        print()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
