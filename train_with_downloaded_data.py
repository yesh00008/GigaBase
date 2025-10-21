#!/usr/bin/env python3
"""
Train LLM with Downloaded Datasets

This script trains your model using the downloaded datasets from Hugging Face.
"""

import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
from datetime import datetime

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    print(f"📖 Loading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except:
                continue
    
    print(f"   ✅ Loaded {len(data):,} samples")
    return data

def extract_text_from_sample(sample):
    """Extract text content from different dataset formats"""
    # Try common text fields
    text_fields = ['text', 'content', 'story', 'question', 'answer', 'instruction', 'output', 'response']
    
    for field in text_fields:
        if field in sample and sample[field]:
            return str(sample[field])
    
    # If multiple fields, combine them
    if 'question' in sample and 'answer' in sample:
        return f"Q: {sample['question']}\nA: {sample['answer']}"
    
    if 'instruction' in sample and 'output' in sample:
        return f"{sample['instruction']}\n{sample['output']}"
    
    # Fallback: convert entire sample to string
    return str(sample)

def prepare_training_data(data_files):
    """Prepare training data from multiple files"""
    all_texts = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            data = load_jsonl_data(file_path)
            texts = [extract_text_from_sample(sample) for sample in data]
            all_texts.extend(texts)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n📊 Total training samples: {len(all_texts):,}")
    return all_texts

def train_model(data_files, output_dir, model_name="distilgpt2", epochs=3, batch_size=4):
    """Train the model with downloaded data"""
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRAINING WITH DOWNLOADED DATASETS                         ║
║                                                                              ║
║  Model: {model_name:62}  ║
║  Output: {output_dir:60}  ║
║  Epochs: {epochs:2}                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Prepare data
    print("\n1️⃣ Loading and preparing data...")
    texts = prepare_training_data(data_files)
    
    if not texts:
        print("❌ No data loaded! Check your data files.")
        return
    
    # Load tokenizer and model
    print(f"\n2️⃣ Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
    )
    
    print(f"   ✅ Model loaded: {model.num_parameters():,} parameters")
    
    # Tokenize data
    print("\n3️⃣ Tokenizing data...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=256,  # Shorter for faster training
            padding='max_length'
        )
    
    # Create dataset
    dataset_dict = {'text': texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"   Tokenizing {len(dataset):,} samples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Split into train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"   ✅ Train: {len(train_dataset):,} samples")
    print(f"   ✅ Validation: {len(eval_dataset):,} samples")
    
    # Training arguments
    print("\n4️⃣ Setting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        save_safetensors=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n5️⃣ Starting training...")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print(f"✅ Training completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Save final model
        final_output = f"{output_dir}/final_model"
        print(f"\n6️⃣ Saving final model to {final_output}...")
        trainer.save_model(final_output)
        tokenizer.save_pretrained(final_output)
        
        print(f"\n✅ Model saved successfully!")
        print(f"📁 Location: {final_output}")
        print(f"\n🎉 Training complete! You can now use this model for generation.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_model(f"{output_dir}/interrupted_checkpoint")
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train LLM with downloaded datasets')
    parser.add_argument('--quick', action='store_true', help='Use quick training data (20k samples)')
    parser.add_argument('--all', action='store_true', help='Use all downloaded datasets')
    parser.add_argument('--data', type=str, help='Specific data file to use')
    parser.add_argument('--model', type=str, default='distilgpt2', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--output', type=str, default='models/trained_on_datasets', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine data files
    data_files = []
    
    if args.quick:
        data_files = ['data/quick_training/quick_training_data.jsonl']
        print("🚀 Quick training mode - using fast dataset")
    elif args.all:
        # Use all downloaded datasets
        dataset_dir = 'data/downloaded_datasets'
        if os.path.exists(dataset_dir):
            data_files = [
                os.path.join(dataset_dir, f)
                for f in os.listdir(dataset_dir)
                if f.endswith('.jsonl')
            ]
        print(f"📚 Training with all {len(data_files)} datasets")
    elif args.data:
        data_files = [args.data]
        print(f"📖 Training with custom data: {args.data}")
    else:
        # Default: use quick if available, otherwise show error
        if os.path.exists('data/quick_training/quick_training_data.jsonl'):
            data_files = ['data/quick_training/quick_training_data.jsonl']
            print("📝 No option specified, using quick training data")
        else:
            print("❌ No data specified! Use --quick, --all, or --data")
            print("\nExamples:")
            print("  python train_with_downloaded_data.py --quick")
            print("  python train_with_downloaded_data.py --all")
            print("  python train_with_downloaded_data.py --data path/to/data.jsonl")
            sys.exit(1)
    
    if not data_files:
        print("❌ No data files found!")
        sys.exit(1)
    
    # Train
    train_model(
        data_files=data_files,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
