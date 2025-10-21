#!/usr/bin/env python3
"""
Quick Dataset Downloader - Downloads essential datasets FAST

This downloads smaller, high-quality datasets first for quick training.
"""

import os
import sys
from datasets import load_dataset
import json
from tqdm import tqdm

# Quick & Essential Datasets (smaller, faster to download)
QUICK_DATASETS = [
    {
        'name': 'roneneldan/TinyStories',
        'config': 'default',
        'split': 'train',
        'num_samples': 10000,
        'description': 'Simple stories - Great for quick training'
    },
    {
        'name': 'zwhe99/DeepMath-103K',
        'config': 'default',
        'split': 'train',
        'num_samples': 5000,
        'description': 'Math problems - Reasoning skills'
    },
    {
        'name': 'HuggingFaceFW/fineweb-edu',
        'config': 'sample-10BT',
        'split': 'train',
        'num_samples': 5000,
        'description': 'Educational content - High quality'
    },
]

OUTPUT_DIR = 'data/quick_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_quick():
    """Download datasets quickly"""
    print("🚀 QUICK DATASET DOWNLOADER - FAST MODE")
    print("="*80)
    
    all_data = []
    total_samples = 0
    
    for i, config in enumerate(QUICK_DATASETS, 1):
        try:
            print(f"\n[{i}/{len(QUICK_DATASETS)}] 📥 {config['name']}")
            print(f"   {config['description']}")
            
            # Load dataset in streaming mode
            dataset = load_dataset(
                config['name'],
                config['config'],
                split=config['split'],
                streaming=True,
                trust_remote_code=True
            )
            
            # Take samples with progress bar
            samples = list(dataset.take(config['num_samples']))
            all_data.extend(samples)
            total_samples += len(samples)
            
            print(f"   ✅ Downloaded {len(samples):,} samples")
            
        except Exception as e:
            print(f"   ⚠️  Skipped: {str(e)}")
    
    # Save combined data
    output_file = os.path.join(OUTPUT_DIR, 'quick_training_data.jsonl')
    print(f"\n💾 Saving {total_samples:,} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Done! {total_samples:,} samples ready for training")
    print(f"📁 File: {output_file}")
    print(f"💡 Use this for quick testing before downloading larger datasets")
    
    return output_file

if __name__ == "__main__":
    download_quick()
