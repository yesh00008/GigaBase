#!/usr/bin/env python3
"""
Check Dataset Download Progress
"""

import os
import json
from datetime import datetime

def check_progress():
    """Check download progress and dataset status"""
    
    print("📊 DATASET DOWNLOAD STATUS")
    print("="*80)
    
    # Check quick training data
    quick_file = 'data/quick_training/quick_training_data.jsonl'
    if os.path.exists(quick_file):
        size = os.path.getsize(quick_file)
        print(f"\n✅ Quick Training Data: READY")
        print(f"   📁 File: {quick_file}")
        print(f"   💾 Size: {size/1024/1024:.2f} MB")
        print(f"   🚀 Ready to train NOW!")
    else:
        print(f"\n❌ Quick Training Data: NOT FOUND")
    
    # Check full datasets
    dataset_dir = 'data/downloaded_datasets'
    print(f"\n📚 Full Dataset Downloads:")
    print("-"*80)
    
    if not os.path.exists(dataset_dir):
        print("❌ Download directory not found")
        print("   Run: python download_all_datasets.py --parallel")
        return
    
    # List all downloaded files
    jsonl_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print("⏳ No datasets downloaded yet... Download in progress?")
    else:
        total_size = 0
        total_samples = 0
        
        for file in sorted(jsonl_files):
            file_path = os.path.join(dataset_dir, file)
            size = os.path.getsize(file_path)
            total_size += size
            
            # Count lines (samples)
            with open(file_path, 'r', encoding='utf-8') as f:
                num_samples = sum(1 for _ in f)
            total_samples += num_samples
            
            print(f"\n✅ {file}")
            print(f"   💾 Size: {size/1024/1024:.2f} MB")
            print(f"   📊 Samples: {num_samples:,}")
        
        print(f"\n{'='*80}")
        print(f"📈 TOTAL STATISTICS")
        print(f"   Datasets: {len(jsonl_files)}")
        print(f"   Total Size: {total_size/1024/1024:.2f} MB ({total_size/1024/1024/1024:.2f} GB)")
        print(f"   Total Samples: {total_samples:,}")
    
    # Check metadata
    metadata_file = os.path.join(dataset_dir, 'download_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📄 Download Metadata:")
        print(f"   Date: {metadata.get('download_date', 'N/A')}")
        print(f"   Duration: {metadata.get('duration_seconds', 0):.2f} seconds")
        print(f"   Successful: {metadata.get('successful', 0)}/{metadata.get('total_datasets', 0)}")
        
        if metadata.get('failed', 0) > 0:
            print(f"   ⚠️  Failed: {metadata['failed']}")
    
    print(f"\n{'='*80}")
    print("\n💡 Next Steps:")
    
    if os.path.exists(quick_file):
        print("   1. Train with quick data: python train_with_downloaded_data.py --quick")
    
    if jsonl_files:
        print("   2. Train with all data: python train_with_downloaded_data.py --all")
    
    print("\n")

if __name__ == "__main__":
    check_progress()
