#!/usr/bin/env python3
"""
Multi-Dataset Downloader for LLM Training

This script downloads multiple datasets from Hugging Face efficiently
using the datasets library with multiprocessing for speed.
"""

import os
import sys
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

# Dataset configurations
DATASETS_CONFIG = [
    {
        'name': 'HuggingFaceFW/fineweb',
        'config': 'default',
        'split': 'train',
        'streaming': True,
        'num_samples': 10000,
        'description': 'FineWeb - High quality web text'
    },
    {
        'name': 'allenai/c4',
        'config': 'en',  # Using English instead of 'af'
        'split': 'train',
        'streaming': True,
        'num_samples': 10000,
        'description': 'C4 - Colossal Clean Crawled Corpus'
    },
    {
        'name': 'zwhe99/DeepMath-103K',
        'config': 'default',
        'split': 'train',
        'streaming': False,
        'num_samples': None,  # Download full dataset
        'description': 'DeepMath - Mathematical reasoning dataset'
    },
    {
        'name': 'nvidia/Nemotron-Personas',
        'config': 'default',
        'split': 'train',
        'streaming': True,
        'num_samples': 5000,
        'description': 'Nemotron Personas - Conversational data'
    },
    {
        'name': 'roneneldan/TinyStories',
        'config': 'default',
        'split': 'train',
        'streaming': True,
        'num_samples': 50000,
        'description': 'TinyStories - Simple story generation'
    },
    {
        'name': 'wikimedia/wikipedia',
        'config': '20231101.en',  # Using English Wikipedia
        'split': 'train',
        'streaming': True,
        'num_samples': 20000,
        'description': 'Wikipedia - Encyclopedia articles'
    },
    {
        'name': 'FreedomIntelligence/medical-o1-reasoning-SFT',
        'config': 'en',
        'split': 'train',
        'streaming': True,
        'num_samples': 5000,
        'description': 'Medical O1 Reasoning - Medical domain'
    },
    {
        'name': 'allenai/WildChat-1M',
        'config': 'default',
        'split': 'train',
        'streaming': True,
        'num_samples': 10000,
        'description': 'WildChat - Natural conversations'
    },
    {
        'name': 'HuggingFaceFW/fineweb-edu',
        'config': 'default',
        'split': 'train',
        'streaming': True,
        'num_samples': 15000,
        'description': 'FineWeb-Edu - Educational content'
    },
]

# Output directory
OUTPUT_DIR = 'data/downloaded_datasets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_dataset(dataset_config):
    """Download a single dataset"""
    try:
        name = dataset_config['name']
        config = dataset_config['config']
        split = dataset_config['split']
        streaming = dataset_config['streaming']
        num_samples = dataset_config['num_samples']
        description = dataset_config['description']
        
        print(f"\n{'='*80}")
        print(f"📥 Downloading: {name}")
        print(f"📝 Description: {description}")
        print(f"⚙️  Config: {config}, Split: {split}")
        print(f"{'='*80}")
        
        # Create safe filename
        safe_name = name.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(OUTPUT_DIR, f"{safe_name}_{config}.jsonl")
        
        # Load dataset
        if streaming:
            print(f"🔄 Loading in streaming mode (faster)...")
            dataset = load_dataset(
                name,
                config,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            
            # Take limited samples for streaming
            if num_samples:
                dataset = dataset.take(num_samples)
                print(f"📊 Taking {num_samples:,} samples...")
        else:
            print(f"💾 Loading full dataset...")
            dataset = load_dataset(
                name,
                config,
                split=split,
                trust_remote_code=True
            )
        
        # Save to JSONL format
        print(f"💾 Saving to {output_file}...")
        count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Saved {count:,} samples...", end='\r')
        
        print(f"✅ Successfully downloaded {count:,} samples from {name}")
        print(f"📁 Saved to: {output_file}")
        
        return {
            'name': name,
            'status': 'success',
            'samples': count,
            'file': output_file
        }
        
    except Exception as e:
        print(f"❌ Error downloading {name}: {str(e)}")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e)
        }

def download_all_datasets_parallel(max_workers=3):
    """Download all datasets in parallel"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MULTI-DATASET DOWNLOADER FOR LLM TRAINING                  ║
║                                                                              ║
║  Downloading {len(DATASETS_CONFIG)} datasets from Hugging Face                              ║
║  Using parallel downloads (max {max_workers} workers) for speed                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    start_time = datetime.now()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_dataset = {
            executor.submit(download_dataset, config): config 
            for config in DATASETS_CONFIG
        }
        
        # Process completed downloads
        for future in as_completed(future_to_dataset):
            result = future.result()
            results.append(result)
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    total_samples = sum(r.get('samples', 0) for r in successful)
    
    print(f"\n✅ Successful: {len(successful)}/{len(DATASETS_CONFIG)}")
    print(f"❌ Failed: {len(failed)}/{len(DATASETS_CONFIG)}")
    print(f"📈 Total samples downloaded: {total_samples:,}")
    print(f"⏱️  Total time: {duration:.2f} seconds")
    
    if successful:
        print(f"\n✅ Successfully Downloaded Datasets:")
        for r in successful:
            print(f"   • {r['name']}: {r['samples']:,} samples → {r['file']}")
    
    if failed:
        print(f"\n❌ Failed Downloads:")
        for r in failed:
            print(f"   • {r['name']}: {r['error']}")
    
    # Save metadata
    metadata_file = os.path.join(OUTPUT_DIR, 'download_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'download_date': start_time.isoformat(),
            'duration_seconds': duration,
            'total_datasets': len(DATASETS_CONFIG),
            'successful': len(successful),
            'failed': len(failed),
            'total_samples': total_samples,
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Metadata saved to: {metadata_file}")
    print(f"\n{'='*80}")
    print(f"🎉 Download complete! Ready for training.")
    print(f"{'='*80}\n")
    
    return results

def download_sequential():
    """Download datasets one by one (slower but more stable)"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SEQUENTIAL DATASET DOWNLOADER                              ║
║                                                                              ║
║  Downloading {len(DATASETS_CONFIG)} datasets one by one                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = []
    for i, config in enumerate(DATASETS_CONFIG, 1):
        print(f"\n[{i}/{len(DATASETS_CONFIG)}] Processing {config['name']}...")
        result = download_dataset(config)
        results.append(result)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download multiple datasets for LLM training')
    parser.add_argument('--parallel', action='store_true', help='Download in parallel (faster)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--sequential', action='store_true', help='Download one by one (more stable)')
    
    args = parser.parse_args()
    
    try:
        if args.sequential:
            results = download_sequential()
        else:
            # Default to parallel for speed
            results = download_all_datasets_parallel(max_workers=args.workers)
        
        # Check if we got any data
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            print(f"\n✅ Ready to train! {len(successful)} datasets available in: {OUTPUT_DIR}")
        else:
            print(f"\n⚠️  No datasets downloaded successfully. Check errors above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
