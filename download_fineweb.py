#!/usr/bin/env python3
"""
Download FineWeb Dataset from Hugging Face

This script downloads the FineWeb dataset which is a large-scale web dataset
suitable for training language models.
"""

import os
import sys
import json
import requests
from tqdm import tqdm

def download_fineweb_sample(output_dir="data/raw", num_samples=1000):
    """
    Download samples from the FineWeb dataset
    
    Args:
        output_dir: Directory to save the downloaded data
        num_samples: Number of samples to download
    """
    print("=" * 60)
    print("Downloading FineWeb Dataset from Hugging Face")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Hugging Face datasets server API endpoint
    base_url = "https://datasets-server.huggingface.co/rows"
    
    # Parameters for the request
    params = {
        "dataset": "HuggingFaceFW/fineweb",
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100  # Number of rows per request
    }
    
    all_data = []
    total_downloaded = 0
    batch_size = 100
    
    print(f"\nDownloading {num_samples} samples...")
    print(f"Saving to: {output_dir}/fineweb_data.jsonl\n")
    
    # Download in batches
    with tqdm(total=num_samples, desc="Downloading") as pbar:
        while total_downloaded < num_samples:
            try:
                # Update offset for pagination
                params["offset"] = total_downloaded
                params["length"] = min(batch_size, num_samples - total_downloaded)
                
                # Make request
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract rows from response
                    if "rows" in data:
                        rows = data["rows"]
                        
                        if not rows:
                            print("\n⚠️  No more data available")
                            break
                        
                        # Process each row
                        for row in rows:
                            # Extract the actual content
                            if "row" in row:
                                content = row["row"]
                                all_data.append(content)
                        
                        total_downloaded += len(rows)
                        pbar.update(len(rows))
                        
                    else:
                        print(f"\n❌ Unexpected response format: {data}")
                        break
                        
                else:
                    print(f"\n❌ Error: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error downloading batch: {str(e)}")
                break
    
    # Save to JSONL file
    output_file = os.path.join(output_dir, "fineweb_data.jsonl")
    
    print(f"\n💾 Saving {len(all_data)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully downloaded and saved {len(all_data)} samples!")
    
    # Also save as plain text for easy reading
    text_output = os.path.join(output_dir, "fineweb_data.txt")
    print(f"\n💾 Saving as plain text to {text_output}...")
    
    with open(text_output, 'w', encoding='utf-8') as f:
        for item in all_data:
            # Extract text content if available
            if isinstance(item, dict) and 'text' in item:
                f.write(item['text'] + '\n\n---\n\n')
            else:
                f.write(str(item) + '\n\n---\n\n')
    
    print(f"✅ Plain text saved!")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Download Statistics")
    print("=" * 60)
    print(f"Total samples downloaded: {len(all_data)}")
    
    if all_data and isinstance(all_data[0], dict):
        print(f"Fields in data: {list(all_data[0].keys())}")
    
    # Show a sample
    if all_data:
        print("\n📝 Sample entry:")
        print("-" * 60)
        sample = all_data[0]
        if isinstance(sample, dict) and 'text' in sample:
            text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
            print(f"Text: {text_preview}")
            print(f"URL: {sample.get('url', 'N/A')}")
        else:
            print(json.dumps(sample, indent=2)[:300])
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    return output_file

def download_using_datasets_library(output_dir="data/raw", num_samples=1000):
    """
    Alternative method: Download using the datasets library (more efficient)
    """
    try:
        from datasets import load_dataset
        
        print("=" * 60)
        print("Downloading FineWeb using Hugging Face Datasets Library")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nLoading {num_samples} samples from FineWeb...")
        
        # Load dataset with streaming for efficiency
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="default",
            split=f"train[:{num_samples}]",
            streaming=False
        )
        
        print(f"✅ Loaded {len(dataset)} samples!")
        
        # Save to JSONL
        output_file = os.path.join(output_dir, "fineweb_data.jsonl")
        dataset.to_json(output_file)
        
        print(f"✅ Saved to {output_file}")
        
        # Save as text
        text_output = os.path.join(output_dir, "fineweb_data.txt")
        with open(text_output, 'w', encoding='utf-8') as f:
            for item in dataset:
                if 'text' in item:
                    f.write(item['text'] + '\n\n---\n\n')
        
        print(f"✅ Plain text saved to {text_output}")
        
        return output_file
        
    except ImportError:
        print("⚠️  'datasets' library not installed. Using API method instead.")
        return None
    except Exception as e:
        print(f"❌ Error using datasets library: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FineWeb dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to download (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--method", choices=["api", "library", "auto"], default="auto",
                        help="Download method (default: auto)")
    
    args = parser.parse_args()
    
    if args.method == "library":
        result = download_using_datasets_library(args.output_dir, args.num_samples)
        if not result:
            print("\nFalling back to API method...")
            download_fineweb_sample(args.output_dir, args.num_samples)
    elif args.method == "api":
        download_fineweb_sample(args.output_dir, args.num_samples)
    else:  # auto
        # Try library first, fall back to API
        result = download_using_datasets_library(args.output_dir, args.num_samples)
        if not result:
            download_fineweb_sample(args.output_dir, args.num_samples)
