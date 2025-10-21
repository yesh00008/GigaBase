#!/usr/bin/env python3
"""
Main Script for LLM Training Pipeline

This script orchestrates the entire LLM training pipeline:
1. Collect data from various sources
2. Preprocess and clean the data
3. Train the model
"""

import argparse
import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    parser.add_argument("--collect-data", action="store_true", 
                        help="Collect data from internet sources")
    parser.add_argument("--preprocess-data", action="store_true", 
                        help="Preprocess collected data")
    parser.add_argument("--train-model", action="store_true", 
                        help="Train the model")
    parser.add_argument("--train-with-collected-data", action="store_true",
                        help="Train model specifically with collected data")
    parser.add_argument("--fast-train", action="store_true",
                        help="Train model with fast settings")
    parser.add_argument("--launch-frontend", action="store_true",
                        help="Launch the web frontend for testing models")
    parser.add_argument("--ultra-fast-train", action="store_true",
                        help="Train model with ultra-fast settings")
    parser.add_argument("--super-fast-train", action="store_true",
                        help="Train model with super-fast settings")
    parser.add_argument("--all", action="store_true", 
                        help="Run all steps")
    parser.add_argument("--num-articles", type=int, default=10, 
                        help="Number of Wikipedia articles to collect")
    parser.add_argument("--data-source", choices=["all", "wikipedia", "gutenberg", "stackexchange", "news", "code", "academic", "reddit", "documentation", "ai_ml_dl", "huggingface", "kaggle"], 
                        default="all", help="Specific data source to collect")
    parser.add_argument("--model", default="gpt2", help="Pre-trained model to fine-tune")
    
    args = parser.parse_args()
    
    # If --all is specified, run all steps
    if args.all:
        args.collect_data = True
        args.preprocess_data = True
        args.train_model = True
    
    # Check if any action is specified
    if not any([args.collect_data, args.preprocess_data, args.train_model, args.train_with_collected_data, args.fast_train, args.ultra_fast_train, args.super_fast_train, args.launch_frontend]):
        print("No action specified. Use --help for usage information.")
        return
    
    # Step 1: Collect data
    if args.collect_data:
        print("Step 1: Collecting data from internet sources...")
        if args.data_source == "all":
            cmd = f"python scripts/data_collection.py --num-articles {args.num_articles}"
        else:
            cmd = f"python scripts/data_collection.py --source {args.data_source} --num-articles {args.num_articles}"
        if not run_command(cmd, "Data Collection"):
            print("Data collection failed. Exiting.")
            return
    
    # Step 2: Preprocess data
    if args.preprocess_data:
        print("Step 2: Preprocessing collected data...")
        cmd = "python scripts/data_preprocessing.py --create-dataset"
        if not run_command(cmd, "Data Preprocessing"):
            print("Data preprocessing failed. Exiting.")
            return
    
    # Step 3: Train model
    if args.train_model:
        print("Step 3: Training the model...")
        cmd = f"python src/train.py --model {args.model} --generate"
        if not run_command(cmd, "Model Training"):
            print("Model training failed. Exiting.")
            return
    
    # Step 4: Train with collected data specifically
    if args.train_with_collected_data:
        print("Step 4: Training model with collected data...")
        cmd = f"python scripts/train_with_collected_data.py --model {args.model} --generate-sample"
        if not run_command(cmd, "Training with Collected Data"):
            print("Training with collected data failed. Exiting.")
            return
    
    # Step 5: Fast train with collected data
    if args.fast_train:
        print("Step 5: Fast training model with collected data...")
        cmd = f"python scripts/fast_train.py --model {args.model} --generate-sample"
        if not run_command(cmd, "Fast Training with Collected Data"):
            print("Fast training with collected data failed. Exiting.")
            return
    
    # Step 6: Ultra-fast train with collected data
    if args.ultra_fast_train:
        print("Step 6: Ultra-fast training model with collected data...")
        cmd = f"python scripts/ultra_fast_train.py --model {args.model} --generate-sample"
        if not run_command(cmd, "Ultra-Fast Training with Collected Data"):
            print("Ultra-fast training with collected data failed. Exiting.")
            return
    
    # Step 7: Super-fast train with collected data
    if args.super_fast_train:
        print("Step 7: Super-fast training model with collected data...")
        cmd = f"python scripts/super_fast_train.py --model {args.model} --generate-sample"
        if not run_command(cmd, "Super-Fast Training with Collected Data"):
            print("Super-fast training with collected data failed. Exiting.")
            return
    
    # Step 8: Launch the web frontend
    if args.launch_frontend:
        print("\n" + "="*50)
        print("Launching Web Frontend for Model Testing...")
        print("="*50)
        try:
            # Import the app dynamically to avoid circular imports
            from app.app import run_app
            run_app(debug=True)
        except ImportError as e:
            print(f"Error importing the web app: {e}")
            print("Make sure you have installed all the required dependencies:")
            print("pip install -r requirements.txt")
            return

    print("\n" + "="*50)
    print("LLM Training Pipeline Completed!")
    print("="*50)

if __name__ == "__main__":
    main()