#!/usr/bin/env python3
"""
List all available pretrained models

This script lists all the pretrained models available in the system.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_utils import get_available_models

def main():
    print("=" * 60)
    print("Available Pretrained Models from Hugging Face")
    print("=" * 60)
    
    models = get_available_models()
    
    print("\nPretrained Models:")
    print("-" * 30)
    pretrained_count = 0
    for model in models:
        if model['id'].startswith('pretrained/'):
            pretrained_count += 1
            print(f"{pretrained_count:2d}. {model['name']}")
            print(f"     ID: {model['id']}")
            print(f"     Hugging Face Path: {model['path']}")
            print()
    
    print(f"\nTotal pretrained models available: {pretrained_count}")
    print("\nNote: These models will be automatically downloaded from Hugging Face")
    print("on first use. Internet connection required for initial download.")
    
    print("\n" + "=" * 60)
    print("Custom Trained Models (if any):")
    print("-" * 30)
    custom_count = 0
    for model in models:
        if not model['id'].startswith('pretrained/'):
            custom_count += 1
            print(f"{custom_count:2d}. {model['name']}")
            print(f"     ID: {model['id']}")
            print()
    
    if custom_count == 0:
        print("No custom trained models found in the models/ directory.")
        print("Train a model first using the training scripts, or place")
        print("your trained models in the models/ directory.")
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("-" * 15)
    print("1. Launch the web interface:")
    print("   python main.py --launch-frontend")
    print()
    print("2. Test specific models:")
    print("   python test_pretrained.py")
    print()
    print("3. List models in the web interface:")
    print("   Visit http://localhost:5000 after launching the frontend")
    print("=" * 60)

if __name__ == "__main__":
    main()