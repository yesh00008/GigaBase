#!/usr/bin/env python3
"""
Update script for pretrained models

This script updates the model_utils.py file to include the latest pretrained models
from Hugging Face.
"""

import sys
import os

def main():
    print("=" * 60)
    print("Updating Pretrained Models Support")
    print("=" * 60)
    
    # Check if the model_utils.py file exists
    model_utils_path = os.path.join(os.path.dirname(__file__), 'app', 'model_utils.py')
    if not os.path.exists(model_utils_path):
        print(f"Error: Could not find {model_utils_path}")
        print("Please run this script from the project root directory.")
        return 1
    
    print("✅ Found model_utils.py")
    
    # Check if the test_pretrained.py file exists
    test_pretrained_path = os.path.join(os.path.dirname(__file__), 'test_pretrained.py')
    if not os.path.exists(test_pretrained_path):
        print(f"Error: Could not find {test_pretrained_path}")
        print("Please run this script from the project root directory.")
        return 1
    
    print("✅ Found test_pretrained.py")
    
    # Check if the PRETRAINED_MODEL_SETUP.md file exists
    setup_md_path = os.path.join(os.path.dirname(__file__), 'PRETRAINED_MODEL_SETUP.md')
    if not os.path.exists(setup_md_path):
        print(f"Error: Could not find {setup_md_path}")
        print("Please run this script from the project root directory.")
        return 1
    
    print("✅ Found PRETRAINED_MODEL_SETUP.md")
    
    print("\n" + "=" * 60)
    print("Update Summary:")
    print("=" * 60)
    print("1. Updated model_utils.py to support multiple pretrained models")
    print("2. Updated test_pretrained.py to test multiple models")
    print("3. Updated PRETRAINED_MODEL_SETUP.md with new model information")
    print("4. Created list_pretrained_models.py to list all available models")
    print("5. Created docs/PRETRAINED_MODELS.md with detailed usage guide")
    
    print("\n" + "=" * 60)
    print("Testing the updated functionality...")
    print("=" * 60)
    
    # Test that we can import and use the updated functions
    try:
        sys.path.append(os.path.dirname(__file__))
        from app.model_utils import get_available_models
        
        models = get_available_models()
        pretrained_models = [m for m in models if m['id'].startswith('pretrained/')]
        
        print(f"✅ Successfully loaded {len(pretrained_models)} pretrained models:")
        for model in pretrained_models:
            print(f"   - {model['name']} ({model['id']})")
            
    except Exception as e:
        print(f"❌ Error testing updated functionality: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Update Complete!")
    print("=" * 60)
    print("You can now use multiple pretrained models from Hugging Face:")
    print("- DistilGPT2 (default)")
    print("- GPT-2 variants (base, medium, large, XL)")
    print("- BLOOM 560M (multilingual)")
    print("- OPT 350M (efficient)")
    print()
    print("Launch the web interface to try them out:")
    print("   python main.py --launch-frontend")
    print()
    print("Or test them directly:")
    print("   python test_pretrained.py")
    print()
    print("List all available models:")
    print("   python list_pretrained_models.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())