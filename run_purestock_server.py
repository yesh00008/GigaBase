#!/usr/bin/env python3
"""
Optimized Purestock Server Runner
Fast startup with model preloading
"""

import sys
import os

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from app.app import app

print("\n" + "="*70)
print("🚀 PURESTOCK SERVER - OPTIMIZED FOR SPEED")
print("="*70)
print("📍 URL: http://localhost:5000")
print("🎯 Model: Purestock (Preloaded for fast inference)")
print("⚡ Mode: Deterministic (Factual, anti-hallucination)")
print("="*70)
print("Press Ctrl+C to stop the server")
print("="*70 + "\n")

# Run server with optimizations
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug for speed
        use_reloader=False,  # Disable reloader to prevent crashes
        threaded=True  # Enable threading for concurrent requests
    )
