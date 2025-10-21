#!/usr/bin/env python3
"""
Simple Flask server runner (NO DEBUG MODE)
This helps see actual errors without debug reload issues
"""

import os
import sys

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import app properly
sys.path.insert(0, 'app')
from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING PURESTOCK SERVER (NO DEBUG)")
    print("="*60)
    print(f"URL: http://localhost:5000")
    print(f"Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Run WITHOUT debug mode to see errors
    app.run(
        debug=False,  # NO DEBUG MODE
        host='0.0.0.0',
        port=5000
    )
