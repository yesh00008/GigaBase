#!/usr/bin/env python3
"""Start Flask server with proper error handling"""

import sys
import os

# Set up paths
sys.path.insert(0, 'e:/LLM')

try:
    from app.app import app
    
    print("\n" + "="*60)
    print("🚀 STARTING PURESTOCK SERVER")
    print("="*60)
    print("URL: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Run with error handling
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,  # Disable reloader to prevent crashes
        threaded=True
    )
    
except Exception as e:
    print(f"\n❌ SERVER ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
