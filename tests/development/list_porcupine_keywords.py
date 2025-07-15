#!/usr/bin/env python3
"""
List all available built-in Porcupine wake words
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pvporcupine
    print("Porcupine SDK Version:", pvporcupine.__version__ if hasattr(pvporcupine, '__version__') else "Unknown")
    print("\nAvailable built-in keywords:")
    print("-" * 40)
    
    # Get list of available keywords
    if hasattr(pvporcupine, 'KEYWORDS'):
        keywords = sorted(pvporcupine.KEYWORDS)
        for i, keyword in enumerate(keywords, 1):
            print(f"{i:2d}. {keyword}")
        print(f"\nTotal: {len(keywords)} built-in keywords")
    else:
        print("ERROR: pvporcupine.KEYWORDS not found")
        print("This may be an older version of the SDK")
    
    # Also check for KEYWORD_PATHS if available
    if hasattr(pvporcupine, 'KEYWORD_PATHS'):
        print("\n\nKeyword paths available:")
        print("-" * 40)
        for keyword, path in sorted(pvporcupine.KEYWORD_PATHS.items()):
            print(f"{keyword}: {path}")
            
except ImportError as e:
    print(f"ERROR: Could not import pvporcupine: {e}")
    print("Make sure Porcupine is installed: pip install pvporcupine")
except Exception as e:
    print(f"ERROR: {e}")