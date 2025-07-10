#!/usr/bin/env python3
"""
Test that websockets import is fixed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# This should not raise UnboundLocalError anymore
try:
    from openai_client.realtime import OpenAIRealtimeClient, WEBSOCKETS_AVAILABLE, WEBSOCKETS_VERSION
    print(f"Import successful!")
    print(f"Websockets available: {WEBSOCKETS_AVAILABLE}")
    print(f"Websockets version: {WEBSOCKETS_VERSION}")
    
    # Try to access websockets
    import openai_client.realtime as rt
    if hasattr(rt, 'websockets'):
        print(f"websockets module accessible: {rt.websockets is not None}")
    else:
        print("websockets not in module namespace (expected)")
        
except Exception as e:
    print(f"Import failed: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nImport fix successful! The UnboundLocalError should be resolved.")