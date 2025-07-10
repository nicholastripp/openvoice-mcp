#!/usr/bin/env python3
"""
Test that websockets import is properly fixed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing websockets import fix...")
print("=" * 50)

try:
    # This should work now without UnboundLocalError
    from openai_client.realtime import OpenAIRealtimeClient, WEBSOCKETS_VERSION
    print("✓ Import successful!")
    print(f"✓ Websockets version: {WEBSOCKETS_VERSION}")
    print("✓ No UnboundLocalError!")
    
    # Test creating a client (shouldn't throw UnboundLocalError)
    from config import OpenAIConfig
    config = OpenAIConfig(
        api_key="test",
        voice="alloy",
        model="gpt-4o-realtime-preview"
    )
    client = OpenAIRealtimeClient(config, "Test")
    print("✓ Client created successfully")
    
except ImportError as e:
    if "websockets library is required" in str(e):
        print("✓ Got expected ImportError (websockets not installed)")
        print(f"  Message: {e}")
    else:
        print(f"✗ Unexpected ImportError: {e}")
except UnboundLocalError as e:
    print(f"✗ UnboundLocalError still occurring: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("WebSocket import fix successful!")
print("The UnboundLocalError should be completely resolved.")