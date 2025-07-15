#!/usr/bin/env python3
"""
Debug WebSocket connection issue
"""
import asyncio
import websockets
import os
from dotenv import load_dotenv

async def debug_connection():
    """Debug OpenAI WebSocket connection"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    
    print(f"API Key found: {api_key[:10]}...")
    print(f"Websockets version: {websockets.__version__}")
    
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    # Test different connection methods
    attempts = [
        {"extra_headers": headers},
        {"additional_headers": headers},
        {"extra_headers": list(headers.items())},
    ]
    
    for i, params in enumerate(attempts):
        print(f"\nAttempt {i+1}: {list(params.keys())}")
        try:
            ws = await websockets.connect(url, **params)
            print(f"SUCCESS with method {i+1}!")
            await ws.close()
            return True
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {e}")
    
    # Try raw connection
    print("\nTrying raw connection with explicit headers...")
    try:
        import websockets.client
        ws = await websockets.client.connect(
            url,
            extra_headers=headers,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
            max_size=2**20,
            max_queue=2**5,
            read_limit=2**16,
            write_limit=2**16
        )
        print("SUCCESS with raw connection!")
        await ws.close()
        return True
    except Exception as e:
        print(f"Raw connection failed: {type(e).__name__}: {e}")
    
    return False

if __name__ == "__main__":
    success = asyncio.run(debug_connection())
    if not success:
        print("\nAll connection methods failed!")
        print("Check:")
        print("1. Your API key is valid")
        print("2. You have network connectivity")  
        print("3. The websockets library version is compatible")