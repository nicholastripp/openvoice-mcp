#!/usr/bin/env python3
"""
Test raw SSE endpoint response from Home Assistant MCP server

This script makes direct HTTP requests to see what the MCP endpoint actually returns.
"""

import asyncio
import sys
import aiohttp
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


async def test_raw_sse():
    """Test raw SSE endpoint response"""
    print("=" * 60)
    print("Home Assistant MCP Raw SSE Test")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    # Build SSE URL
    sse_url = config.home_assistant.url.rstrip('/') + config.home_assistant.mcp.sse_endpoint
    print(f"\n2. Testing SSE endpoint: {sse_url}")
    
    headers = {
        "Authorization": f"Bearer {config.home_assistant.token}",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache"
    }
    
    # Test 1: Basic GET request to see response
    print("\n3. Making basic GET request to SSE endpoint...")
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(sse_url, headers=headers) as response:
                print(f"   Status: {response.status}")
                print(f"   Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
                print(f"   Headers: {dict(response.headers)}")
                
                # Read first chunk of response
                chunk = await response.content.read(500)
                text = chunk.decode('utf-8', errors='replace')
                
                print(f"\n   First 500 chars of response:")
                print("   " + "-" * 50)
                print(f"   {text}")
                print("   " + "-" * 50)
                
                # Check if it looks like HTML
                if text.strip().startswith('<!DOCTYPE') or text.strip().startswith('<html'):
                    print("\n   [ERROR] Response is HTML, not SSE!")
                    print("   This suggests MCP endpoint is not properly configured.")
                elif text.strip().startswith('{'):
                    print("\n   [WARNING] Response looks like JSON, not SSE format")
                elif 'data:' in text or 'event:' in text:
                    print("\n   [OK] Response appears to be SSE format")
                else:
                    print("\n   [WARNING] Response format unclear")
                    
        except asyncio.TimeoutError:
            print("   [ERROR] Request timed out")
        except Exception as e:
            print(f"   [ERROR] Request failed: {type(e).__name__}: {e}")
    
    # Test 2: Try streaming SSE
    print("\n4. Testing SSE streaming (5 seconds)...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(sse_url, headers=headers) as response:
                if response.status != 200:
                    print(f"   [ERROR] Status {response.status}")
                    body = await response.text()
                    print(f"   Response: {body[:200]}")
                    return
                
                print("   Waiting for SSE events...")
                start_time = asyncio.get_event_loop().time()
                event_count = 0
                
                async for data in response.content:
                    if asyncio.get_event_loop().time() - start_time > 5:
                        break
                        
                    line = data.decode('utf-8', errors='replace').strip()
                    if line:
                        event_count += 1
                        print(f"   Event {event_count}: {line[:100]}...")
                        
                        # Parse SSE format
                        if line.startswith('data:'):
                            print(f"      -> Data: {line[5:].strip()[:80]}...")
                        elif line.startswith('event:'):
                            print(f"      -> Event type: {line[6:].strip()}")
                        elif line.startswith(':'):
                            print(f"      -> Comment: {line[1:].strip()}")
                
                if event_count == 0:
                    print("   [WARNING] No SSE events received in 5 seconds")
                else:
                    print(f"   [OK] Received {event_count} events")
                    
        except Exception as e:
            print(f"   [ERROR] Streaming failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Check if it's a WebSocket endpoint instead
    print("\n5. Checking if endpoint expects WebSocket...")
    ws_url = sse_url.replace('http://', 'ws://').replace('https://', 'wss://')
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                ws_url, 
                headers={"Authorization": f"Bearer {config.home_assistant.token}"},
                timeout=5
            ) as ws:
                print("   [WARNING] Endpoint accepted WebSocket connection!")
                print("   MCP should use SSE, not WebSocket")
                await ws.close()
    except Exception as e:
        print(f"   [OK] Not a WebSocket endpoint (expected): {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    print("\nDiagnosis:")
    print("- If response is HTML: MCP integration not properly configured")
    print("- If no SSE events: Check MCP server settings in HA")
    print("- If authentication error: Check access token permissions")


async def main():
    """Main entry point"""
    try:
        await test_raw_sse()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting raw SSE test...\n")
    asyncio.run(main())