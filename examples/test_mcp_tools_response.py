#!/usr/bin/env python3
"""
Test MCP tools/list response specifically

This script focuses on debugging why the tools/list response
isn't being received before the SSE stream closes.
"""

import asyncio
import json
import sys
import aiohttp
from pathlib import Path
from urllib.parse import urljoin

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


async def test_tools_response():
    """Test the tools/list response pattern"""
    print("=" * 60)
    print("MCP Tools/List Response Test")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    base_url = config.home_assistant.url
    token = config.home_assistant.token
    sse_endpoint = config.home_assistant.mcp.sse_endpoint
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Connect to SSE and get endpoint
        print("\n2. Connecting to SSE endpoint...")
        sse_url = urljoin(base_url, sse_endpoint)
        
        async with session.get(sse_url, headers=headers) as sse_response:
            print(f"   SSE Status: {sse_response.status}")
            
            # Read the endpoint event
            message_endpoint = None
            buffer = ""
            event_data = ""
            
            print("\n3. Reading SSE events...")
            try:
                async for chunk in sse_response.content:
                    buffer += chunk.decode('utf-8', errors='replace')
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip('\r')
                        
                        if line.startswith('event:'):
                            event_type = line[6:].strip()
                            print(f"   Event type: {event_type}")
                        elif line.startswith('data:'):
                            event_data = line[5:].strip()
                            print(f"   Event data: {event_data}")
                        elif not line and event_data:
                            # End of event
                            if event_data.startswith('/'):
                                message_endpoint = event_data
                                print(f"\n[OK] Got message endpoint: {message_endpoint}")
                                break
                            event_data = ""
                    
                    if message_endpoint:
                        break
            except Exception as e:
                print(f"   Error reading SSE: {e}")
            
            if not message_endpoint:
                print("[ERROR] Failed to get message endpoint")
                return
            
            # Step 2: Send initialize request
            print("\n4. Sending initialize request...")
            message_url = urljoin(base_url, message_endpoint)
            
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                },
                "id": "init-1"
            }
            
            # Keep reading SSE in background
            responses = {}
            
            async def read_sse():
                """Keep reading SSE responses"""
                try:
                    event_data = ""
                    event_type = "message"
                    
                    async for chunk in sse_response.content:
                        buffer = chunk.decode('utf-8', errors='replace')
                        
                        for line in buffer.split('\n'):
                            line = line.strip('\r')
                            
                            if line.startswith('data:'):
                                if event_data:
                                    event_data += " "
                                event_data += line[5:].strip()
                            elif not line and event_data:
                                # Try to parse as JSON
                                try:
                                    msg = json.loads(event_data)
                                    if 'id' in msg:
                                        print(f"\n   [SSE] Received response for ID: {msg['id']}")
                                        print(f"   [SSE] Response preview: {json.dumps(msg)[:200]}...")
                                        responses[msg['id']] = msg
                                except:
                                    print(f"   [SSE] Non-JSON data: {event_data[:100]}")
                                event_data = ""
                except Exception as e:
                    print(f"\n   [SSE] Stream closed: {type(e).__name__}: {e}")
            
            # Start SSE reader
            sse_task = asyncio.create_task(read_sse())
            
            # Send initialize
            async with session.post(message_url, json=init_request, headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }) as resp:
                print(f"   POST Status: {resp.status}")
            
            # Wait for response
            await asyncio.sleep(1.0)
            
            if 'init-1' in responses:
                print("\n[OK] Got initialize response")
                
                # Step 3: Send tools/list request
                print("\n5. Sending tools/list request...")
                tools_request = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": "tools-1"
                }
                
                async with session.post(message_url, json=tools_request, headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }) as resp:
                    print(f"   POST Status: {resp.status}")
                
                # Wait for response
                await asyncio.sleep(2.0)
                
                if 'tools-1' in responses:
                    print("\n[OK] Got tools/list response!")
                    result = responses['tools-1'].get('result', {})
                    tools = result.get('tools', [])
                    print(f"   Found {len(tools)} tools")
                    for tool in tools[:3]:
                        print(f"   - {tool.get('name', 'Unknown')}")
                else:
                    print("\n[ERROR] No tools/list response received")
                    print("   Received responses:", list(responses.keys()))
            else:
                print("\n[ERROR] No initialize response received")
            
            # Cancel SSE reader
            sse_task.cancel()
            try:
                await sse_task
            except asyncio.CancelledError:
                pass
    
    print("\n" + "=" * 60)
    print("Test Complete")


async def main():
    """Main entry point"""
    try:
        await test_tools_response()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting MCP tools/list response pattern...\n")
    asyncio.run(main())