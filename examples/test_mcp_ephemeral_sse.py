#!/usr/bin/env python3
"""
Test MCP with ephemeral SSE connections

This script tests the hypothesis that Home Assistant expects
a new SSE connection for each request/response cycle.
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


async def execute_mcp_request(session, base_url, token, sse_endpoint, request):
    """Execute a single MCP request with its own SSE connection"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    # Step 1: Open SSE connection
    sse_url = urljoin(base_url, sse_endpoint)
    print(f"\n[{request['id']}] Opening SSE connection...")
    
    async with session.get(sse_url, headers=headers) as sse_response:
        if sse_response.status != 200:
            print(f"[{request['id']}] Failed to open SSE: {sse_response.status}")
            return None
            
        # Step 2: Get endpoint
        message_endpoint = None
        buffer = ""
        
        try:
            async for chunk in sse_response.content:
                buffer += chunk.decode('utf-8', errors='replace')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip('\r')
                    
                    if line.startswith('data:') and line[5:].strip().startswith('/'):
                        message_endpoint = line[5:].strip()
                        print(f"[{request['id']}] Got endpoint: {message_endpoint}")
                        break
                    elif not line and message_endpoint:
                        break
                
                if message_endpoint:
                    break
        except Exception as e:
            print(f"[{request['id']}] Error getting endpoint: {e}")
            return None
        
        if not message_endpoint:
            print(f"[{request['id']}] No endpoint received")
            return None
        
        # Step 3: Send request
        message_url = urljoin(base_url, message_endpoint)
        print(f"[{request['id']}] Sending {request['method']} request...")
        
        # Start SSE reader for response
        response_future = asyncio.Future()
        
        async def read_response():
            """Read the response from SSE"""
            buffer = ""
            event_data = ""
            
            try:
                async for chunk in sse_response.content:
                    buffer += chunk.decode('utf-8', errors='replace')
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip('\r')
                        
                        if line.startswith('data:'):
                            if event_data:
                                event_data += " "
                            event_data += line[5:].strip()
                        elif not line and event_data:
                            # Try to parse response
                            try:
                                msg = json.loads(event_data)
                                if msg.get('id') == request['id']:
                                    print(f"[{request['id']}] Got response!")
                                    response_future.set_result(msg)
                                    return
                            except:
                                pass
                            event_data = ""
            except Exception as e:
                print(f"[{request['id']}] SSE closed: {type(e).__name__}")
                response_future.set_exception(e)
        
        # Start reader
        reader_task = asyncio.create_task(read_response())
        
        # Send POST request
        async with session.post(message_url, json=request, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }) as resp:
            if resp.status != 200:
                print(f"[{request['id']}] POST failed: {resp.status}")
                reader_task.cancel()
                return None
        
        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(response_future, timeout=5.0)
            print(f"[{request['id']}] Success! Response received")
            return response
        except asyncio.TimeoutError:
            print(f"[{request['id']}] Timeout waiting for response")
            reader_task.cancel()
            return None
        except Exception as e:
            print(f"[{request['id']}] Error: {e}")
            return None
        finally:
            if not reader_task.done():
                reader_task.cancel()


async def test_ephemeral_sse():
    """Test MCP with ephemeral SSE connections"""
    print("=" * 60)
    print("MCP Ephemeral SSE Test")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    base_url = config.home_assistant.url
    token = config.home_assistant.token
    sse_endpoint = config.home_assistant.mcp.sse_endpoint
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Initialize request
        print("\n2. Testing initialize request with new SSE connection...")
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
        
        init_response = await execute_mcp_request(session, base_url, token, sse_endpoint, init_request)
        if init_response:
            print("[OK] Initialize succeeded with ephemeral SSE!")
        else:
            print("[ERROR] Initialize failed")
            return
        
        # Test 2: Tools/list request with NEW SSE connection
        print("\n3. Testing tools/list request with new SSE connection...")
        tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": "tools-1"
        }
        
        tools_response = await execute_mcp_request(session, base_url, token, sse_endpoint, tools_request)
        if tools_response:
            print("[OK] Tools/list succeeded with ephemeral SSE!")
            result = tools_response.get('result', {})
            tools = result.get('tools', [])
            print(f"\nFound {len(tools)} tools:")
            for tool in tools[:5]:
                print(f"  - {tool.get('name', 'Unknown')}")
        else:
            print("[ERROR] Tools/list failed")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    if init_response and tools_response:
        print("\n[SUCCESS] Home Assistant MCP requires ephemeral SSE connections!")
        print("Each request/response cycle needs its own SSE connection.")
    else:
        print("\n[FAILURE] Ephemeral SSE pattern did not work.")


async def main():
    """Main entry point"""
    try:
        await test_ephemeral_sse()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting MCP with ephemeral SSE connections...\n")
    asyncio.run(main())