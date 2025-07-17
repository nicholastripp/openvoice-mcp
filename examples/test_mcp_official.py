#!/usr/bin/env python3
"""
Test Home Assistant MCP connection using the official MCP SDK

This script tests connecting to Home Assistant's MCP server
using the official mcp package with SSE transport.
"""

import asyncio
import sys
from pathlib import Path
import httpx

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


# Custom auth for Home Assistant bearer tokens
class BearerTokenAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token
    
    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


async def test_official_mcp():
    """Test connection using official MCP SDK with SSE transport"""
    print("=" * 60)
    print("Home Assistant MCP Test with Official SDK")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    print("\n2. Importing MCP SDK...")
    try:
        from mcp.client.sse import sse_client
        from mcp import ClientSession
        print("[OK] MCP SDK imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import MCP SDK: {e}")
        print("\nPlease install the SDK:")
        print("  pip install mcp httpx-sse")
        return
    
    # Build SSE URL
    sse_url = f"{config.home_assistant.url.rstrip('/')}{config.home_assistant.mcp.sse_endpoint}"
    print(f"\n3. Connecting to: {sse_url}")
    
    try:
        # Create SSE client with Home Assistant auth
        print("\n4. Creating SSE client with bearer token auth...")
        
        async with sse_client(
            url=sse_url,
            auth=BearerTokenAuth(config.home_assistant.token),
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            timeout=10.0,
            sse_read_timeout=30.0
        ) as (read_stream, write_stream):
            print("[OK] SSE client connected")
            
            # Create MCP session
            print("\n5. Creating MCP session...")
            async with ClientSession(read_stream, write_stream) as session:
                print("[OK] MCP session created")
                
                # Initialize the session
                print("\n6. Initializing MCP protocol...")
                try:
                    init_result = await session.initialize()
                    print(f"[OK] Initialized successfully")
                    print(f"   Protocol version: {init_result.protocolVersion}")
                    print(f"   Server capabilities: {init_result.capabilities}")
                except Exception as e:
                    print(f"[ERROR] Initialization failed: {e}")
                    return
                
                # Try to list tools
                print("\n7. Listing available tools...")
                try:
                    tools_result = await session.list_tools()
                    if tools_result and tools_result.tools:
                        print(f"[OK] Found {len(tools_result.tools)} tools:")
                        for tool in tools_result.tools[:5]:  # Show first 5
                            print(f"   - {tool.name}: {tool.description}")
                    else:
                        print("[WARNING] No tools found")
                except Exception as e:
                    print(f"[WARNING] Failed to list tools: {e}")
                    print("   This is a known issue with some Home Assistant versions")
                    print("   The integration can still work without tools")
                
                # Try to list resources if available
                print("\n8. Checking for resources...")
                try:
                    if hasattr(session, 'list_resources'):
                        resources = await session.list_resources()
                        print(f"[OK] Found {len(resources.resources) if resources else 0} resources")
                    else:
                        print("[INFO] Resources not available in this version")
                except Exception as e:
                    print(f"[INFO] Resources check: {e}")
                
                # Try to list prompts if available
                print("\n9. Checking for prompts...")
                try:
                    if hasattr(session, 'list_prompts'):
                        prompts = await session.list_prompts()
                        print(f"[OK] Found {len(prompts.prompts) if prompts else 0} prompts")
                    else:
                        print("[INFO] Prompts not available in this version")
                except Exception as e:
                    print(f"[INFO] Prompts check: {e}")
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if MCP Server integration is installed in Home Assistant")
        print("  2. Verify Home Assistant version is 2025.2 or later")
        print("  3. Check your access token permissions")
        print("  4. Ensure the SSE endpoint is accessible")
        
        import traceback
        print("\nFull error:")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


async def main():
    """Main entry point"""
    try:
        await test_official_mcp()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting Home Assistant MCP with official SDK...\n")
    asyncio.run(main())