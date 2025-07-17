#!/usr/bin/env python3
"""
Test MCP connection with manual SSE parsing

This script tests the updated MCP client that uses manual SSE parsing
instead of the aiohttp-sse-client library.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.ha_client.mcp import MCPClient
from src.config import load_config


async def test_mcp_manual_sse():
    """Test MCP connection with manual SSE parsing"""
    print("=" * 60)
    print("Home Assistant MCP Manual SSE Test")
    print("=" * 60)
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config = load_config("config/config.yaml")
        print("[OK] Configuration loaded")
        
        # Display connection details
        print(f"\n2. Connection Details:")
        print(f"   - Home Assistant URL: {config.home_assistant.url}")
        print(f"   - MCP Endpoint: {config.home_assistant.mcp.sse_endpoint}")
        print(f"   - Using manual SSE parsing (no aiohttp-sse-client)")
        
        # Create MCP client
        print("\n3. Creating MCP client with manual SSE...")
        client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            reconnect_attempts=config.home_assistant.mcp.reconnect_attempts,
            ssl_verify=config.home_assistant.mcp.ssl_verify,
            ssl_ca_bundle=config.home_assistant.mcp.ssl_ca_bundle
        )
        print("[OK] MCP client created")
        
        # Enable debug logging for detailed output
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Test connection
        print("\n4. Testing MCP connection with manual SSE parsing...")
        print("   This will use our custom SSE parser instead of aiohttp-sse-client")
        print("   Watch for SSE event parsing in the debug output...")
        
        await client.connect()
        print("\n[OK] Successfully connected using manual SSE parsing!")
        
        # Get available tools
        print("\n5. Testing tool discovery...")
        tools = client.get_tools()
        
        if tools:
            print(f"[OK] Found {len(tools)} tools:")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"   - {tool['name']}")
        else:
            print("[WARNING] No tools discovered")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Manual SSE implementation is working!")
        print("=" * 60)
        print("\nThe MCP client is now using manual SSE parsing instead of")
        print("the aiohttp-sse-client library. This should resolve the")
        print("ValueError that was preventing connection.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {type(e).__name__}: {e}")
        print("\n[DEBUG] This might mean:")
        print("  1. The manual SSE parsing has a bug")
        print("  2. The SSE format from HA is different than expected")
        print("  3. Network/authentication issues")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
    finally:
        if 'client' in locals() and client.is_connected:
            print("\n6. Disconnecting...")
            await client.disconnect()
            print("[OK] Disconnected")


async def main():
    """Main entry point"""
    try:
        await test_mcp_manual_sse()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting manual SSE test...\n")
    asyncio.run(main())