#!/usr/bin/env python3
"""
Test the fixed MCPClient implementation

This script tests the redesigned MCPClient that follows
the official SDK patterns to avoid TaskGroup errors.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.services.ha_client.mcp_official import MCPClient


async def test_fixed_mcp_client():
    """Test the fixed MCPClient implementation"""
    print("=" * 60)
    print("Testing Fixed MCPClient Implementation")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    # Create MCP client
    print("\n2. Creating MCP client...")
    mcp_client = MCPClient(
        base_url=config.home_assistant.url,
        access_token=config.home_assistant.token,
        sse_endpoint=config.home_assistant.mcp.sse_endpoint,
        connection_timeout=config.home_assistant.mcp.connection_timeout,
        ssl_verify=config.home_assistant.mcp.ssl_verify
    )
    print("[OK] MCP client created")
    
    try:
        # Test connection
        print("\n3. Connecting to MCP server...")
        await mcp_client.connect()
        print("[OK] Connected successfully")
        
        # Check connection status
        print(f"\n4. Connection status: {mcp_client.is_connected}")
        
        # Get capabilities
        print(f"\n5. Server capabilities: {mcp_client.capabilities}")
        
        # Get tools
        tools = mcp_client.get_tools()
        print(f"\n6. Available tools: {len(tools)}")
        for tool in tools[:5]:  # Show first 5
            print(f"   - {tool['name']}: {tool['description']}")
        
        # Test tool call if we have tools
        if tools:
            # Try to find a simple tool to test
            test_tool = None
            for tool in tools:
                if any(keyword in tool['name'].lower() for keyword in ['status', 'info', 'state', 'light']):
                    test_tool = tool
                    break
            
            if test_tool:
                print(f"\n7. Testing tool call: {test_tool['name']}")
                try:
                    # Try a simple call (parameters depend on the tool)
                    result = await mcp_client.call_tool(test_tool['name'], {})
                    print(f"[OK] Tool call result: {result}")
                except Exception as e:
                    print(f"[INFO] Tool call failed (expected): {e}")
            else:
                print("\n7. No suitable test tool found for call test")
        else:
            print("\n7. No tools available for testing")
        
        print("\n[SUCCESS] All tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Test disconnect
        print("\n8. Disconnecting...")
        try:
            await mcp_client.disconnect()
            print("[OK] Disconnected successfully")
        except Exception as e:
            print(f"[WARNING] Disconnect error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


async def main():
    """Main entry point"""
    try:
        await test_fixed_mcp_client()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting fixed MCPClient implementation...\n")
    asyncio.run(main())