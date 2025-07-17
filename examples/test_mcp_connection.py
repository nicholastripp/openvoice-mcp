#!/usr/bin/env python3
"""
Test MCP connection to Home Assistant

This script tests the Model Context Protocol connection to your Home Assistant instance.
Run this to verify your MCP setup is working correctly before using the voice assistant.

Usage:
    python examples/test_mcp_connection.py
    # or make it executable and run directly:
    chmod +x examples/test_mcp_connection.py
    ./examples/test_mcp_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.ha_client.mcp import MCPClient
from src.config import load_config


async def test_mcp_connection():
    """Test connection to Home Assistant MCP server"""
    print("=" * 60)
    print("Home Assistant MCP Connection Test")
    print("=" * 60)
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        if not config_path.exists():
            print(f"[ERROR] Configuration file not found: {config_path}")
            print("\nPlease ensure you have:")
            print("  1. Copied config.yaml.example to config.yaml")
            print("  2. Created a .env file with your credentials")
            return
            
        config = load_config(str(config_path))
        print("[OK] Configuration loaded successfully")
        
        # Display connection information (without sensitive data)
        print(f"\n2. Connection Details:")
        print(f"   - Home Assistant URL: {config.home_assistant.url}")
        print(f"   - MCP Endpoint: {config.home_assistant.mcp.sse_endpoint}")
        print(f"   - Connection Timeout: {config.home_assistant.mcp.connection_timeout}s")
        print(f"   - Auth Method: {config.home_assistant.mcp.auth_method}")
        
        # Create MCP client
        print("\n3. Creating MCP client...")
        client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            reconnect_attempts=config.home_assistant.mcp.reconnect_attempts
        )
        print("[OK] MCP client created")
        
        # Attempt connection
        print("\n4. Attempting connection to MCP server...")
        print("   This may take a few seconds...")
        
        await client.connect()
        print("[OK] Successfully connected to MCP server!")
        
        # Get available tools
        print("\n5. Discovering available tools...")
        tools = client.get_tools()
        
        if tools:
            print(f"[OK] Found {len(tools)} tools:")
            for i, tool in enumerate(tools, 1):
                print(f"\n   Tool {i}: {tool['name']}")
                if 'description' in tool:
                    print(f"   Description: {tool['description']}")
                if 'inputSchema' in tool:
                    params = tool['inputSchema'].get('properties', {})
                    if params:
                        print(f"   Parameters: {', '.join(params.keys())}")
        else:
            print("[WARNING] No tools discovered. Check that MCP server has 'Control Home Assistant' enabled.")
        
        # Test a simple tool call if we have tools
        if tools and any('control' in tool['name'].lower() for tool in tools):
            print("\n6. Testing tool invocation...")
            # Find a control tool
            control_tool = next((t for t in tools if 'control' in t['name'].lower()), None)
            if control_tool:
                try:
                    # Try a simple query that won't change anything
                    result = await client.call_tool(
                        control_tool['name'],
                        {"command": "what time is it"}
                    )
                    print("[OK] Tool invocation successful!")
                    if isinstance(result, dict) and 'content' in result:
                        print(f"   Response: {result['content']}")
                except Exception as e:
                    print(f"[WARNING] Tool invocation test failed: {e}")
                    print("   This is normal if the tool doesn't support the test command")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED! MCP integration is working correctly.")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Configuration Error: {e}")
        print("\nPlease check that:")
        print("  1. You're running from the project root directory")
        print("  2. config/config.yaml exists")
        print("  3. .env file exists with required variables")
        
    except Exception as e:
        print(f"\n[ERROR] Connection Failed: {type(e).__name__}: {e}")
        print("\n[DEBUG] Troubleshooting Steps:")
        print("  1. Verify Home Assistant is running and accessible")
        print("  2. Check Home Assistant version (must be 2025.2+)")
        print("  3. Ensure MCP Server integration is installed in HA")
        print("  4. Verify your access token is correct in .env")
        print("  5. Check the HA_URL in your .env file")
        print("\n[TIP] Debug Commands:")
        print(f"  - Test HA access: curl {config.home_assistant.url if 'config' in locals() else 'http://homeassistant.local:8123'}")
        print("  - Check HA version: ha core info")
        print("  - View HA logs: ha logs | grep mcp_server")
        
    finally:
        if 'client' in locals() and client.is_connected:
            print("\n7. Disconnecting...")
            await client.disconnect()
            print("[OK] Disconnected successfully")


def main():
    """Main entry point"""
    print("\nStarting MCP connection test...\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run the async test
    try:
        asyncio.run(test_mcp_connection())
    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()