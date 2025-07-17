#!/usr/bin/env python3
"""
Test all available MCP tools from Home Assistant

This script discovers and tests all MCP tools to understand
what functionality is available for device state queries.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.services.ha_client.mcp_official import MCPClient


async def test_all_mcp_tools():
    """Discover and analyze all available MCP tools"""
    print("=" * 60)
    print("Discovering All Home Assistant MCP Tools")
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
        # Connect
        print("\n3. Connecting to MCP server...")
        await mcp_client.connect()
        print("[OK] Connected successfully")
        
        # Get all tools
        tools = mcp_client.get_tools()
        print(f"\n4. Found {len(tools)} tools:")
        print("=" * 60)
        
        # Analyze each tool
        state_related_tools = []
        control_tools = []
        other_tools = []
        
        for i, tool in enumerate(tools, 1):
            print(f"\n{i}. {tool['name']}")
            print(f"   Description: {tool['description']}")
            
            # Show input schema if available
            if 'inputSchema' in tool and tool['inputSchema']:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    print("   Parameters:")
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', 'No description')
                        required = param_name in schema.get('required', [])
                        req_text = " (required)" if required else " (optional)"
                        print(f"     - {param_name}: {param_type}{req_text} - {param_desc}")
            else:
                print("   Parameters: None")
            
            # Categorize tools
            name_lower = tool['name'].lower()
            desc_lower = tool['description'].lower()
            
            if any(keyword in name_lower or keyword in desc_lower 
                   for keyword in ['state', 'get', 'list', 'show', 'status', 'info']):
                state_related_tools.append(tool)
            elif any(keyword in name_lower or keyword in desc_lower 
                     for keyword in ['turn', 'set', 'control', 'command', 'activate']):
                control_tools.append(tool)
            else:
                other_tools.append(tool)
        
        # Summary analysis
        print("\n" + "=" * 60)
        print("TOOL ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 CATEGORIES:")
        print(f"   State/Query tools: {len(state_related_tools)}")
        print(f"   Control tools: {len(control_tools)}")
        print(f"   Other tools: {len(other_tools)}")
        
        if state_related_tools:
            print(f"\n🔍 STATE/QUERY TOOLS:")
            for tool in state_related_tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print(f"\n⚠️  NO DIRECT STATE QUERY TOOLS FOUND")
            print("   This means we cannot directly fetch device states via MCP")
        
        print(f"\n🎮 CONTROL TOOLS:")
        for tool in control_tools[:5]:  # Show first 5
            print(f"   - {tool['name']}: {tool['description']}")
        if len(control_tools) > 5:
            print(f"   ... and {len(control_tools) - 5} more")
        
        # Test if any tools might provide state information
        print(f"\n🧪 TESTING FOR HIDDEN STATE CAPABILITIES:")
        
        # Look for tools that might respond to state queries
        test_tools = []
        for tool in tools:
            name_lower = tool['name'].lower()
            if any(keyword in name_lower for keyword in ['process', 'command', 'handle', 'execute']):
                test_tools.append(tool)
        
        if test_tools:
            print(f"   Found {len(test_tools)} tools that might handle natural language queries:")
            for tool in test_tools:
                print(f"   - {tool['name']}: {tool['description']}")
                
                # Test with a simple state query
                try:
                    print(f"   Testing {tool['name']} with state query...")
                    result = await mcp_client.call_tool(tool['name'], {
                        "command": "show me all lights"
                    })
                    if result:
                        print(f"   ✓ Got response: {result}")
                        break
                except Exception as e:
                    print(f"   ✗ Failed: {e}")
        else:
            print("   No tools found that might handle natural language state queries")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if not state_related_tools:
            print("   1. MCP currently doesn't provide direct state access")
            print("   2. Consider hybrid approach: MCP for control, REST API for state queries")
            print("   3. Or use natural language queries via control tools")
            print("   4. Wait for Home Assistant to add state query tools to MCP")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Disconnect
        print(f"\n5. Disconnecting...")
        try:
            await mcp_client.disconnect()
            print("[OK] Disconnected successfully")
        except Exception as e:
            print(f"[WARNING] Disconnect error: {e}")
    
    print("\n" + "=" * 60)
    print("Tool Discovery Complete")
    print("=" * 60)


async def main():
    """Main entry point"""
    try:
        await test_all_mcp_tools()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nDiscovering all MCP tools from Home Assistant...\n")
    asyncio.run(main())