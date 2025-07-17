#!/usr/bin/env python3
"""
Test GetLiveContext tool specifically to understand its response format

This script tests the GetLiveContext tool with various queries to analyze
the response structure and improve our parsing logic.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.services.ha_client.mcp_official import MCPClient


async def test_getlivecontext_tool():
    """Test GetLiveContext tool specifically"""
    print("=" * 60)
    print("Testing GetLiveContext Tool")
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
        
        # Find GetLiveContext tool
        tools = mcp_client.get_tools()
        getlivecontext_tool = None
        
        for tool in tools:
            if tool['name'] == 'GetLiveContext':
                getlivecontext_tool = tool
                break
        
        if not getlivecontext_tool:
            print("[ERROR] GetLiveContext tool not found!")
            return
        
        print(f"\n4. Found GetLiveContext tool:")
        print(f"   Description: {getlivecontext_tool['description']}")
        print(f"   Parameters: {getlivecontext_tool.get('inputSchema', 'None')}")
        
        # Test various queries
        test_queries = [
            # No parameters (as designed)
            None,
            # Simple queries
            "show all lights",
            "show all switches",
            "show all sensors",
            "list devices",
            "get status",
            # Empty dict
            {},
            # Query parameter
            {"query": "show all lights"},
            # Command parameter
            {"command": "show all lights"}
        ]
        
        print(f"\n5. Testing GetLiveContext with different parameters:")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 40)
            
            try:
                # Call GetLiveContext with different parameter formats
                if query is None:
                    # No parameters
                    result = await mcp_client.call_tool('GetLiveContext', {})
                elif isinstance(query, dict):
                    # Dict parameters
                    result = await mcp_client.call_tool('GetLiveContext', query)
                else:
                    # String query - try different parameter names
                    try:
                        result = await mcp_client.call_tool('GetLiveContext', {"query": query})
                    except:
                        result = await mcp_client.call_tool('GetLiveContext', {"command": query})
                
                print(f"SUCCESS! Got result:")
                print(f"  Type: {type(result)}")
                print(f"  Length: {len(str(result)) if result else 0} characters")
                
                # Detailed analysis of the response
                if result:
                    if isinstance(result, list):
                        print(f"  List with {len(result)} items")
                        for j, item in enumerate(result[:3]):  # Show first 3 items
                            print(f"    Item {j}: {type(item)} - {item}")
                    elif isinstance(result, dict):
                        print(f"  Dict with keys: {list(result.keys())}")
                        if 'content' in result:
                            print(f"    Content items: {len(result['content'])}")
                            for j, item in enumerate(result['content'][:3]):
                                print(f"      Content {j}: {item}")
                    elif hasattr(result, 'text'):
                        print(f"  Object with text: {result.text[:200]}...")
                    else:
                        print(f"  Raw result: {str(result)[:200]}...")
                    
                    # Try to extract structured data
                    print(f"  Attempting to extract device info...")
                    
                    if isinstance(result, list) and len(result) > 0:
                        for item in result:
                            if hasattr(item, 'text'):
                                text = item.text
                                print(f"    Found text: {text[:100]}...")
                                # Look for device patterns
                                if 'light.' in text or 'switch.' in text or 'sensor.' in text:
                                    print(f"    -> Contains device entities!")
                            elif isinstance(item, dict) and 'text' in item:
                                text = item['text']
                                print(f"    Found dict text: {text[:100]}...")
                                if 'light.' in text or 'switch.' in text or 'sensor.' in text:
                                    print(f"    -> Contains device entities!")
                
                # If this query worked, we found the right format!
                if result and any(keyword in str(result).lower() for keyword in ['light', 'switch', 'sensor', 'entity']):
                    print(f"  *** THIS QUERY FORMAT WORKS - CONTAINS DEVICE DATA! ***")
                    break
                    
            except Exception as e:
                print(f"  Failed: {type(e).__name__}: {e}")
        
        print(f"\n6. Summary:")
        print("   Look for the test marked with '*** THIS QUERY FORMAT WORKS ***'")
        print("   That shows the correct parameter format and response structure")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Disconnect
        print(f"\n7. Disconnecting...")
        try:
            await mcp_client.disconnect()
            print("[OK] Disconnected successfully")
        except Exception as e:
            print(f"[WARNING] Disconnect error: {e}")
    
    print("\n" + "=" * 60)
    print("GetLiveContext Test Complete")
    print("=" * 60)


async def main():
    """Main entry point"""
    try:
        await test_getlivecontext_tool()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting GetLiveContext tool to understand response format...\n")
    asyncio.run(main())