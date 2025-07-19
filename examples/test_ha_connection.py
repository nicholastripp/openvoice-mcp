#!/usr/bin/env python3
"""
Test script for Home Assistant MCP (Model Context Protocol) connection

This script tests the MCP connection to Home Assistant, which replaced the
old Conversation API in v1.0.0. Requires Home Assistant 2025.2 or later
with the MCP Server integration installed and enabled.

Usage:
    ./venv/bin/python examples/test_ha_connection.py
    
Options:
    --config PATH     Configuration file path (default: config/config.yaml)
    --tools           List all available MCP tools
    --entities        Test entity discovery using GetLiveContext
    --execute TOOL    Execute a specific tool (with --args)
    --args JSON       Arguments for tool execution (JSON format)
    
Examples:
    # Basic connection test
    python examples/test_ha_connection.py
    
    # List available MCP tools
    python examples/test_ha_connection.py --tools
    
    # Test entity discovery
    python examples/test_ha_connection.py --entities
    
    # Execute a specific tool
    python examples/test_ha_connection.py --execute GetLiveContext --args '{}'

Note: Must be run from the project root using the virtual environment.
Requires config/config.yaml to be configured with your HA settings.
"""
import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from config import load_config
    from services.ha_client.mcp_official import MCPClient, MCPError, MCPConnectionError
    from utils.logger import setup_logging, get_logger
    from utils.text_utils import clean_entity_name, safe_str
except Exception as e:
    print(f"Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


async def test_connection(config_path: str) -> bool:
    """Test basic MCP connection to Home Assistant"""
    logger = get_logger("ha_voice_assistant.MCPTest")
    
    # Step 1: Load configuration
    try:
        logger.info("Step 1: Loading configuration...")
        config = load_config(config_path)
        logger.info("[OK] Configuration loaded successfully")
        
        # Log connection details (masked for security)
        masked_url = config.home_assistant.url
        masked_token = f"{config.home_assistant.token[:8]}...{config.home_assistant.token[-4:]}" if len(config.home_assistant.token) > 12 else "****"
        logger.info(f"HA URL: {masked_url}")
        logger.info(f"HA Token: {masked_token}")
        logger.info(f"MCP Endpoint: {config.home_assistant.mcp.sse_endpoint}")
        logger.info(f"SSL Verify: {config.home_assistant.mcp.ssl_verify}")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load configuration: {e}")
        logger.error(f"   Check that {config_path} exists and is valid YAML")
        return False
    
    # Step 2: Test basic connectivity
    try:
        logger.info("\nStep 2: Testing basic HTTP connectivity...")
        import httpx
        from urllib.parse import urlparse
        
        parsed_url = urlparse(config.home_assistant.url)
        test_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Test basic HTTP connectivity
        async with httpx.AsyncClient(verify=config.home_assistant.mcp.ssl_verify) as client:
            try:
                response = await client.get(test_url, timeout=5.0)
                logger.info(f"[OK] Basic connectivity OK (status: {response.status_code})")
            except httpx.ConnectError as e:
                logger.error(f"[ERROR] Connection failed: {e}")
                logger.error("   Check that the Home Assistant URL is correct and accessible")
                return False
            except httpx.TimeoutException:
                logger.error("[ERROR] Connection timeout")
                logger.error("   Home Assistant may be unreachable or very slow")
                return False
                
    except Exception as e:
        logger.error(f"[ERROR] Connectivity test failed: {e}")
        return False
    
    # Step 3: Create MCP client
    mcp_client = None
    try:
        logger.info("\nStep 3: Creating MCP client...")
        mcp_client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            ssl_verify=config.home_assistant.mcp.ssl_verify
        )
        logger.info("[OK] MCP client created successfully")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create MCP client: {e}")
        return False
    
    # Step 4: Connect to MCP server
    try:
        logger.info("\nStep 4: Connecting to MCP server...")
        await mcp_client.connect()
        logger.info("[OK] Successfully connected to MCP server")
        
        # Display capabilities
        capabilities = mcp_client._capabilities
        if capabilities:
            logger.info("Server capabilities:")
            for cap_type, cap_value in capabilities.items():
                if cap_value:
                    logger.info(f"  - {cap_type}: {cap_value}")
        
    except MCPConnectionError as e:
        logger.error(f"[ERROR] MCP connection failed: {e}")
        logger.error("   Check that the MCP Server integration is installed in Home Assistant")
        logger.error("   Verify your access token has proper permissions")
        return False
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("[ERROR] Authentication failed (401 Unauthorized)")
            logger.error("   Check that your HA token is valid")
        elif e.response.status_code == 404:
            logger.error("[ERROR] MCP endpoint not found (404)")
            logger.error("   The MCP Server integration may not be installed")
        else:
            logger.error(f"[ERROR] HTTP error {e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Connection failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False
    
    # Step 5: List available tools
    try:
        logger.info("\nStep 5: Listing available MCP tools...")
        tools = mcp_client.get_tools()
        
        if tools:
            logger.info(f"[OK] Found {len(tools)} tools:")
            for tool in tools[:5]:  # Show first 5 tools
                logger.info(f"  - {tool['name']}: {tool.get('description', 'No description')[:60]}...")
            if len(tools) > 5:
                logger.info(f"  ... and {len(tools) - 5} more tools")
        else:
            logger.warning("No tools found - this may indicate an MCP configuration issue")
            
    except Exception as e:
        logger.warning(f"Tool listing failed: {e}")
        logger.warning("This is often normal - Home Assistant MCP may not expose tools directly")
    
    # Step 6: Test basic tool execution
    try:
        logger.info("\nStep 6: Testing GetLiveContext tool...")
        
        # Try to execute GetLiveContext
        result = await mcp_client.call_tool("GetLiveContext", {})
        
        if result:
            logger.info("[OK] GetLiveContext executed successfully")
            
            # Parse result based on type
            if isinstance(result, list):
                for item in result:
                    if hasattr(item, 'text'):
                        logger.info(f"  Context: {item.text[:100]}...")
                        break
            elif isinstance(result, dict):
                logger.info(f"  Result type: {type(result)}")
                logger.info(f"  Keys: {list(result.keys())[:5]}")
            else:
                logger.info(f"  Result: {str(result)[:100]}...")
        else:
            logger.warning("GetLiveContext returned empty result")
            
    except MCPError as e:
        logger.warning(f"GetLiveContext failed: {e}")
        logger.warning("This tool may not be available in your HA version")
    except Exception as e:
        logger.warning(f"Tool execution failed: {e}")
    
    # Cleanup
    try:
        await mcp_client.disconnect()
        logger.info("\n[OK] MCP connection test completed successfully")
    except Exception as e:
        logger.warning(f"Disconnect warning: {e}")
    
    return True


async def test_tools(config_path: str) -> bool:
    """List and describe all available MCP tools"""
    logger = get_logger("ha_voice_assistant.MCPTools")
    
    try:
        config = load_config(config_path)
        mcp_client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            ssl_verify=config.home_assistant.mcp.ssl_verify
        )
        
        await mcp_client.connect()
        logger.info("Connected to MCP server")
        
        tools = mcp_client.get_tools()
        
        if tools:
            logger.info(f"\nAvailable MCP Tools ({len(tools)} total):")
            logger.info("=" * 70)
            
            for tool in tools:
                logger.info(f"\nTool: {tool['name']}")
                logger.info(f"Description: {tool.get('description', 'No description')}")
                
                # Show input schema if available
                schema = tool.get('inputSchema', {})
                if schema:
                    props = schema.get('properties', {})
                    if props:
                        logger.info("Parameters:")
                        for param, details in props.items():
                            param_type = details.get('type', 'any')
                            param_desc = details.get('description', '')
                            required = param in schema.get('required', [])
                            req_str = " (required)" if required else " (optional)"
                            logger.info(f"  - {param}: {param_type}{req_str}")
                            if param_desc:
                                logger.info(f"    {param_desc}")
                
                logger.info("-" * 70)
        else:
            logger.warning("No tools found")
            logger.info("\nThis may be normal - Home Assistant MCP implementation may not expose tools directly")
            logger.info("The integration still works through the OpenAI function calling bridge")
        
        await mcp_client.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Tool listing failed: {e}")
        return False


async def test_entities(config_path: str) -> bool:
    """Test entity discovery using MCP"""
    logger = get_logger("ha_voice_assistant.MCPEntities")
    
    try:
        config = load_config(config_path)
        mcp_client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            ssl_verify=config.home_assistant.mcp.ssl_verify
        )
        
        await mcp_client.connect()
        logger.info("Connected to MCP server")
        
        # Try GetLiveContext to get entity information
        logger.info("\nFetching live context (entities)...")
        
        try:
            result = await mcp_client.call_tool("GetLiveContext", {})
            
            if result:
                logger.info("Live context retrieved successfully")
                
                # Parse result - it may be a list of TextContent objects
                if isinstance(result, list):
                    for item in result:
                        if hasattr(item, 'text'):
                            context_text = item.text
                            
                            # Try to parse entities from context
                            lines = context_text.split('\n')
                            domains = {}
                            
                            for line in lines:
                                line = line.strip()
                                if '.' in line and ':' in line:
                                    # Looks like an entity state line
                                    entity_part = line.split(':', 1)[0].strip()
                                    if '.' in entity_part:
                                        domain = entity_part.split('.')[0]
                                        if domain not in domains:
                                            domains[domain] = 0
                                        domains[domain] += 1
                            
                            if domains:
                                logger.info(f"\nEntity domains found:")
                                for domain, count in sorted(domains.items()):
                                    logger.info(f"  {domain}: {count} entities")
                            
                            # Show sample of context
                            logger.info(f"\nSample context (first 500 chars):")
                            logger.info(context_text[:500] + "..." if len(context_text) > 500 else context_text)
                            
                            break
                else:
                    logger.info(f"Unexpected result type: {type(result)}")
                    logger.info(f"Result: {str(result)[:200]}")
            else:
                logger.warning("GetLiveContext returned empty result")
                
        except MCPError as e:
            logger.error(f"GetLiveContext failed: {e}")
            logger.info("\nAlternative: Use the Home Assistant REST API directly for entity discovery")
            logger.info("The MCP interface may not expose raw entity access")
        
        await mcp_client.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Entity test failed: {e}")
        return False


async def execute_tool(config_path: str, tool_name: str, args: Dict[str, Any]) -> bool:
    """Execute a specific MCP tool with arguments"""
    logger = get_logger("ha_voice_assistant.MCPExecute")
    
    try:
        config = load_config(config_path)
        mcp_client = MCPClient(
            base_url=config.home_assistant.url,
            access_token=config.home_assistant.token,
            sse_endpoint=config.home_assistant.mcp.sse_endpoint,
            connection_timeout=config.home_assistant.mcp.connection_timeout,
            ssl_verify=config.home_assistant.mcp.ssl_verify
        )
        
        await mcp_client.connect()
        logger.info(f"Connected to MCP server")
        
        logger.info(f"\nExecuting tool: {tool_name}")
        logger.info(f"Arguments: {json.dumps(args, indent=2)}")
        
        result = await mcp_client.call_tool(tool_name, args)
        
        logger.info("\nResult:")
        if isinstance(result, list):
            for item in result:
                if hasattr(item, 'text'):
                    logger.info(item.text)
                else:
                    logger.info(str(item))
        else:
            logger.info(json.dumps(result, indent=2) if isinstance(result, dict) else str(result))
        
        await mcp_client.disconnect()
        return True
        
    except MCPError as e:
        logger.error(f"Tool execution failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Home Assistant MCP connection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--tools", action="store_true", help="List all available MCP tools")
    parser.add_argument("--entities", action="store_true", help="Test entity discovery")
    parser.add_argument("--execute", metavar="TOOL", help="Execute a specific tool")
    parser.add_argument("--args", default="{}", help="Tool arguments as JSON (default: {})")
    
    args = parser.parse_args()
    
    # Setup logging
    try:
        setup_logging("INFO", console=True)
        logger = get_logger()
    except Exception as e:
        print(f"Logging setup failed: {e}", flush=True)
        return
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"[ERROR] Configuration file not found: {args.config}")
        logger.error("Please create the configuration file:")
        logger.error(f"  cp {args.config}.example {args.config}")
        logger.error("Then edit it with your Home Assistant settings.")
        return
    
    # Validate tool arguments if executing
    tool_args = {}
    if args.execute:
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Invalid JSON in --args: {e}")
            logger.error("Example: --args '{\"key\": \"value\"}'")
            return
    
    async def run_tests():
        try:
            if args.tools:
                await test_tools(str(config_path))
            elif args.entities:
                await test_entities(str(config_path))
            elif args.execute:
                await execute_tool(str(config_path), args.execute, tool_args)
            else:
                # Run basic connection test
                success = await test_connection(str(config_path))
                if not success:
                    logger.error("\n[FAILED] Connection test failed")
                    logger.info("\nTroubleshooting:")
                    logger.info("1. Verify Home Assistant 2025.2+ is installed")
                    logger.info("2. Check MCP Server integration is installed and enabled")
                    logger.info("3. Verify your access token has proper permissions")
                    logger.info("4. Check SSL settings if using self-signed certificates")
                    
        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Test failed with unexpected error: {e}")
            logger.debug("Full traceback:", exc_info=True)
    
    # Run tests
    try:
        asyncio.run(run_tests())
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()