#!/usr/bin/env python3
"""
Test SSL/HTTPS connection to Home Assistant MCP server

This script helps diagnose SSL-related issues with the MCP connection.
"""

import asyncio
import ssl
import sys
import aiohttp
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.ha_client.mcp import MCPClient
from src.config import load_config


async def test_ssl_connection():
    """Test SSL connection with various configurations"""
    print("=" * 60)
    print("Home Assistant MCP SSL Diagnostic Test")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    # Check if using HTTPS
    if not config.home_assistant.url.startswith('https'):
        print("\n[INFO] Not using HTTPS, SSL testing not applicable")
        return
    
    print(f"\n2. Testing HTTPS connection to: {config.home_assistant.url}")
    
    # Test 1: Basic HTTPS with default SSL
    print("\n3. Test with default SSL settings...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.home_assistant.url) as resp:
                print(f"   [OK] Status: {resp.status}")
                # Get SSL info
                if hasattr(resp.connection, 'transport'):
                    ssl_obj = resp.connection.transport.get_extra_info('ssl_object')
                    if ssl_obj:
                        print(f"   SSL Version: {ssl_obj.version()}")
                        print(f"   SSL Cipher: {ssl_obj.cipher()}")
    except aiohttp.ClientConnectorSSLError as e:
        print(f"   [ERROR] SSL Error: {e}")
    except Exception as e:
        print(f"   [ERROR] Connection failed: {e}")
    
    # Test 2: HTTPS without SSL verification
    print("\n4. Test with SSL verification DISABLED...")
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(config.home_assistant.url) as resp:
                print(f"   [OK] Status: {resp.status} (without SSL verification)")
    except Exception as e:
        print(f"   [ERROR] Connection failed even without SSL verification: {e}")
    
    # Test 3: MCP connection with SSL verification enabled
    print("\n5. Testing MCP connection WITH SSL verification...")
    client = MCPClient(
        base_url=config.home_assistant.url,
        access_token=config.home_assistant.token,
        sse_endpoint=config.home_assistant.mcp.sse_endpoint,
        connection_timeout=10,
        reconnect_attempts=1,
        ssl_verify=True
    )
    
    try:
        await client.connect()
        print("   [OK] MCP connection successful with SSL verification")
        await client.disconnect()
    except Exception as e:
        print(f"   [ERROR] MCP connection failed: {e}")
    
    # Test 4: MCP connection with SSL verification disabled
    print("\n6. Testing MCP connection WITHOUT SSL verification...")
    client_no_ssl = MCPClient(
        base_url=config.home_assistant.url,
        access_token=config.home_assistant.token,
        sse_endpoint=config.home_assistant.mcp.sse_endpoint,
        connection_timeout=10,
        reconnect_attempts=1,
        ssl_verify=False
    )
    
    try:
        await client_no_ssl.connect()
        print("   [OK] MCP connection successful WITHOUT SSL verification")
        print("   [WARNING] This indicates an SSL certificate issue!")
        await client_no_ssl.disconnect()
    except Exception as e:
        print(f"   [ERROR] MCP connection failed even without SSL: {e}")
    
    # Test 5: Check SSL certificate details
    print("\n7. SSL Certificate Information...")
    try:
        import socket
        from urllib.parse import urlparse
        
        parsed = urlparse(config.home_assistant.url)
        hostname = parsed.hostname
        port = parsed.port or 443
        
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                print(f"   Subject: {cert.get('subject', 'N/A')}")
                print(f"   Issuer: {cert.get('issuer', 'N/A')}")
                print(f"   Not Before: {cert.get('notBefore', 'N/A')}")
                print(f"   Not After: {cert.get('notAfter', 'N/A')}")
                print(f"   SubjectAltName: {cert.get('subjectAltName', 'N/A')}")
    except Exception as e:
        print(f"   [ERROR] Could not retrieve certificate: {e}")
    
    print("\n" + "=" * 60)
    print("SSL Diagnostic Complete")
    print("=" * 60)
    
    print("\nRecommendations:")
    print("1. If connection works without SSL but fails with SSL:")
    print("   - This is likely a certificate verification issue")
    print("   - Check if using self-signed certificates")
    print("   - Consider adding ssl_verify: false to config (FOR TESTING ONLY)")
    print("\n2. If connection fails in both cases:")
    print("   - Check if MCP server integration is installed")
    print("   - Verify Home Assistant version (2025.2+)")
    print("   - Check firewall/proxy settings")
    

async def main():
    """Main entry point"""
    try:
        await test_ssl_connection()
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting SSL diagnostic test...\n")
    asyncio.run(main())