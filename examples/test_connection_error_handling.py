#!/usr/bin/env python3
"""
Test script to verify Home Assistant connection error handling

This script intentionally uses invalid connection details to test
the error handling and user-friendly error messages.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config, HomeAssistantConfig
from ha_client.conversation import HomeAssistantConversationClient
from utils.logger import setup_logging


async def test_connection_errors():
    """Test various connection error scenarios"""
    # Setup logging
    logger = setup_logging(level="INFO", console=True)
    
    print("Testing Home Assistant Connection Error Handling")
    print("=" * 50)
    
    # Test 1: Invalid URL format
    print("\nTest 1: Invalid URL format")
    print("-" * 30)
    try:
        config = HomeAssistantConfig(
            url="not-a-valid-url",
            token="dummy-token",
            language="en",
            timeout=5
        )
        client = HomeAssistantConversationClient(config)
        await client.start()
    except Exception as e:
        print(f"Error caught (as expected): {type(e).__name__}")
        print(f"Message: {e}")
    
    # Test 2: Unreachable host
    print("\n\nTest 2: Unreachable host")
    print("-" * 30)
    try:
        config = HomeAssistantConfig(
            url="http://192.168.99.99:8123",  # Likely unreachable IP
            token="dummy-token",
            language="en",
            timeout=5
        )
        client = HomeAssistantConversationClient(config)
        await client.start()
    except Exception as e:
        print(f"Error caught (as expected): {type(e).__name__}")
        print(f"Message: {e}")
    
    # Test 3: Invalid token (requires real HA instance)
    print("\n\nTest 3: Invalid token (if HA is reachable)")
    print("-" * 30)
    try:
        # Try to load actual config first
        actual_config = load_config()
        config = HomeAssistantConfig(
            url=actual_config.home_assistant.url,
            token="invalid-token-12345",
            language="en",
            timeout=5
        )
        client = HomeAssistantConversationClient(config)
        await client.start()
    except FileNotFoundError:
        print("Skipping - no config file found")
    except Exception as e:
        print(f"Error caught (as expected): {type(e).__name__}")
        print(f"Message: {e}")
    
    # Test 4: Test the test_connection method directly
    print("\n\nTest 4: Direct test_connection method")
    print("-" * 30)
    try:
        config = HomeAssistantConfig(
            url="http://homeassistant.local:8123",
            token="test-token",
            language="en",
            timeout=5
        )
        client = HomeAssistantConversationClient(config)
        # Initialize the REST client without full start
        await client.rest_client.start()
        
        # Test connection
        result = await client.test_connection()
        
        print(f"Connected: {result['connected']}")
        print(f"URL: {result['url']}")
        if result['error']:
            print(f"Error: {result['error']}")
        if result['suggestions']:
            print("\nSuggestions:")
            for suggestion in result['suggestions']:
                print(f"  - {suggestion}")
        
        await client.rest_client.stop()
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("Connection error handling tests completed!")


if __name__ == "__main__":
    asyncio.run(test_connection_errors())