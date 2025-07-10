#!/usr/bin/env python3
"""
Quick test to verify WebSocket connection fix
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from openai_client.realtime import OpenAIRealtimeClient
from utils.logger import setup_logging, get_logger

async def test_connection():
    """Test OpenAI WebSocket connection"""
    print("Testing OpenAI WebSocket connection...")
    
    # Setup logging
    logger = setup_logging(level="DEBUG")
    
    # Load config
    config = load_config("config/config.yaml")
    
    # Create client
    client = OpenAIRealtimeClient(config.openai, "Test personality")
    
    # Try to connect
    try:
        success = await client.connect()
        if success:
            print("SUCCESS: Connected to OpenAI Realtime API!")
            print("Connection test passed - WebSocket is working")
            await client.disconnect()
            return True
        else:
            print("FAILED: Could not connect to OpenAI")
            return False
    except Exception as e:
        print(f"ERROR: Connection failed with exception: {e}")
        return False

async def main():
    """Main test function"""
    success = await test_connection()
    
    if success:
        print("\nConnection fix successful! The app should now work normally.")
        print("You can now run: python src/main.py")
    else:
        print("\nConnection still failing. Check your API key and network.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)