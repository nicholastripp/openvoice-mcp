#!/usr/bin/env python3
"""
Test script for the web UI
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.web.app import WebApp


async def main():
    """Test the web UI"""
    print("Starting Web UI test...")
    
    # Use config directory
    config_dir = Path(__file__).parent.parent / "config"
    
    # Create web app
    web_app = WebApp(config_dir, port=8080)
    
    try:
        # Start the web server
        await web_app.start()
        
        print("\n" + "="*70)
        print("WEB UI TEST SERVER RUNNING")
        print("="*70)
        print("URL: http://localhost:8080")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await web_app.stop()
        print("Web UI stopped")


if __name__ == "__main__":
    asyncio.run(main())