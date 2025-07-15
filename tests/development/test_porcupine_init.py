#!/usr/bin/env python3
"""
Test Porcupine initialization issue
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from wake_word import create_wake_word_detector


async def test_init():
    """Test Porcupine initialization"""
    print("Loading config...")
    config = load_config()
    
    print(f"Wake word enabled: {config.wake_word.enabled}")
    print(f"Wake word engine: {getattr(config.wake_word, 'engine', 'not set')}")
    print(f"Wake word model: {config.wake_word.model}")
    
    # Force Porcupine
    config.wake_word.engine = 'porcupine'
    config.wake_word.model = 'picovoice'
    
    print("\nCreating wake word detector...")
    detector = create_wake_word_detector(config.wake_word)
    
    print("\nStarting detector...")
    try:
        await detector.start()
        print("Detector started successfully!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Stop it
    print("\nStopping detector...")
    await detector.stop()
    print("Done!")


def main():
    """Main entry point"""
    # Check access key
    access_key = os.getenv('PICOVOICE_ACCESS_KEY')
    print(f"Access key present: {bool(access_key)}")
    if access_key:
        print(f"Access key length: {len(access_key)}")
    
    # Run test
    asyncio.run(test_init())


if __name__ == "__main__":
    main()