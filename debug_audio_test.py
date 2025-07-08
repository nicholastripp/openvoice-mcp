#!/usr/bin/env python3
"""
Minimal reproduction case for audio buffer error
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from openai_client.realtime import OpenAIRealtimeClient
from utils.logger import setup_logging, get_logger

async def debug_audio_test():
    """Minimal test to debug audio buffer issue"""
    # Setup logging
    setup_logging("DEBUG", console=True)
    logger = get_logger("AudioDebug")
    
    # Load config
    config = load_config("config/config.yaml")
    
    # Create audio-enabled client (NOT text-only)
    logger.info("Creating audio-enabled client...")
    client = OpenAIRealtimeClient(config.openai, "You are a test assistant.", text_only=False)
    
    logger.info(f"Client created: text_only={client.text_only}")
    logger.info(f"Session config: {client.session_config}")
    
    # Connect
    logger.info("Connecting...")
    success = await client.connect()
    if not success:
        logger.error("Failed to connect")
        return
    
    logger.info("Connected successfully")
    
    # Create minimal audio data (100ms of silence)
    sample_rate = 24000
    duration_seconds = 0.1  # 100ms
    samples = int(sample_rate * duration_seconds)
    
    # Create 100ms of zero audio (silence)
    import struct
    audio_data = b''.join(struct.pack('<h', 0) for _ in range(samples))
    
    logger.info(f"Created {len(audio_data)} bytes of audio data ({duration_seconds*1000}ms)")
    
    # Send the audio
    logger.info("Sending audio...")
    await client.send_audio(audio_data)
    
    # Small delay
    await asyncio.sleep(0.1)
    
    # Try to commit
    logger.info("Attempting to commit...")
    await client.commit_audio()
    
    # Cleanup
    await client.disconnect()
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(debug_audio_test())