#!/usr/bin/env python3
"""
Test script to verify server VAD behavior without manual commits
This script should be run on the Raspberry Pi to test the fix.
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from openai_client.realtime import OpenAIRealtimeClient
from utils.logger import setup_logging, get_logger

async def test_server_vad():
    """Test server VAD behavior without manual commits"""
    # Setup logging
    setup_logging("DEBUG", console=True)
    logger = get_logger("ServerVADTest")
    
    # Load config
    config = load_config("config/config.yaml")
    
    # Create audio-enabled client (NOT text-only)
    logger.info("Creating audio-enabled client with server VAD...")
    client = OpenAIRealtimeClient(config.openai, "You are a helpful test assistant.", text_only=False)
    
    # Connect
    logger.info("Connecting...")
    success = await client.connect()
    if not success:
        logger.error("Failed to connect")
        return
    
    logger.info("Connected successfully - server VAD is active")
    
    # Use the minimal test audio from the client
    logger.info("Creating minimal test audio with exact OpenAI PCM16 specification...")
    test_audio = client._create_minimal_test_audio()
    
    if not test_audio:
        logger.error("Failed to create test audio")
        return
    
    # Calculate duration
    duration_ms = len(test_audio) / 2 / 24000 * 1000
    logger.info(f"Test audio: {len(test_audio)} bytes, {duration_ms:.1f}ms duration")
    
    # Send audio in chunks to simulate streaming
    chunk_size = int(24000 * 0.05 * 2)  # 50ms chunks
    
    logger.info(f"Sending audio in {chunk_size}-byte chunks...")
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i+chunk_size]
        chunk_duration_ms = len(chunk) / 2 / 24000 * 1000
        logger.info(f"Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes ({chunk_duration_ms:.1f}ms)")
        
        await client.send_audio(chunk)
        # Small delay between chunks
        await asyncio.sleep(0.01)
    
    logger.info(f"Sent total of {duration_ms:.1f}ms of test audio")
    
    # DO NOT manually commit - let server VAD handle it
    logger.info("Waiting for server VAD to process audio...")
    logger.info("Looking for 'input_audio_buffer.speech_stopped' event...")
    
    # Wait for server VAD to process
    await asyncio.sleep(10)
    
    # Cleanup
    await client.disconnect()
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_server_vad())