#!/usr/bin/env python3
"""
Test script for OpenAI Realtime API connection

Usage:
    ./venv/bin/python examples/test_openai_connection.py
    
Note: Must be run from the project root using the virtual environment.
Requires OpenAI API key configured in .env file.
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from openai_client.realtime import OpenAIRealtimeClient
from utils.logger import setup_logging, get_logger


async def test_connection(config_path):
    """Test basic connection to OpenAI Realtime API"""
    print("DEBUG: test_connection() starting...")
    logger = get_logger("OpenAITest")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create client in text-only mode for testing
        client = OpenAIRealtimeClient(config.openai, "You are a helpful test assistant.", text_only=True)
        
        # Test connection
        logger.info("Testing connection to OpenAI Realtime API...")
        success = await client.connect()
        
        if not success:
            logger.error("❌ Failed to connect to OpenAI")
            return False
        
        logger.info("✅ Successfully connected to OpenAI Realtime API")
        logger.info(f"Session ID: {client.session_id}")
        
        # Verify text-only mode configuration
        logger.info("Verifying text-only mode configuration...")
        session_config = client.session_config
        if "input_audio_format" in session_config:
            logger.warning("⚠️  Warning: Audio format fields present in text-only mode")
        if session_config.get("modalities") != ["text"]:
            logger.warning(f"⚠️  Warning: Expected modalities ['text'], got {session_config.get('modalities')}")
        if session_config.get("turn_detection") is not None:
            logger.warning(f"⚠️  Warning: Expected turn_detection=None in text-only mode, got {session_config.get('turn_detection')}")
        else:
            logger.info("✅ VAD properly disabled in text-only mode")
        
        # Test text message
        logger.info("Testing text message...")
        await client.send_text("Hello, this is a test message. Please respond with a brief greeting.")
        
        # Wait for response
        await asyncio.sleep(3)
        
        await client.disconnect()
        logger.info("✅ OpenAI text-only test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ OpenAI test failed: {e}")
        return False
    
    return True


async def test_function_calling(config_path):
    """Test function calling functionality"""
    logger = get_logger("OpenAIFunctions")
    
    try:
        config = load_config(config_path)
        client = OpenAIRealtimeClient(config.openai, "You are a helpful assistant that can call functions.", text_only=True)
        
        # Register test function
        async def test_function(args):
            """Test function for verification"""
            message = args.get("message", "No message")
            logger.info(f"Test function called with: {message}")
            return {"status": "success", "received": message}
        
        client.register_function(
            name="test_function",
            handler=test_function,
            description="A test function that echoes back a message",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo back"
                    }
                },
                "required": ["message"]
            }
        )
        
        # Connect
        logger.info("Testing function calling...")
        success = await client.connect()
        if not success:
            logger.error("❌ Failed to connect")
            return False
        
        # Send message that should trigger function call
        await client.send_text("Please call the test_function with the message 'Hello from function test'")
        
        # Wait for response
        await asyncio.sleep(5)
        
        await client.disconnect()
        logger.info("✅ Function calling test completed")
        
    except Exception as e:
        logger.error(f"❌ Function calling test failed: {e}")
        return False
    
    return True


async def test_audio_format(config_path):
    """Test audio format requirements"""
    logger = get_logger("OpenAIAudio")
    
    try:
        import numpy as np
        
        config = load_config(config_path)
        client = OpenAIRealtimeClient(config.openai, "You are a helpful assistant.")
        
        # Connect
        logger.info("Testing audio format requirements...")
        success = await client.connect()
        if not success:
            logger.error("❌ Failed to connect")
            return False
        
        # Generate test audio (PCM16, 24kHz, mono)
        sample_rate = 24000
        duration = 0.5  # 500ms - shorter test
        frequency = 440  # A4 note
        
        # Generate sine wave with proper amplitude for int16
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Use amplitude that maps well to int16 range (-32768 to 32767)
        audio_signal = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
        
        # Convert to PCM16 format (signed 16-bit integers)
        audio_pcm16 = (audio_signal * 32767).astype(np.int16)
        
        # Ensure we have the expected number of samples
        expected_samples = int(sample_rate * duration)
        actual_samples = len(audio_pcm16)
        logger.info(f"Generated {actual_samples} samples, expected {expected_samples}")
        
        # Convert to bytes in little-endian format (standard for PCM16)
        audio_bytes = audio_pcm16.tobytes()
        
        # Calculate expected duration and verify
        expected_duration_ms = len(audio_bytes) / 2 / sample_rate * 1000
        logger.info(f"Audio data: {len(audio_bytes)} bytes, expected duration: {expected_duration_ms:.1f}ms")
        
        # Send audio in chunks to simulate streaming (50ms chunks)
        chunk_size = int(sample_rate * 0.05 * 2)  # 50ms of 16-bit audio (2 bytes per sample)
        
        logger.info(f"Sending audio in {chunk_size}-byte chunks (50ms each)...")
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            chunk_duration_ms = len(chunk) / 2 / sample_rate * 1000
            logger.debug(f"Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes ({chunk_duration_ms:.1f}ms)")
            
            await client.send_audio(chunk)
            await asyncio.sleep(0.05)  # Wait 50ms between chunks
        
        logger.info(f"Sent total of {expected_duration_ms:.1f}ms of audio in {len(audio_bytes)} bytes")
        
        # Commit audio buffer (client now validates minimum duration)
        logger.info("Committing audio buffer...")
        await client.commit_audio()
        
        # Wait for response
        await asyncio.sleep(3)
        
        await client.disconnect()
        logger.info("✅ Audio format test completed")
        
    except ImportError:
        logger.warning("❌ NumPy not available, skipping audio test")
        return False
    except Exception as e:
        logger.error(f"❌ Audio test failed: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI Realtime API connection")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--functions", action="store_true", help="Test function calling")
    parser.add_argument("--audio", action="store_true", help="Test audio format")
    parser.add_argument("--connection", action="store_true", help="Test connection only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", console=True)
    logger = get_logger("OpenAITest")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {args.config}")
        logger.error("Please create the configuration file:")
        logger.error(f"  cp {args.config}.example {args.config}")
        logger.error("Then edit it with your OpenAI API key.")
        return
    
    async def run_tests():
        if args.connection:
            await test_connection(args.config)
        elif args.functions:
            await test_function_calling(args.config)
        elif args.audio:
            await test_audio_format(args.config)
        else:
            # Run all tests
            logger = get_logger("OpenAITest")
            logger.info("Running all OpenAI tests...")
            
            success1 = await test_connection(args.config)
            if success1:
                await test_function_calling(args.config)
                await test_audio_format(args.config)
    
    # Run tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()