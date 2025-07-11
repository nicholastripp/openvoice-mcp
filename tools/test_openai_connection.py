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
    """Test audio format requirements using live microphone recording"""
    logger = get_logger("OpenAIAudio")
    
    try:
        import numpy as np
        import sounddevice as sd
        from scipy import signal
        from utils.dependencies import validate_audio_dependencies, validate_audio_device_access
        
        # Validate dependencies first
        try:
            validate_audio_dependencies()
            logger.info("✅ All audio dependencies validated")
        except Exception as e:
            logger.error(f"❌ Audio dependency validation failed: {e}")
            return False
        
        config = load_config(config_path)
        client = OpenAIRealtimeClient(config.openai, "You are a helpful assistant.")
        
        # Connect
        logger.info("Testing audio format requirements with live microphone...")
        success = await client.connect()
        if not success:
            logger.error("❌ Failed to connect")
            return False
        
        # Check microphone availability using validation utility
        logger.info("Checking microphone availability...")
        try:
            input_devices, output_devices = validate_audio_device_access()
            logger.info(f"✅ Found {len(input_devices)} input devices")
        except Exception as e:
            logger.error(f"❌ Audio device validation failed: {e}")
            return False
        
        # Use default input device
        try:
            default_input = sd.default.device[0]
            device_info = sd.query_devices(default_input)
            logger.info(f"Using microphone: {device_info['name']}")
        except Exception as e:
            logger.warning(f"Could not get default device info: {e}")
        
        # Record audio from microphone
        recording_duration = 2.0  # Record 2 seconds
        device_sample_rate = 44100  # Common microphone sample rate
        target_sample_rate = 24000  # OpenAI requirement
        
        logger.info(f"Recording {recording_duration} seconds of audio...")
        logger.info("Please speak into the microphone or make some noise...")
        
        # Record audio
        recording = sd.rec(
            frames=int(device_sample_rate * recording_duration),
            samplerate=device_sample_rate,
            channels=1,
            device=None,  # Use default
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        # Check if we got audio data
        if recording is None or len(recording) == 0:
            logger.error("❌ No audio data recorded")
            return False
        
        # Flatten to mono if needed
        if recording.ndim > 1:
            recording = np.mean(recording, axis=1)
        
        # Check audio level
        max_amplitude = np.max(np.abs(recording))
        logger.info(f"Recorded audio level: {max_amplitude:.4f}")
        
        if max_amplitude < 0.001:  # Very quiet
            logger.warning("⚠️ Audio level very low - you may need to speak louder or check microphone")
        
        # Resample to 24kHz if needed
        if device_sample_rate != target_sample_rate:
            logger.info(f"Resampling from {device_sample_rate}Hz to {target_sample_rate}Hz...")
            new_length = int(len(recording) * target_sample_rate / device_sample_rate)
            resampled = signal.resample(recording, new_length)
        else:
            resampled = recording
        
        # Convert to PCM16
        # Clamp to [-1, 1] range
        resampled = np.clip(resampled, -1.0, 1.0)
        
        # Convert to int16
        pcm16_data = (resampled * 32767).astype(np.int16)
        audio_bytes = pcm16_data.tobytes()
        
        # Calculate duration
        duration_ms = len(audio_bytes) / 2 / target_sample_rate * 1000
        logger.info(f"Processed audio: {len(audio_bytes)} bytes, {duration_ms:.1f}ms duration")
        
        # Send audio in chunks to simulate streaming (50ms chunks)
        chunk_size = int(target_sample_rate * 0.05 * 2)  # 50ms of 16-bit audio (2 bytes per sample)
        
        logger.info(f"Sending audio in {chunk_size}-byte chunks (50ms each)...")
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            chunk_duration_ms = len(chunk) / 2 / target_sample_rate * 1000
            logger.info(f"Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes ({chunk_duration_ms:.1f}ms)")
            
            await client.send_audio(chunk)
            # Wait appropriate time for chunk duration (simulate real-time)
            await asyncio.sleep(chunk_duration_ms / 1000.0)
        
        logger.info(f"Sent total of {duration_ms:.1f}ms of recorded audio")
        
        # With server VAD enabled, do NOT manually commit the audio buffer
        # The server will automatically detect speech end and process the audio
        logger.info("Waiting for server VAD to detect speech end...")
        logger.info("(Server VAD will automatically commit the audio buffer when speech stops)")
        
        # Wait longer for server VAD to process and respond
        await asyncio.sleep(5)
        
        await client.disconnect()
        logger.info("✅ Audio format test completed")
        
    except ImportError as e:
        logger.warning(f"❌ Required audio libraries not available: {e}")
        logger.info("Please install: pip install numpy scipy sounddevice")
        return False
    except Exception as e:
        logger.error(f"❌ Microphone access failed: {e}")
        logger.info("Please check microphone permissions and hardware")
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