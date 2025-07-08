#!/usr/bin/env python3
"""
Minimal reproduction case for audio buffer error using live microphone recording
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
    """Minimal test to debug audio buffer issue using microphone recording"""
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
    
    # Record audio from microphone
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
            return
        
        # Check microphone availability using validation utility
        logger.info("Checking microphone availability...")
        try:
            input_devices, output_devices = validate_audio_device_access()
            logger.info(f"✅ Found {len(input_devices)} input devices")
        except Exception as e:
            logger.error(f"❌ Audio device validation failed: {e}")
            return
        
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
            logger.error("No audio data recorded")
            return
        
        # Flatten to mono if needed
        if recording.ndim > 1:
            recording = np.mean(recording, axis=1)
        
        # Check audio level
        max_amplitude = np.max(np.abs(recording))
        logger.info(f"Recorded audio level: {max_amplitude:.4f}")
        
        if max_amplitude < 0.001:  # Very quiet
            logger.warning("Audio level very low - you may need to speak louder or check microphone")
        
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
        audio_data = pcm16_data.tobytes()
        
        # Calculate duration
        duration_ms = len(audio_data) / 2 / target_sample_rate * 1000
        logger.info(f"Processed audio: {len(audio_data)} bytes, {duration_ms:.1f}ms duration")
        
        # Send audio in chunks to simulate streaming (50ms chunks)
        chunk_size = int(target_sample_rate * 0.05 * 2)  # 50ms of 16-bit audio (2 bytes per sample)
        
        logger.info(f"Sending audio in {chunk_size}-byte chunks (50ms each)...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            chunk_duration_ms = len(chunk) / 2 / target_sample_rate * 1000
            logger.debug(f"Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes ({chunk_duration_ms:.1f}ms)")
            
            await client.send_audio(chunk)
            await asyncio.sleep(0.05)  # Wait 50ms between chunks
        
        logger.info(f"Sent total of {duration_ms:.1f}ms of recorded audio")
        
        # Try to commit
        logger.info("Attempting to commit...")
        await client.commit_audio()
        
        # Wait for response
        await asyncio.sleep(3)
        
    except ImportError as e:
        logger.error(f"Required audio libraries not available: {e}")
        logger.info("Please install: pip install numpy scipy sounddevice")
        return
    except Exception as e:
        logger.error(f"Microphone recording failed: {e}")
        logger.info("Please check microphone permissions and hardware")
        return
    
    # Cleanup
    await client.disconnect()
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(debug_audio_test())