#!/usr/bin/env python3
"""
Test script for basic audio output functionality

This script tests:
1. Audio device enumeration and selection
2. Basic audio tone generation and playback
3. Audio format handling (similar to OpenAI PCM16 format)
4. AudioPlayback class functionality

Usage:
    python examples/test_audio_output.py [--device DEVICE] [--duration SECONDS]
"""
import sys
import argparse
import numpy as np
import sounddevice as sd
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from audio.playback import AudioPlayback
from utils.logger import setup_logging, get_logger


def list_audio_devices():
    """List all available audio output devices"""
    print("\n=== Available Audio Output Devices ===")
    try:
        devices = sd.query_devices()
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                })
                print(f"  [{i}] {device['name']}")
                print(f"      Channels: {device['max_output_channels']}, Sample Rate: {device['default_samplerate']} Hz")
                
        if not output_devices:
            print("  No output devices found!")
            return []
            
        print(f"\nFound {len(output_devices)} output device(s)")
        return output_devices
        
    except Exception as e:
        print(f"Error listing devices: {e}")
        return []


def generate_test_tone(duration=2.0, frequency=440.0, sample_rate=24000):
    """
    Generate a test tone similar to OpenAI audio format
    
    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz
        sample_rate: Sample rate (OpenAI uses 24kHz)
        
    Returns:
        PCM16 audio data as bytes
    """
    # Generate sine wave
    frames = int(sample_rate * duration)
    t = np.linspace(0, duration, frames, False)
    
    # Create tone with fade in/out to avoid clicks
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Apply fade in/out (10% of duration each)
    fade_frames = int(frames * 0.1)
    tone[:fade_frames] *= np.linspace(0, 1, fade_frames)
    tone[-fade_frames:] *= np.linspace(1, 0, fade_frames)
    
    # Scale to reasonable volume (30% of max)
    tone *= 0.3
    
    # Convert to PCM16 format (same as OpenAI output)
    tone_clipped = np.clip(tone, -1.0, 1.0)
    pcm16_data = (tone_clipped * 32767).astype(np.int16)
    
    return pcm16_data.tobytes()


def test_sounddevice_direct(device=None, duration=2.0):
    """Test direct sounddevice playback"""
    print(f"\n=== Testing Direct sounddevice Playback ===")
    
    try:
        # Generate test tone at device sample rate for direct playback
        sample_rate = 44100  # Common device sample rate
        frames = int(sample_rate * duration)
        t = np.linspace(0, duration, frames, False)
        
        # Generate 440Hz tone
        tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        
        print(f"Playing {duration}s test tone at 440Hz...")
        print(f"Device: {device if device != 'default' else 'system default'}")
        print(f"Sample rate: {sample_rate} Hz")
        
        # Play using sounddevice
        sd.play(
            tone,
            samplerate=sample_rate,
            device=device if device != "default" else None
        )
        sd.wait()  # Wait for playback to complete
        
        print("[SUCCESS] Direct sounddevice playback completed")
        return True
        
    except Exception as e:
        print(f"[FAILED] Direct sounddevice playback failed: {e}")
        return False


def test_audioplayback_class(config, duration=2.0):
    """Test the AudioPlayback class from the main application"""
    print(f"\n=== Testing AudioPlayback Class ===")
    
    try:
        # Create AudioPlayback instance
        playback = AudioPlayback(config.audio)
        
        # Start the playback system
        print("Starting AudioPlayback system...")
        import asyncio
        
        async def test_playback():
            await playback.start()
            
            # Generate test audio in OpenAI format (PCM16, 24kHz)
            test_audio = generate_test_tone(duration=duration, sample_rate=24000)
            
            print(f"Playing {duration}s test tone through AudioPlayback class...")
            print(f"Audio format: PCM16, 24kHz mono (OpenAI format)")
            print(f"Device: {config.audio.output_device}")
            print(f"Target sample rate: {config.audio.sample_rate} Hz")
            
            # Play the audio
            playback.play_audio(test_audio)
            
            # Wait for playback to complete
            await asyncio.sleep(duration + 1.0)
            
            # Stop the playback system
            await playback.stop()
            
            return True
            
        # Run the async test
        result = asyncio.run(test_playback())
        
        if result:
            print("[SUCCESS] AudioPlayback class test completed")
        return result
        
    except Exception as e:
        print(f"[FAILED] AudioPlayback class test failed: {e}")
        return False


def test_device_selection(device_id):
    """Test if a specific device can be selected and used"""
    print(f"\n=== Testing Device Selection ===")
    
    try:
        if device_id == "default":
            print("Testing default device selection...")
            device_info = sd.query_devices(sd.default.device[1])  # Output device
        else:
            print(f"Testing device selection: {device_id}")
            if isinstance(device_id, str) and device_id.isdigit():
                device_id = int(device_id)
                
            device_info = sd.query_devices(device_id)
            
        print(f"Selected device: {device_info['name']}")
        print(f"Channels: {device_info['max_output_channels']}")
        print(f"Default sample rate: {device_info['default_samplerate']} Hz")
        
        # Quick test with this device
        return test_sounddevice_direct(device=device_id, duration=1.0)
        
    except Exception as e:
        print(f"[FAILED] Device selection test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test audio output functionality")
    parser.add_argument("--device", default="default", help="Audio output device (name, index, or 'default')")
    parser.add_argument("--duration", type=float, default=2.0, help="Test tone duration in seconds")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--list-only", action="store_true", help="Only list devices, don't run tests")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logger = setup_logging(level="INFO", console=True)
    
    print("Audio Output Test Script")
    print("=" * 50)
    
    # List available devices
    devices = list_audio_devices()
    
    if args.list_only:
        return
        
    if not devices:
        print("\nNo audio output devices found. Cannot proceed with tests.")
        return
        
    # Load configuration
    try:
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
        print(f"Configured output device: {config.audio.output_device}")
        print(f"Configured sample rate: {config.audio.sample_rate} Hz")
        print(f"Configured volume: {config.audio.output_volume}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default audio settings...")
        # Create minimal config for testing
        from config import AudioConfig
        config = type('Config', (), {})()
        config.audio = AudioConfig()
        config.audio.output_device = args.device
        config.audio.sample_rate = 48000
        config.audio.output_volume = 1.0
        config.audio.channels = 1
        config.audio.chunk_size = 1200
    
    # Test Results
    test_results = []
    
    print(f"\nStarting audio tests with device: {args.device}")
    print("-" * 50)
    
    # Test 1: Device selection
    print("\n1. Testing device selection and access...")
    result1 = test_device_selection(args.device)
    test_results.append(("Device Selection", result1))
    
    # Test 2: Direct sounddevice playback
    print("\n2. Testing direct sounddevice playback...")
    result2 = test_sounddevice_direct(device=args.device, duration=args.duration)
    test_results.append(("Direct Playback", result2))
    
    # Test 3: AudioPlayback class
    print("\n3. Testing AudioPlayback class...")
    result3 = test_audioplayback_class(config, duration=args.duration)
    test_results.append(("AudioPlayback Class", result3))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print(f"\nOverall Result: {'[SUCCESS] All tests passed!' if all_passed else '[FAILURE] Some tests failed'}")
    
    if all_passed:
        print("\nAudio output is working correctly!")
        print("You can proceed with testing the full voice assistant application.")
    else:
        print("\nAudio output has issues that need to be resolved.")
        print("Check device configuration and hardware connections.")
        
    print(f"\nIf you heard test tones during this test, audio output is working.")
    print(f"If you didn't hear anything, there may be:")
    print(f"  - Hardware connection issues")
    print(f"  - Wrong device selected") 
    print(f"  - Volume/mute settings")
    print(f"  - Driver or system audio issues")


if __name__ == "__main__":
    main()