#!/usr/bin/env python3
"""
Simple audio device test - minimal dependencies

This script provides a quick test of audio output devices without
requiring the full application configuration or complex audio processing.

Usage:
    python examples/test_simple_audio.py
    python examples/test_simple_audio.py --device 0  # Test specific device
"""
import argparse
import numpy as np
import sounddevice as sd
import time


def list_devices():
    """List all audio devices with simple formatting"""
    print("Available Audio Devices:")
    print("-" * 40)
    
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("Input")
            if device['max_output_channels'] > 0:
                device_type.append("Output")
            
            print(f"[{i}] {device['name']}")
            print(f"    Type: {', '.join(device_type)}")
            print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
            print(f"    Sample Rate: {device['default_samplerate']} Hz")
            print()
            
    except Exception as e:
        print(f"Error listing devices: {e}")


def test_device(device_id=None, duration=2.0):
    """Test audio playback on a specific device"""
    
    # Generate a simple 440Hz sine wave
    sample_rate = 44100
    frames = int(sample_rate * duration)
    t = np.linspace(0, duration, frames, False)
    
    # Create tone with volume ramp to avoid clicks
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    
    # Fade in/out
    fade_len = int(sample_rate * 0.1)  # 100ms fade
    tone[:fade_len] *= np.linspace(0, 1, fade_len)
    tone[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    try:
        device_name = "default"
        if device_id is not None:
            device_info = sd.query_devices(device_id)
            device_name = device_info['name']
            
        print(f"Playing test tone on device: {device_name}")
        print(f"Duration: {duration} seconds")
        print(f"Frequency: 440 Hz")
        print("Listen for the tone...")
        
        # Play the tone
        sd.play(tone, samplerate=sample_rate, device=device_id)
        sd.wait()
        
        print("Playback completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple audio device test")
    parser.add_argument("--device", type=int, help="Device index to test (see list)")
    parser.add_argument("--duration", type=float, default=2.0, help="Test duration in seconds")
    parser.add_argument("--list", action="store_true", help="List devices and exit")
    
    args = parser.parse_args()
    
    print("Simple Audio Device Test")
    print("=" * 30)
    
    # Always list devices first
    list_devices()
    
    if args.list:
        return
        
    # Test default device if no specific device specified
    if args.device is None:
        print("Testing default output device...")
        test_device(duration=args.duration)
    else:
        print(f"Testing device index {args.device}...")
        test_device(device_id=args.device, duration=args.duration)
    
    print("\nIf you heard a tone, audio output is working!")
    print("If you didn't hear anything, check:")
    print("  - Volume settings")
    print("  - Hardware connections") 
    print("  - Try a different device from the list above")


if __name__ == "__main__":
    main()