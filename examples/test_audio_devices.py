#!/usr/bin/env python3
"""
Test script to list and test audio devices

Usage:
    ./venv/bin/python examples/test_audio_devices.py
    
Note: Must be run from the project root using the virtual environment.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from utils.logger import setup_logging


def list_devices():
    """List available audio devices"""
    print("=== Available Input Devices ===")
    input_devices = AudioCapture.list_devices()
    for device in input_devices:
        print(f"Index: {device['index']}")
        print(f"Name: {device['name']}")
        print(f"Channels: {device['channels']}")
        print(f"Sample Rate: {device['sample_rate']}")
        print("-" * 40)
    
    print("\n=== Available Output Devices ===")
    output_devices = AudioPlayback.list_devices()
    for device in output_devices:
        print(f"Index: {device['index']}")
        print(f"Name: {device['name']}")
        print(f"Channels: {device['channels']}")
        print(f"Sample Rate: {device['sample_rate']}")
        print("-" * 40)


def test_input_device(device_id, duration=3.0):
    """Test an input device"""
    print(f"Testing input device: {device_id}")
    print(f"Recording for {duration} seconds...")
    
    success = AudioCapture.test_device(device_id, duration)
    
    if success:
        print("✅ Input device test PASSED")
    else:
        print("❌ Input device test FAILED")
    
    return success


def test_output_device(device_id, duration=2.0, frequency=440.0):
    """Test an output device"""
    print(f"Testing output device: {device_id}")
    print(f"Playing {frequency}Hz tone for {duration} seconds...")
    
    success = AudioPlayback.test_device(device_id, duration, frequency)
    
    if success:
        print("✅ Output device test PASSED")
    else:
        print("❌ Output device test FAILED")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Test audio devices")
    parser.add_argument("--list", action="store_true", help="List available devices")
    parser.add_argument("--test-input", type=str, help="Test input device (index or 'default')")
    parser.add_argument("--test-output", type=str, help="Test output device (index or 'default')")
    parser.add_argument("--test-all", action="store_true", help="Test all devices")
    parser.add_argument("--duration", type=float, default=2.0, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", console=True)
    
    if args.list or not any([args.test_input, args.test_output, args.test_all]):
        list_devices()
    
    if args.test_input:
        test_input_device(args.test_input, args.duration)
    
    if args.test_output:
        test_output_device(args.test_output, args.duration)
    
    if args.test_all:
        print("\n=== Testing All Input Devices ===")
        input_devices = AudioCapture.list_devices()
        for device in input_devices:
            print(f"\nTesting: {device['name']}")
            test_input_device(device['index'], 1.0)
        
        print("\n=== Testing All Output Devices ===")
        output_devices = AudioPlayback.list_devices()
        for device in output_devices:
            print(f"\nTesting: {device['name']}")
            test_output_device(device['index'], 1.0)


if __name__ == "__main__":
    main()