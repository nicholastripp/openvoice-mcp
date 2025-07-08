#!/usr/bin/env python3
"""
Comprehensive audio device testing and diagnostics script

This script helps diagnose audio input issues and test different audio devices
on Raspberry Pi for the wake word detection system.

Usage:
    ./venv/bin/python examples/test_audio_devices.py --list
    ./venv/bin/python examples/test_audio_devices.py --test-input-levels
    ./venv/bin/python examples/test_audio_devices.py --record-test
    ./venv/bin/python examples/test_audio_devices.py --device-diagnosis 2
    ./venv/bin/python examples/test_audio_devices.py --system-check
    
Note: Must be run from the project root using the virtual environment.
"""
import sys
import argparse
import asyncio
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from utils.logger import setup_logging, get_logger

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


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


def list_devices_detailed():
    """List all available audio devices with detailed information"""
    logger = get_logger("AudioDeviceTest")
    
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice library not available for detailed listing")
        list_devices()  # Fall back to basic listing
        return
    
    logger.info("Listing all available audio devices...")
    print("\n" + "="*80)
    print("DETAILED AUDIO DEVICE INFORMATION")
    print("="*80)
    
    devices = sd.query_devices()
    
    input_devices = []
    output_devices = []
    
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
            input_devices.append((i, device))
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
            output_devices.append((i, device))
        
        type_str = "/".join(device_type) if device_type else "NONE"
        
        print(f"Device {i:2d}: {device['name']}")
        print(f"           Type: {type_str}")
        print(f"           Sample Rate: {device['default_samplerate']:.0f} Hz")
        print(f"           Input Channels: {device['max_input_channels']}")
        print(f"           Output Channels: {device['max_output_channels']}")
        print(f"           API: {sd.query_hostapis()[device['hostapi']]['name']}")
        print()
    
    # Show default devices
    try:
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        print("DEFAULT DEVICES:")
        print(f"  Input:  Device {default_input} - {devices[default_input]['name']}")
        print(f"  Output: Device {default_output} - {devices[default_output]['name']}")
    except Exception as e:
        print(f"Could not determine default devices: {e}")
    
    print("\n" + "="*80)
    print(f"Found {len(input_devices)} input devices and {len(output_devices)} output devices")
    
    return input_devices, output_devices


def test_input_levels(device_id=None, duration=10, sample_rate=48000):
    """Test input audio levels for microphone diagnosis"""
    logger = get_logger("AudioInputTest")
    
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice library not available for level testing")
        return False
    
    if device_id is not None:
        try:
            device_info = sd.query_devices(device_id)
            logger.info(f"Testing input levels for device {device_id}: {device_info['name']}")
            if device_info['max_input_channels'] == 0:
                logger.error(f"Device {device_id} has no input channels!")
                return False
        except Exception as e:
            logger.error(f"Cannot query device {device_id}: {e}")
            return False
    else:
        logger.info("Testing input levels for default input device")
    
    print(f"\nTesting microphone input for {duration} seconds...")
    print("🎤 SPEAK NORMALLY, then LOUDLY, then whisper...")
    print("Press Ctrl+C to stop early")
    print("-" * 60)
    
    # Audio data collection
    audio_levels = []
    max_level = 0.0
    start_time = time.time()
    
    def audio_callback(indata, frames, time, status):
        nonlocal max_level
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Calculate audio level
        level = np.max(np.abs(indata))
        audio_levels.append(level)
        max_level = max(max_level, level)
        
        # Real-time feedback with visual bar
        current_time = time.time() - start_time
        bar_length = int(level * 50)  # Scale to 50 chars
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\r{current_time:5.1f}s │{bar}│ {level:.6f} (max: {max_level:.6f})", end="")
    
    try:
        # Start audio stream
        stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            callback=audio_callback,
            blocksize=1024
        )
        
        with stream:
            time.sleep(duration)
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        logger.error(f"Audio test failed: {e}")
        return False
    
    print()  # New line after progress bar
    
    # Analysis
    if audio_levels:
        avg_level = np.mean(audio_levels)
        percentile_95 = np.percentile(audio_levels, 95)
        
        print("-" * 60)
        print("🔍 AUDIO LEVEL ANALYSIS:")
        print(f"  Maximum level:     {max_level:.6f}")
        print(f"  Average level:     {avg_level:.6f}")
        print(f"  95th percentile:   {percentile_95:.6f}")
        print(f"  Samples collected: {len(audio_levels)}")
        
        # Assessment
        print("\n📊 ASSESSMENT:")
        if max_level < 0.001:
            print("  ❌ VERY LOW - Microphone may not be working or gain extremely low")
            print("     Expected for wake word detection: > 0.05")
        elif max_level < 0.01:
            print("  ⚠️  LOW - Microphone working but needs significant gain adjustment")
            print("     Current vs Expected: {:.6f} vs > 0.05".format(max_level))
        elif max_level < 0.05:
            print("  ⚠️  MARGINAL - May work but could be unreliable for wake word detection")
        elif max_level < 0.1:
            print("  ✅ GOOD - Should work well for wake word detection")
        else:
            print("  ✅ EXCELLENT - Strong microphone input levels")
        
        # Specific recommendations for wake word detection
        if max_level < 0.05:
            print("\n💡 WAKE WORD DETECTION RECOMMENDATIONS:")
            print("  1. Increase microphone gain: alsamixer (F4 for capture)")
            print("  2. Check microphone connection and power")
            print("  3. Try a USB microphone with built-in amplification")
            print("  4. Verify microphone is selected as input device")
            print("  5. Check if microphone is muted (amixer)")
    
    return True


def record_and_playback_test(device_id=None, duration=3, sample_rate=48000):
    """Record audio and play it back for verification"""
    logger = get_logger("AudioRecordTest")
    
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice library not available for record test")
        return False
    
    if device_id is not None:
        try:
            device_info = sd.query_devices(device_id)
            logger.info(f"Recording test for device {device_id}: {device_info['name']}")
        except Exception as e:
            logger.error(f"Cannot query device {device_id}: {e}")
            return False
    else:
        logger.info("Recording test for default input device")
    
    print(f"\n🎙️  Recording {duration} seconds of audio...")
    print("Say: 'Alexa, test microphone one two three'")
    
    try:
        # Record audio
        print("🔴 Recording... (speak now)")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        # Analyze recording
        max_level = np.max(np.abs(recording))
        avg_level = np.mean(np.abs(recording))
        
        print(f"✅ Recording complete!")
        print(f"  Max level: {max_level:.6f}")
        print(f"  Avg level: {avg_level:.6f}")
        
        if max_level < 0.001:
            print("❌ No audio detected - check microphone connection")
            return False
        
        # Play back the recording
        print(f"\n🔊 Playing back recording...")
        sd.play(recording, samplerate=sample_rate)
        sd.wait()  # Wait for playback to complete
        
        print("✅ Record and playback test complete")
        print("   Does the playback sound clear and at reasonable volume?")
        
        return True
        
    except Exception as e:
        logger.error(f"Record/playback test failed: {e}")
        return False


def diagnose_device(device_id):
    """Comprehensive diagnosis of a specific audio device"""
    logger = get_logger("DeviceDiagnosis")
    
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice library not available for device diagnosis")
        return False
    
    try:
        device_info = sd.query_devices(device_id)
    except Exception as e:
        logger.error(f"Cannot query device {device_id}: {e}")
        return False
    
    if device_info['max_input_channels'] == 0:
        logger.error(f"Device {device_id} has no input channels")
        return False
    
    logger.info(f"Comprehensive diagnosis for device {device_id}")
    print(f"\n🔬 DIAGNOSING Device {device_id}: {device_info['name']}")
    print("=" * 70)
    
    # Device info
    print(f"📋 DEVICE INFORMATION:")
    print(f"  Sample Rate: {device_info['default_samplerate']:.0f} Hz")
    print(f"  Input Channels: {device_info['max_input_channels']}")
    print(f"  API: {sd.query_hostapis()[device_info['hostapi']]['name']}")
    
    # Test 1: Input levels
    print(f"\n🎚️  TEST 1: INPUT LEVEL ANALYSIS (10 seconds)")
    success1 = test_input_levels(device_id, duration=10)
    
    if not success1:
        print("❌ Input level test failed - device may not be functional")
        return False
    
    # Test 2: Record and playback
    print(f"\n🎙️  TEST 2: RECORD AND PLAYBACK VERIFICATION (5 seconds)")
    success2 = record_and_playback_test(device_id, duration=5)
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT:")
    if success1 and success2:
        print("✅ Device appears to be functional")
        print("   If wake word detection still fails, check:")
        print("   - Audio gain settings (alsamixer)")
        print("   - Device selection in config")
        print("   - OpenWakeWord sensitivity settings")
    else:
        print("❌ Device has issues - try a different audio device")
    
    return success1 and success2


def system_audio_check():
    """Display system audio troubleshooting information"""
    logger = get_logger("SystemAudioCheck")
    
    print("\n🔧 SYSTEM AUDIO TROUBLESHOOTING GUIDE")
    print("=" * 70)
    
    print("1️⃣  CHECK MICROPHONE GAIN (ALSA):")
    print("   alsamixer")
    print("   - Press F4 to show capture devices")
    print("   - Use arrow keys to select microphone")
    print("   - Use +/- or Page Up/Down to increase gain")
    print("   - Press M to unmute if showing 'MM'")
    print("   - Press Esc to exit")
    print()
    
    print("2️⃣  LIST AUDIO HARDWARE:")
    print("   arecord -l    # List recording devices")
    print("   aplay -l      # List playback devices")
    print("   lsusb         # Check if USB microphone is detected")
    print()
    
    print("3️⃣  TEST RECORDING WITH ALSA:")
    print("   arecord -D hw:1,0 -f cd -t wav -d 5 test.wav")
    print("   aplay test.wav")
    print("   (Replace hw:1,0 with your device from arecord -l)")
    print()
    
    print("4️⃣  CHECK MIXER SETTINGS:")
    print("   amixer          # Show all mixer controls")
    print("   amixer sget Mic # Show microphone settings")
    print("   amixer sset Mic 80%  # Set microphone to 80%")
    print()
    
    print("5️⃣  USB MICROPHONE TROUBLESHOOTING:")
    print("   - Check if it appears in lsusb")
    print("   - Try different USB port")
    print("   - Some need power - check USB power settings")
    print("   - May need to set as default in ~/.asoundrc")
    print()
    
    print("6️⃣  RASPBERRY PI SPECIFIC:")
    print("   sudo raspi-config")
    print("   - Advanced Options → Audio")
    print("   - Enable audio if disabled")
    print("   - Check if audio group membership: groups $USER")
    print()
    
    print("7️⃣  EXPECTED AUDIO LEVELS FOR WAKE WORD DETECTION:")
    print("   - Normal speech: 0.01 - 0.05")
    print("   - Loud speech:   0.05 - 0.15")
    print("   - Current issue: levels < 0.01 (too low)")
    print()
    
    print("💡 If levels are still too low after trying above:")
    print("   - Consider USB microphone with built-in amplification")
    print("   - Add external microphone amplifier")
    print("   - Use closer microphone placement")


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
    parser = argparse.ArgumentParser(
        description="Comprehensive audio device testing for wake word detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available audio devices
  python examples/test_audio_devices.py --list
  
  # Test microphone input levels for diagnosis
  python examples/test_audio_devices.py --test-input-levels
  
  # Record and playback test
  python examples/test_audio_devices.py --record-test
  
  # Comprehensive diagnosis of device 2
  python examples/test_audio_devices.py --device-diagnosis 2
  
  # Show system audio troubleshooting guide
  python examples/test_audio_devices.py --system-check
        """
    )
    
    # Main testing options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List available devices (basic)")
    group.add_argument("--list-detailed", action="store_true", help="List all devices with detailed info")
    group.add_argument("--test-input-levels", action="store_true", help="Test microphone input levels (RECOMMENDED)")
    group.add_argument("--record-test", action="store_true", help="Record and playback test")
    group.add_argument("--device-diagnosis", type=int, metavar="ID", help="Comprehensive diagnosis of device ID")
    group.add_argument("--system-check", action="store_true", help="Show system audio troubleshooting guide")
    
    # Legacy options for backwards compatibility
    group.add_argument("--test-input", type=str, help="Legacy: Test input device (index or 'default')")
    group.add_argument("--test-output", type=str, help="Legacy: Test output device (index or 'default')")
    group.add_argument("--test-all", action="store_true", help="Legacy: Test all devices")
    
    # Optional parameters
    parser.add_argument("--device", type=int, help="Device ID for input level testing")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10)")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate (default: 48000)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, console=True)
    logger = get_logger("AudioDeviceTest")
    
    if not SOUNDDEVICE_AVAILABLE:
        logger.warning("sounddevice library not available - some features limited")
    
    try:
        # New comprehensive options
        if args.list_detailed:
            list_devices_detailed()
        
        elif args.test_input_levels:
            logger.info("Testing microphone input levels - this is the recommended test for wake word issues")
            test_input_levels(device_id=args.device, duration=args.duration, sample_rate=args.sample_rate)
        
        elif args.record_test:
            record_and_playback_test(device_id=args.device, duration=5, sample_rate=args.sample_rate)
        
        elif args.device_diagnosis is not None:
            diagnose_device(args.device_diagnosis)
        
        elif args.system_check:
            system_audio_check()
        
        # Legacy options for backwards compatibility
        elif args.list or not any([args.test_input, args.test_output, args.test_all]):
            list_devices()
        
        elif args.test_input:
            test_input_device(args.test_input, args.duration)
        
        elif args.test_output:
            test_output_device(args.test_output, args.duration)
        
        elif args.test_all:
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
        
        else:
            # Default action if no arguments
            logger.info("No specific test requested. Showing available devices...")
            list_devices_detailed() if SOUNDDEVICE_AVAILABLE else list_devices()
            print("\n💡 For microphone issues, try: --test-input-levels")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Audio test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()