#!/usr/bin/env python3
"""
Audio diagnostics test script
Run this to check audio system configuration before running the main voice assistant
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.audio_diagnostics import AudioDiagnostics

def main():
    print("Home Assistant Realtime Voice Assistant - Audio Diagnostics")
    print("=" * 60)
    
    diagnostics = AudioDiagnostics()
    results = diagnostics.validate_system_audio_config()
    diagnostics.print_diagnostic_report(results)
    
    # Additional quick tests
    print("\nADDITIONAL QUICK TESTS:")
    print("-" * 30)
    
    try:
        import sounddevice as sd
        print(f"✓ sounddevice library available (version: {sd.__version__})")
        
        # Test default device
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            print(f"✓ Default input device: {default_input}")
            print(f"✓ Default output device: {default_output}")
        except Exception as e:
            print(f"✗ Error getting default devices: {e}")
        
        # Test device query
        try:
            devices = sd.query_devices()
            input_count = sum(1 for d in devices if d['max_input_channels'] > 0)
            output_count = sum(1 for d in devices if d['max_output_channels'] > 0)
            print(f"✓ Found {input_count} input devices and {output_count} output devices")
        except Exception as e:
            print(f"✗ Error querying devices: {e}")
            
    except ImportError:
        print("✗ sounddevice library not available")
    
    try:
        import numpy as np
        print(f"✓ numpy library available (version: {np.__version__})")
    except ImportError:
        print("✗ numpy library not available")
    
    try:
        import scipy
        print(f"✓ scipy library available (version: {scipy.__version__})")
    except ImportError:
        print("✗ scipy library not available")
    
    # Test system commands
    import subprocess
    
    try:
        result = subprocess.run(['amixer', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ amixer command available")
        else:
            print("✗ amixer command failed")
    except FileNotFoundError:
        print("✗ amixer command not found")
    except subprocess.TimeoutExpired:
        print("✗ amixer command timed out")
    
    try:
        result = subprocess.run(['lsusb', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ lsusb command available")
        else:
            print("✗ lsusb command failed")
    except FileNotFoundError:
        print("✗ lsusb command not found")
    except subprocess.TimeoutExpired:
        print("✗ lsusb command timed out")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete. Review recommendations above.")
    print("=" * 60)

if __name__ == "__main__":
    main()