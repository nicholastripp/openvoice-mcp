#!/usr/bin/env python3
"""
Simple test of Porcupine wake word detection
"""
import os
import sys
import wave
import struct
import time
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

try:
    import pvporcupine
except ImportError:
    print("ERROR: pvporcupine not installed")
    sys.exit(1)

# Get access key
access_key = os.getenv('PICOVOICE_ACCESS_KEY')
if not access_key:
    print("ERROR: PICOVOICE_ACCESS_KEY not set")
    sys.exit(1)

print("Testing Porcupine Wake Word Detection")
print("=" * 50)

# Test different sensitivity values
for sensitivity in [0.5, 0.9, 1.0]:
    print(f"\nTesting with sensitivity: {sensitivity}")
    
    try:
        # Create Porcupine instance
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=['jarvis'],
            sensitivities=[sensitivity]
        )
        
        print(f"[OK] Porcupine created successfully")
        print(f"  Sample rate: {porcupine.sample_rate}")
        print(f"  Frame length: {porcupine.frame_length}")
        
        # Test with silence (should not detect)
        silence = [0] * porcupine.frame_length
        result = porcupine.process(silence)
        print(f"  Silence test: {'PASS' if result == -1 else 'FAIL'} (result={result})")
        
        # Test with noise (should not detect)
        import random
        noise = [random.randint(-1000, 1000) for _ in range(porcupine.frame_length)]
        result = porcupine.process(noise)
        print(f"  Noise test: {'PASS' if result == -1 else 'FAIL'} (result={result})")
        
        # Clean up
        porcupine.delete()
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")

print("\n" + "=" * 50)
print("If all tests show -1, Porcupine is working correctly")
print("The issue may be with audio resampling or data format")