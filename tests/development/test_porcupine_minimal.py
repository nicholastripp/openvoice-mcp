#!/usr/bin/env python3
"""
Minimal test of Porcupine with known audio pattern
"""
import os
import sys
import numpy as np
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

import pvporcupine

# Get access key
access_key = os.getenv('PICOVOICE_ACCESS_KEY')
if not access_key:
    print("ERROR: PICOVOICE_ACCESS_KEY not set")
    sys.exit(1)

print("Minimal Porcupine Test")
print("=" * 50)

# Create Porcupine with jarvis keyword
porcupine = pvporcupine.create(
    access_key=access_key,
    keywords=['jarvis'],
    sensitivities=[1.0]
)

print(f"Porcupine created for 'jarvis' keyword")
print(f"Frame length: {porcupine.frame_length}")

# Test 1: Process 100 frames of silence
print("\nTest 1: Processing 100 frames of silence...")
detections = 0
for i in range(100):
    frame = [0] * porcupine.frame_length
    result = porcupine.process(frame)
    if result >= 0:
        detections += 1
print(f"Detections in silence: {detections} (should be 0)")

# Test 2: Process frames with varying noise levels
print("\nTest 2: Processing frames with increasing noise...")
import random
for noise_level in [100, 500, 1000, 5000, 10000]:
    detections = 0
    for i in range(10):
        frame = [random.randint(-noise_level, noise_level) for _ in range(porcupine.frame_length)]
        result = porcupine.process(frame)
        if result >= 0:
            detections += 1
    print(f"Noise level {noise_level}: {detections} detections in 10 frames")

# Test 3: Check what happens with actual speech-like patterns
print("\nTest 3: Speech-like pattern (sine wave bursts)...")
sample_rate = 16000
duration = porcupine.frame_length / sample_rate
t = np.linspace(0, duration, porcupine.frame_length)

detections = 0
for freq in [200, 400, 800, 1600]:  # Speech frequency range
    # Create a sine wave burst
    signal = (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)
    result = porcupine.process(signal.tolist())
    if result >= 0:
        detections += 1
        print(f"  Detection at {freq}Hz!")

print(f"Speech-like patterns: {detections} detections")

# Cleanup
porcupine.delete()

print("\n" + "=" * 50)
print("Test complete")
print("\nConclusion:")
print("If no detections occurred, Porcupine is working correctly")
print("and only responds to actual 'jarvis' speech, not random signals")