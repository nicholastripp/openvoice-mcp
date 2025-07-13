#!/usr/bin/env python3
"""
Test audio gain clipping issue
"""
import numpy as np

# Simulate the current (broken) code
def broken_gain(audio_data, gain=3.5):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # WRONG: multiply then cast to int16 - causes truncation
    audio_array = (audio_array * gain).astype(np.int16)
    audio_array = np.clip(audio_array, -32768, 32767)
    return audio_array

# Correct implementation
def correct_gain(audio_data, gain=3.5):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # Convert to float for multiplication
    audio_float = audio_array.astype(np.float32) * gain
    # Clip BEFORE converting back to int16
    audio_float = np.clip(audio_float, -32768, 32767)
    # Then convert to int16
    audio_array = audio_float.astype(np.int16)
    return audio_array

# Test with sample audio
test_values = np.array([1000, 5000, 10000, 20000], dtype=np.int16)
test_data = test_values.tobytes()

print("Test Audio Gain Issue")
print("=" * 40)
print(f"Original values: {test_values}")
print(f"With gain 3.5:")

broken = broken_gain(test_data, 3.5)
print(f"Broken implementation: {broken}")
print(f"  Note: 20000 * 3.5 = 70000, but cast to int16 = {np.int16(70000)}")

correct = correct_gain(test_data, 3.5)
print(f"Correct implementation: {correct}")
print(f"  Note: Values properly clipped to int16 range")

print("\nThe issue: with gain=3.5, any audio > 9362 gets wrapped around!")
print("This causes severe distortion and makes wake word detection impossible.")