#!/usr/bin/env python3
"""
Test script to verify audio format conversion fixes
"""
import numpy as np
import sys

# Test PCM16 conversion symmetry
def test_pcm16_conversion():
    print("Testing PCM16 conversion symmetry...")
    
    # Test values including edge cases
    test_values = [0.0, 0.5, -0.5, 0.999, -0.999, 1.0, -1.0]
    
    print("\nOLD METHOD (asymmetric):")
    print("Float32 -> PCM16 -> Float32")
    for val in test_values:
        # Old method: * 32767 / 32768
        pcm16_old = int(val * 32767)
        pcm16_old = np.clip(pcm16_old, -32768, 32767)
        float32_back_old = pcm16_old / 32768.0
        error_old = abs(val - float32_back_old)
        print(f"{val:7.4f} -> {pcm16_old:6d} -> {float32_back_old:7.4f} (error: {error_old:.6f})")
    
    print("\nNEW METHOD (symmetric):")
    print("Float32 -> PCM16 -> Float32")
    for val in test_values:
        # New method: * 32768 / 32768
        pcm16_new = int(val * 32768)
        pcm16_new = np.clip(pcm16_new, -32768, 32767)
        float32_back_new = pcm16_new / 32768.0
        error_new = abs(val - float32_back_new)
        print(f"{val:7.4f} -> {pcm16_new:6d} -> {float32_back_new:7.4f} (error: {error_new:.6f})")
    
    # Test with actual numpy arrays
    print("\n\nTesting with numpy arrays:")
    test_array = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    
    # Old method
    pcm16_old = (test_array * 32767).astype(np.int16)
    float32_back_old = pcm16_old.astype(np.float32) / 32768.0
    
    # New method  
    # Must clip BEFORE converting to int16 to avoid overflow
    pcm16_new_float = test_array * 32768
    pcm16_new_float = np.clip(pcm16_new_float, -32768, 32767)
    pcm16_new = pcm16_new_float.astype(np.int16)
    float32_back_new = pcm16_new.astype(np.float32) / 32768.0
    
    print("\nOriginal:     ", test_array)
    print("Old roundtrip:", float32_back_old)
    print("New roundtrip:", float32_back_new)
    print("\nOld errors:   ", np.abs(test_array - float32_back_old))
    print("New errors:   ", np.abs(test_array - float32_back_new))

    # Test audio-like signal
    print("\n\nTesting with audio-like signal:")
    t = np.linspace(0, 1, 1000)
    audio_signal = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Old method
    pcm16_old = (audio_signal * 32767).astype(np.int16)
    float32_back_old = pcm16_old.astype(np.float32) / 32768.0
    
    # New method
    pcm16_new = (audio_signal * 32768).astype(np.int16)
    pcm16_new = np.clip(pcm16_new, -32768, 32767)
    float32_back_new = pcm16_new.astype(np.float32) / 32768.0
    
    rms_error_old = np.sqrt(np.mean((audio_signal - float32_back_old) ** 2))
    rms_error_new = np.sqrt(np.mean((audio_signal - float32_back_new) ** 2))
    
    print(f"RMS error old method: {rms_error_old:.6f}")
    print(f"RMS error new method: {rms_error_new:.6f}")
    
    # Check DC bias
    dc_bias_old = np.mean(float32_back_old - audio_signal)
    dc_bias_new = np.mean(float32_back_new - audio_signal)
    
    print(f"\nDC bias old method: {dc_bias_old:.6f}")
    print(f"DC bias new method: {dc_bias_new:.6f}")

if __name__ == "__main__":
    test_pcm16_conversion()