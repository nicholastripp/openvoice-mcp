#!/usr/bin/env python3
"""
Debug audio format issues
"""
import numpy as np
import sys
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

# Simulate the audio processing pipeline
def test_audio_pipeline():
    print("Testing Audio Processing Pipeline")
    print("=" * 50)
    
    # Simulate incoming audio (48kHz, 600 samples)
    sample_rate = 48000
    samples = 600
    
    # Create test audio with a tone
    t = np.linspace(0, samples/sample_rate, samples)
    freq = 1000  # 1kHz tone
    audio_48k = (np.sin(2 * np.pi * freq * t) * 5000).astype(np.int16)
    
    print(f"1. Original audio (48kHz):")
    print(f"   Shape: {audio_48k.shape}")
    print(f"   Dtype: {audio_48k.dtype}")
    print(f"   Range: [{audio_48k.min()}, {audio_48k.max()}]")
    print(f"   First 10 samples: {audio_48k[:10]}")
    
    # Apply gain (3.5)
    gain = 3.5
    audio_float = audio_48k.astype(np.float32) * gain
    audio_float = np.clip(audio_float, -32768, 32767)
    audio_gained = audio_float.astype(np.int16)
    
    print(f"\n2. After gain ({gain}):")
    print(f"   Range: [{audio_gained.min()}, {audio_gained.max()}]")
    print(f"   First 10 samples: {audio_gained[:10]}")
    
    # Resample to 16kHz
    resample_ratio = 16000 / 48000
    new_length = int(len(audio_gained) * resample_ratio)
    old_indices = np.arange(len(audio_gained))
    new_indices = np.linspace(0, len(audio_gained) - 1, new_length)
    audio_16k = np.interp(new_indices, old_indices, audio_gained).astype(np.int16)
    
    print(f"\n3. After resampling to 16kHz:")
    print(f"   Shape: {audio_16k.shape}")
    print(f"   Range: [{audio_16k.min()}, {audio_16k.max()}]")
    print(f"   First 10 samples: {audio_16k[:10]}")
    
    # Check frame extraction
    frame_length = 512
    if len(audio_16k) >= frame_length:
        frame = audio_16k[:frame_length]
        print(f"\n4. Extracted frame (512 samples):")
        print(f"   Shape: {frame.shape}")
        print(f"   Dtype: {frame.dtype}")
        print(f"   Range: [{frame.min()}, {frame.max()}]")
        
        # Check if it's a Python list or numpy array
        print(f"\n5. Data type check:")
        print(f"   Type: {type(frame)}")
        print(f"   Is numpy array: {isinstance(frame, np.ndarray)}")
        
        # Convert to list (what Porcupine expects)
        frame_list = frame.tolist()
        print(f"   As list - type: {type(frame_list)}")
        print(f"   As list - length: {len(frame_list)}")
        print(f"   As list - first 5: {frame_list[:5]}")

if __name__ == "__main__":
    test_audio_pipeline()