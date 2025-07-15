#!/usr/bin/env python3
"""
Test script to analyze audio levels through the processing pipeline
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import sounddevice as sd
import time
from scipy import signal as scipy_signal

# Configuration
SAMPLE_RATE = 48000  # Initial capture rate
TARGET_RATE = 24000  # OpenAI rate
WAKE_WORD_RATE = 16000  # Wake word rate
DURATION = 3  # seconds


def test_audio_pipeline():
    """Test audio at different gain levels and processing stages"""
    
    print("Audio Pipeline Test")
    print("=" * 70)
    print("This test will record 3 seconds of audio and analyze it through")
    print("the processing pipeline to identify where distortion occurs.")
    print("=" * 70)
    
    # List audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']} (inputs: {device['max_input_channels']})")
    
    device_id = input("\nSelect input device number (or press Enter for default): ").strip()
    device_id = int(device_id) if device_id else None
    
    print(f"\nSpeak normally for {DURATION} seconds after the beep...")
    print("Try different volumes: whisper, normal speech, and loud speech")
    
    # Play start beep
    beep_duration = 0.2
    beep_freq = 800
    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration))
    beep = 0.3 * np.sin(2 * np.pi * beep_freq * t)
    sd.play(beep, SAMPLE_RATE)
    sd.wait()
    
    # Record audio
    print("Recording...")
    audio_raw = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                       channels=1, device=device_id, dtype='float32')
    sd.wait()
    print("Recording complete!\n")
    
    # Flatten to 1D if needed
    if audio_raw.ndim > 1:
        audio_raw = audio_raw.flatten()
    
    # Analyze at each stage
    print("Audio Analysis Results:")
    print("=" * 70)
    
    # Stage 1: Raw input
    analyze_audio(audio_raw, "1. Raw Input (float32)", SAMPLE_RATE)
    
    # Stage 2: Input volume adjustment (simulate config.audio.input_volume)
    for gain in [1.0, 1.5, 2.0]:
        audio_gained = audio_raw * gain
        analyze_audio(audio_gained, f"2. After input gain ({gain}x)", SAMPLE_RATE)
    
    # Stage 3: Resample to 24kHz
    audio_24k = scipy_signal.resample(audio_gained, int(len(audio_gained) * TARGET_RATE / SAMPLE_RATE))
    analyze_audio(audio_24k, "3. After resample to 24kHz", TARGET_RATE)
    
    # Stage 4: Convert to PCM16
    audio_pcm16 = (np.clip(audio_24k, -1.0, 1.0) * 32767).astype(np.int16)
    analyze_audio_pcm16(audio_pcm16, "4. After PCM16 conversion", TARGET_RATE)
    
    # Stage 5: Wake word normalization
    audio_normalized = audio_pcm16.astype(np.float32) / 32767.0
    analyze_audio(audio_normalized, "5. Wake word normalized", TARGET_RATE)
    
    # Stage 6: Wake word gain (test different values)
    for gain in [1.0, 2.0, 3.0]:
        audio_wake_gained = audio_normalized * gain
        # Apply soft limiting
        audio_wake_limited = np.tanh(audio_wake_gained * 0.8) / 0.8
        analyze_audio(audio_wake_limited, f"6. After wake word gain ({gain}x) + soft limit", TARGET_RATE)
    
    # Stage 7: Resample to 16kHz for wake word
    audio_16k = scipy_signal.resample(audio_wake_limited, int(len(audio_wake_limited) * WAKE_WORD_RATE / TARGET_RATE))
    analyze_audio(audio_16k, "7. After resample to 16kHz", WAKE_WORD_RATE)
    
    # Stage 8: OpenAI normalization (on original 24kHz audio)
    rms = np.sqrt(np.mean(audio_normalized ** 2))
    if rms > 0:
        target_rms = 0.05  # New reduced target
        openai_gain = np.clip(target_rms / rms, 0.5, 3.0)  # New reduced max gain
        audio_openai = audio_normalized * openai_gain
        audio_openai = np.tanh(audio_openai * 0.9) / 0.9
        analyze_audio(audio_openai, f"8. After OpenAI norm (gain: {openai_gain:.2f}x)", TARGET_RATE)
    
    print("\nRecommendations:")
    print("-" * 70)
    print("1. Keep input_volume at 1.0 unless your mic is very quiet")
    print("2. Wake word audio_gain should be 1.0 (no amplification)")
    print("3. Use 'fixed' gain mode to avoid unpredictable amplification")
    print("4. If audio is clipping, reduce input_volume below 1.0")
    print("5. Soft limiting helps prevent harsh distortion")


def analyze_audio(audio, stage_name, sample_rate):
    """Analyze audio array and print statistics"""
    if len(audio) == 0:
        print(f"\n{stage_name}: No audio data")
        return
    
    # Calculate metrics
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    mean = np.mean(audio)
    
    # Clipping detection
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
    clipped_percent = (clipped_samples / len(audio)) * 100
    
    # Dynamic range
    if rms > 0:
        dynamic_range = 20 * np.log10(peak / rms)
    else:
        dynamic_range = 0
    
    print(f"\n{stage_name}:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  RMS level: {rms:.6f} ({20 * np.log10(rms) if rms > 0 else -np.inf:.1f} dB)")
    print(f"  Peak level: {peak:.6f} ({20 * np.log10(peak) if peak > 0 else -np.inf:.1f} dB)")
    print(f"  DC offset: {mean:.6f}")
    print(f"  Dynamic range: {dynamic_range:.1f} dB")
    
    if clipped_percent > 0:
        print(f"  ⚠️  CLIPPING: {clipped_samples} samples ({clipped_percent:.1f}%)")
    else:
        print(f"  ✓ No clipping detected")
    
    # Headroom
    headroom_db = 20 * np.log10(1.0 / peak) if peak > 0 else np.inf
    print(f"  Headroom: {headroom_db:.1f} dB")
    
    # Warnings
    if peak > 0.95:
        print("  ⚠️  WARNING: Audio is very close to clipping!")
    if rms < 0.001:
        print("  ⚠️  WARNING: Audio level is very low!")
    if abs(mean) > 0.01:
        print("  ⚠️  WARNING: Significant DC offset detected!")


def analyze_audio_pcm16(audio, stage_name, sample_rate):
    """Analyze PCM16 audio array"""
    # Convert to float for analysis
    audio_float = audio.astype(np.float32) / 32767.0
    analyze_audio(audio_float, stage_name, sample_rate)


if __name__ == "__main__":
    test_audio_pipeline()