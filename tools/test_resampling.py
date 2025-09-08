#!/usr/bin/env python3
"""
Standalone test script for resampling analysis
Tests available resampling methods without full pipeline dependencies
"""

import sys
import numpy as np
import time
from pathlib import Path
from scipy import signal

def test_scipy_methods():
    """Test scipy resampling methods"""
    print("\n" + "="*60)
    print("Testing SciPy Resampling Methods")
    print("="*60)
    
    # Generate test signal
    orig_sr = 48000
    target_sr = 24000
    duration = 0.1  # 100ms
    samples = int(duration * orig_sr)
    
    # Create test tone (1kHz)
    t = np.linspace(0, duration, samples, False)
    test_signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    print(f"\nTest signal: 1kHz tone, {duration*1000:.0f}ms, {orig_sr}Hz -> {target_sr}Hz")
    print(f"Input samples: {len(test_signal)}")
    
    # Test 1: scipy.signal.resample (current method)
    print("\n1. SciPy FFT (signal.resample) - Current Method:")
    start = time.perf_counter()
    new_length = int(len(test_signal) * target_sr / orig_sr)
    resampled_fft = signal.resample(test_signal, new_length)
    time_fft = (time.perf_counter() - start) * 1000
    print(f"   Output samples: {len(resampled_fft)}")
    print(f"   Processing time: {time_fft:.2f}ms")
    print(f"   RMS level: {np.sqrt(np.mean(resampled_fft**2)):.4f}")
    
    # Test 2: scipy.signal.resample_poly
    print("\n2. SciPy Polyphase (signal.resample_poly):")
    start = time.perf_counter()
    # Calculate GCD for up/down factors
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled_poly = signal.resample_poly(test_signal, up, down)
    time_poly = (time.perf_counter() - start) * 1000
    print(f"   Up/Down factors: {up}/{down}")
    print(f"   Output samples: {len(resampled_poly)}")
    print(f"   Processing time: {time_poly:.2f}ms")
    print(f"   RMS level: {np.sqrt(np.mean(resampled_poly**2)):.4f}")
    
    # Compare methods
    print("\n3. Method Comparison:")
    print(f"   Speed advantage (poly vs FFT): {time_fft/time_poly:.2f}x")
    
    # Measure quality difference
    min_len = min(len(resampled_fft), len(resampled_poly))
    diff = resampled_fft[:min_len] - resampled_poly[:min_len]
    rmse = np.sqrt(np.mean(diff**2))
    print(f"   RMSE between methods: {rmse:.6f}")
    
    # Test aliasing with high frequency
    print("\n4. Aliasing Test (8kHz tone):")
    test_high = 0.5 * np.sin(2 * np.pi * 8000 * t)
    
    # FFT method
    resampled_high_fft = signal.resample(test_high, new_length)
    rms_fft = np.sqrt(np.mean(resampled_high_fft**2))
    
    # Polyphase method
    resampled_high_poly = signal.resample_poly(test_high, up, down)
    rms_poly = np.sqrt(np.mean(resampled_high_poly**2))
    
    print(f"   FFT method RMS: {rms_fft:.4f}")
    print(f"   Polyphase RMS: {rms_poly:.4f}")
    print(f"   Aliasing rejection difference: {20*np.log10(rms_fft/(rms_poly+1e-10)):.1f}dB")
    
    return {
        'fft_time_ms': time_fft,
        'poly_time_ms': time_poly,
        'quality_rmse': rmse,
        'speed_ratio': time_fft/time_poly
    }

def test_performance_scaling():
    """Test performance with different chunk sizes"""
    print("\n" + "="*60)
    print("Performance Scaling Test")
    print("="*60)
    
    chunk_sizes = [1200, 2400, 4800, 9600]  # 50ms, 100ms, 200ms, 400ms at 24kHz
    orig_sr = 48000
    target_sr = 24000
    
    print("\nChunk Size | FFT (ms) | Poly (ms) | Speed Ratio")
    print("-" * 50)
    
    for chunk_size in chunk_sizes:
        # Generate test data
        test_data = np.random.randn(chunk_size * 2).astype(np.float32) * 0.5  # Input size
        
        # FFT method
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            new_len = chunk_size
            _ = signal.resample(test_data, new_len)
        time_fft = (time.perf_counter() - start) / iterations * 1000
        
        # Polyphase method
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = signal.resample_poly(test_data, up, down)
        time_poly = (time.perf_counter() - start) / iterations * 1000
        
        ratio = time_fft / time_poly
        print(f"{chunk_size:10} | {time_fft:8.2f} | {time_poly:9.2f} | {ratio:11.2f}x")
        
        # Check if real-time capable
        chunk_duration_ms = (chunk_size / target_sr) * 1000
        rt_fft = "Y" if time_fft < chunk_duration_ms * 0.5 else "N"
        rt_poly = "Y" if time_poly < chunk_duration_ms * 0.5 else "N"
        print(f"{'':10}   Real-time: {rt_fft:^8}   {rt_poly:^9}")

def test_quality_metrics():
    """Test quality metrics for both methods"""
    print("\n" + "="*60)
    print("Quality Metrics Test")
    print("="*60)
    
    orig_sr = 48000
    target_sr = 24000
    duration = 0.5
    samples = int(duration * orig_sr)
    
    # Generate complex test signal (speech-like)
    t = np.linspace(0, duration, samples, False)
    # Mix of formants typical for vowels
    f1, f2, f3 = 700, 1220, 2600  # Formants for 'a' sound
    speech_like = (
        0.6 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    # Add amplitude modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    speech_like *= envelope * 0.5
    
    print("\nProcessing speech-like signal with formants at 700Hz, 1220Hz, 2600Hz")
    
    # Resample with both methods
    new_length = int(len(speech_like) * target_sr / orig_sr)
    
    # FFT method
    resampled_fft = signal.resample(speech_like, new_length)
    
    # Polyphase method
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled_poly = signal.resample_poly(speech_like, up, down)
    
    # Calculate THD (simplified)
    def calculate_thd_simple(signal_data, sample_rate):
        """Simple THD calculation"""
        from scipy.fft import rfft, rfftfreq
        
        # Apply window
        windowed = signal_data * signal.windows.blackman(len(signal_data))
        
        # FFT
        spectrum = rfft(windowed)
        freqs = rfftfreq(len(windowed), 1/sample_rate)
        magnitude = np.abs(spectrum)
        
        # Find fundamental (highest peak)
        fund_idx = np.argmax(magnitude[1:1000]) + 1  # Skip DC
        fund_power = magnitude[fund_idx] ** 2
        
        # Sum harmonic power (2nd to 5th harmonics)
        harmonic_power = 0
        for h in range(2, 6):
            h_idx = fund_idx * h
            if h_idx < len(magnitude):
                harmonic_power += magnitude[h_idx] ** 2
        
        thd = np.sqrt(harmonic_power / fund_power) * 100 if fund_power > 0 else 0
        return thd
    
    thd_fft = calculate_thd_simple(resampled_fft, target_sr)
    thd_poly = calculate_thd_simple(resampled_poly, target_sr)
    
    print(f"\nTHD Results:")
    print(f"   FFT method: {thd_fft:.2f}%")
    print(f"   Polyphase: {thd_poly:.2f}%")
    
    # SNR (simplified)
    noise_floor_fft = np.std(resampled_fft[:100])  # Quiet portion
    signal_rms_fft = np.sqrt(np.mean(resampled_fft**2))
    snr_fft = 20 * np.log10(signal_rms_fft / (noise_floor_fft + 1e-10))
    
    noise_floor_poly = np.std(resampled_poly[:100])
    signal_rms_poly = np.sqrt(np.mean(resampled_poly**2))
    snr_poly = 20 * np.log10(signal_rms_poly / (noise_floor_poly + 1e-10))
    
    print(f"\nSNR Results:")
    print(f"   FFT method: {snr_fft:.1f}dB")
    print(f"   Polyphase: {snr_poly:.1f}dB")

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print(" Audio Resampling Analysis - Quick Test")
    print("="*70)
    print("\nTesting available SciPy resampling methods for HA Realtime Assistant")
    print("Comparing current implementation (FFT) with alternative (Polyphase)")
    
    # Run tests
    results = test_scipy_methods()
    test_performance_scaling()
    test_quality_metrics()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATION")
    print("="*60)
    
    if results['speed_ratio'] > 2:
        print("\n[OK] Polyphase method is significantly faster")
    else:
        print("\n[-] Polyphase method shows modest speed improvement")
    
    if results['quality_rmse'] < 0.001:
        print("[OK] Both methods produce very similar quality output")
    else:
        print(f"- Quality difference detected: RMSE = {results['quality_rmse']:.6f}")
    
    print("\nRECOMMENDATION:")
    print("Consider switching from scipy.signal.resample (FFT) to")
    print("scipy.signal.resample_poly (Polyphase) for:")
    print("  - Better performance (especially on Raspberry Pi)")
    print("  - Better aliasing rejection")
    print("  - Lower CPU usage")
    print("  - More predictable latency")
    
    print("\nIMPLEMENTATION:")
    print("In src/audio/capture.py line 319, replace:")
    print("  resampled = signal.resample(audio_data, new_length)")
    print("With:")
    print("  from math import gcd")
    print("  g = gcd(self.device_sample_rate, self.target_sample_rate)")
    print("  up = self.target_sample_rate // g")
    print("  down = self.device_sample_rate // g")
    print("  resampled = signal.resample_poly(audio_data, up, down)")

if __name__ == '__main__':
    main()