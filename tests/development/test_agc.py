#!/usr/bin/env python3
"""
Test script for Automatic Gain Control (AGC)

This script tests the AGC with various audio scenarios:
- Quiet audio that needs amplification
- Normal audio that should remain unchanged
- Loud audio that needs attenuation
- Clipping audio that needs immediate reduction
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from audio.agc import AutomaticGainControl


@dataclass
class TestAudioConfig:
    """Minimal config for AGC testing"""
    input_volume: float = 1.0
    agc_enabled: bool = True
    agc_target_rms: float = 0.3
    agc_max_gain: float = 3.0
    agc_min_gain: float = 0.1
    agc_attack_time: float = 0.5
    agc_release_time: float = 2.0
    agc_clipping_threshold: float = 0.05


def generate_test_signal(duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate a test signal with varying amplitude"""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)
    
    # Create segments with different characteristics
    signal = np.zeros(samples)
    
    # Segment 1: Quiet (0-2s) - needs amplification
    mask1 = t < 2
    signal[mask1] = 0.1 * np.sin(2 * np.pi * 440 * t[mask1])
    
    # Segment 2: Normal (2-4s) - should stay roughly the same
    mask2 = (t >= 2) & (t < 4)
    signal[mask2] = 0.3 * np.sin(2 * np.pi * 440 * t[mask2])
    
    # Segment 3: Loud (4-6s) - needs slight attenuation
    mask3 = (t >= 4) & (t < 6)
    signal[mask3] = 0.8 * np.sin(2 * np.pi * 440 * t[mask3])
    
    # Segment 4: Clipping (6-8s) - needs immediate reduction
    mask4 = (t >= 6) & (t < 8)
    signal[mask4] = 1.5 * np.sin(2 * np.pi * 440 * t[mask4])  # Will clip
    signal[mask4] = np.clip(signal[mask4], -1.0, 1.0)  # Simulate clipping
    
    # Segment 5: Back to quiet (8-10s) - test release time
    mask5 = t >= 8
    signal[mask5] = 0.1 * np.sin(2 * np.pi * 440 * t[mask5])
    
    return signal.astype(np.float32)


def plot_results(original: np.ndarray, processed: np.ndarray, gain_history: list, 
                sample_rate: int = 48000):
    """Plot the test results"""
    duration = len(original) / sample_rate
    t = np.linspace(0, duration, len(original))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original signal
    ax1.plot(t, original, alpha=0.7, label='Original')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-1.5, 1.5)
    
    # Plot processed signal
    ax2.plot(t, processed, alpha=0.7, label='AGC Processed', color='green')
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Target RMS')
    ax2.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Amplitude')
    ax2.set_title('AGC Processed Signal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-1.5, 1.5)
    
    # Plot gain over time
    gain_t = np.linspace(0, duration, len(gain_history))
    ax3.plot(gain_t, gain_history, label='AGC Gain', color='orange', linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unity Gain')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Gain')
    ax3.set_title('AGC Gain Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 3.5)
    
    # Add segment labels
    segments = [
        (1, 'Quiet\n(needs boost)'),
        (3, 'Normal\n(target level)'),
        (5, 'Loud\n(needs reduction)'),
        (7, 'Clipping\n(immediate reduction)'),
        (9, 'Quiet again\n(slow release)')
    ]
    
    for x, label in segments:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=x-1, color='gray', alpha=0.3, linestyle=':')
            if ax == ax1:
                ax.text(x, 1.3, label, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def test_agc():
    """Test the AGC system"""
    print("Testing Automatic Gain Control (AGC)")
    print("=" * 50)
    
    # Create AGC with test config
    config = TestAudioConfig()
    agc = AutomaticGainControl(config, sample_rate=48000)
    
    # Generate test signal
    duration = 10  # seconds
    sample_rate = 48000
    chunk_size = 1024  # Process in chunks like real audio
    
    print(f"Generating {duration}s test signal...")
    original = generate_test_signal(duration, sample_rate)
    
    # Process through AGC
    print("Processing through AGC...")
    processed = np.zeros_like(original)
    gain_history = []
    
    # Process in chunks
    for i in range(0, len(original), chunk_size):
        chunk = original[i:i+chunk_size]
        if len(chunk) < chunk_size:
            # Pad last chunk if needed
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # Process chunk
        processed_chunk = agc.process_audio(chunk)
        processed[i:i+len(chunk)] = processed_chunk[:len(original[i:i+chunk_size])]
        
        # Record gain
        gain_history.append(agc.current_gain)
        
        # Print progress
        if i % (sample_rate * 2) == 0:  # Every 2 seconds
            stats = agc.get_stats()
            print(f"  t={i/sample_rate:.1f}s: gain={stats.current_gain:.2f}, "
                  f"rms={stats.last_rms:.3f}, clipping={stats.last_clipping_ratio:.1%}")
    
    # Final stats
    print("\nFinal AGC Statistics:")
    stats = agc.get_stats()
    print(f"  Total adjustments: {stats.total_adjustments}")
    print(f"  Clipping events: {stats.clipping_events}")
    print(f"  Final gain: {stats.current_gain:.2f}")
    
    # Calculate overall metrics
    original_rms = np.sqrt(np.mean(original**2))
    processed_rms = np.sqrt(np.mean(processed**2))
    print(f"\nOverall RMS:")
    print(f"  Original: {original_rms:.3f}")
    print(f"  Processed: {processed_rms:.3f}")
    print(f"  Target: {config.agc_target_rms:.3f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(original, processed, gain_history, sample_rate)


if __name__ == "__main__":
    test_agc()