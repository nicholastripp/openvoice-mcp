#!/usr/bin/env python3
"""
Bit Depth Validator for PCM16 Conversion
Part of Task 1.4 - Audio Format Validation

Validates bit depth preservation and quantization quality in PCM16 conversion.
"""

import numpy as np
from typing import Dict, Callable
from dataclasses import dataclass


@dataclass
class BitDepthMetrics:
    """Bit depth validation results"""
    effective_bits: float
    quantization_noise: float
    dynamic_range_db: float
    noise_floor_db: float
    unique_levels: int
    theoretical_levels: int
    preservation_ratio: float


class BitDepthValidator:
    """Validate bit depth preservation in PCM16 conversion"""
    
    def __init__(self, target_bits: int = 16):
        self.target_bits = target_bits
        self.max_levels = 2 ** target_bits
    
    def measure_effective_bits(self, conversion_method: Callable, 
                              test_signal: np.ndarray = None) -> BitDepthMetrics:
        """
        Measure effective bit depth after conversion
        """
        if test_signal is None:
            # Generate ramp signal covering full range
            test_signal = np.linspace(-1.0, 1.0, 10000)
        
        # Convert signal
        pcm16 = conversion_method(test_signal)
        
        # Count unique quantization levels
        unique_levels = len(np.unique(pcm16))
        
        # Calculate effective bits
        if unique_levels > 1:
            effective_bits = np.log2(unique_levels)
        else:
            effective_bits = 0
        
        # Convert back to float for noise analysis
        reconstructed = pcm16.astype(np.float32) / 32768.0
        
        # Calculate quantization noise
        noise = test_signal[:len(reconstructed)] - reconstructed
        quantization_noise = np.sqrt(np.mean(noise ** 2))
        
        # Calculate dynamic range
        if quantization_noise > 0:
            dynamic_range_db = 20 * np.log10(2.0 / quantization_noise)
            noise_floor_db = 20 * np.log10(quantization_noise)
        else:
            dynamic_range_db = 96.33  # Theoretical max for 16-bit
            noise_floor_db = -96.33
        
        # Calculate preservation ratio
        theoretical_levels = min(self.max_levels, len(test_signal))
        preservation_ratio = unique_levels / theoretical_levels
        
        return BitDepthMetrics(
            effective_bits=effective_bits,
            quantization_noise=quantization_noise,
            dynamic_range_db=dynamic_range_db,
            noise_floor_db=noise_floor_db,
            unique_levels=unique_levels,
            theoretical_levels=theoretical_levels,
            preservation_ratio=preservation_ratio
        )
    
    def test_low_level_signals(self, conversion_method: Callable) -> Dict:
        """
        Test quantization of low-level signals
        """
        results = {}
        sample_rate = 24000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test different amplitude levels
        amplitudes = [1.0, 0.1, 0.01, 0.001, 0.0001]
        
        for amplitude in amplitudes:
            # Generate test signal
            signal = amplitude * np.sin(2 * np.pi * 1000 * t)
            
            # Convert
            pcm16 = conversion_method(signal)
            
            # Analyze
            unique_values = len(np.unique(pcm16))
            max_val = np.max(np.abs(pcm16))
            
            # Check if signal is preserved
            is_preserved = unique_values > 2  # More than just zero
            
            results[f'amplitude_{amplitude}'] = {
                'unique_values': unique_values,
                'max_pcm_value': int(max_val),
                'signal_preserved': is_preserved,
                'expected_pcm_range': int(amplitude * 32767)
            }
        
        return results


def demonstrate_bit_depth():
    """Demonstrate bit depth validation"""
    print("Bit Depth Validator Demonstration")
    print("=" * 50)
    
    validator = BitDepthValidator()
    
    # Test current implementation
    def current_method(audio: np.ndarray) -> np.ndarray:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    print("\nTesting bit depth preservation...")
    metrics = validator.measure_effective_bits(current_method)
    
    print(f"\nBit Depth Metrics:")
    print(f"  Effective Bits: {metrics.effective_bits:.2f}")
    print(f"  Quantization Noise: {metrics.quantization_noise:.6f}")
    print(f"  Dynamic Range: {metrics.dynamic_range_db:.1f} dB")
    print(f"  Noise Floor: {metrics.noise_floor_db:.1f} dB")
    print(f"  Unique Levels: {metrics.unique_levels}")
    print(f"  Preservation Ratio: {metrics.preservation_ratio:.2%}")
    
    print("\n" + "=" * 50)
    print("Testing low-level signals...")
    low_level_results = validator.test_low_level_signals(current_method)
    
    for amplitude, results in low_level_results.items():
        print(f"\n{amplitude}:")
        print(f"  Signal Preserved: {results['signal_preserved']}")
        print(f"  Unique Values: {results['unique_values']}")
        print(f"  Max PCM Value: {results['max_pcm_value']}")


if __name__ == "__main__":
    demonstrate_bit_depth()