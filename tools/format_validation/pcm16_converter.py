#!/usr/bin/env python3
"""
PCM16 Converter Comparison Module
Part of Task 1.4 - Audio Format Validation

Implements and compares different PCM16 conversion methods to identify
the optimal approach for the HA Realtime Voice Assistant.
"""

import numpy as np
from typing import Dict, Callable, Tuple
import time


class PCM16Converter:
    """Collection of PCM16 conversion methods for comparison"""
    
    @staticmethod
    def method_32767_symmetric(audio: np.ndarray) -> np.ndarray:
        """Current implementation - symmetric scaling with 32767"""
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    @staticmethod
    def method_32768_asymmetric(audio: np.ndarray) -> np.ndarray:
        """Traditional asymmetric scaling with 32768"""
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32768).astype(np.int16)
    
    @staticmethod
    def method_hybrid(audio: np.ndarray) -> np.ndarray:
        """Hybrid approach - different scaling for positive/negative"""
        audio = np.clip(audio, -1.0, 1.0)
        result = np.zeros_like(audio, dtype=np.int16)
        negative_mask = audio < 0
        result[negative_mask] = (audio[negative_mask] * 32768).astype(np.int16)
        result[~negative_mask] = (audio[~negative_mask] * 32767).astype(np.int16)
        return result
    
    @staticmethod
    def method_clamped(audio: np.ndarray) -> np.ndarray:
        """Clamped scaling with explicit bounds checking"""
        audio = np.clip(audio, -1.0, 1.0)
        scaled = audio * 32767
        return np.clip(scaled, -32768, 32767).astype(np.int16)
    
    @staticmethod
    def method_dithered(audio: np.ndarray) -> np.ndarray:
        """Dithered conversion for improved low-level signal handling"""
        audio = np.clip(audio, -1.0, 1.0)
        # Add triangular probability density function (TPDF) dither
        dither = (np.random.random(audio.shape) + np.random.random(audio.shape) - 1.0) / 65536
        dithered = audio + dither
        dithered = np.clip(dithered, -1.0, 1.0)
        return (dithered * 32767).astype(np.int16)
    
    @staticmethod
    def method_rounded(audio: np.ndarray) -> np.ndarray:
        """Rounded conversion for better quantization"""
        audio = np.clip(audio, -1.0, 1.0)
        scaled = audio * 32767
        return np.round(scaled).astype(np.int16)
    
    @classmethod
    def get_all_methods(cls) -> Dict[str, Callable]:
        """Get all conversion methods"""
        return {
            'symmetric_32767': cls.method_32767_symmetric,
            'asymmetric_32768': cls.method_32768_asymmetric,
            'hybrid': cls.method_hybrid,
            'clamped': cls.method_clamped,
            'dithered': cls.method_dithered,
            'rounded': cls.method_rounded
        }
    
    @staticmethod
    def benchmark_method(method: Callable, audio: np.ndarray, 
                        iterations: int = 100) -> Tuple[float, float]:
        """
        Benchmark a conversion method
        
        Returns:
            Tuple of (mean_time, std_time) in milliseconds
        """
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = method(audio)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return np.mean(times), np.std(times)
    
    @staticmethod
    def compare_methods(test_audio: np.ndarray = None) -> Dict:
        """
        Compare all conversion methods
        
        Args:
            test_audio: Test signal (if None, generates default)
            
        Returns:
            Comparison results
        """
        if test_audio is None:
            # Generate 1 second of test audio at 24kHz
            t = np.linspace(0, 1, 24000)
            test_audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        converter = PCM16Converter()
        methods = converter.get_all_methods()
        results = {}
        
        for name, method in methods.items():
            # Convert
            pcm16 = method(test_audio.copy())
            
            # Measure performance
            mean_time, std_time = converter.benchmark_method(method, test_audio)
            
            # Analyze output
            results[name] = {
                'min_value': int(np.min(pcm16)),
                'max_value': int(np.max(pcm16)),
                'mean_value': float(np.mean(pcm16)),
                'std_value': float(np.std(pcm16)),
                'unique_values': len(np.unique(pcm16)),
                'mean_time_ms': mean_time,
                'std_time_ms': std_time,
                'uses_full_range': (np.min(pcm16) <= -32767) or (np.max(pcm16) >= 32767)
            }
        
        return results
    
    @staticmethod
    def recommend_optimal_method(comparison_results: Dict) -> str:
        """
        Recommend the optimal conversion method based on results
        
        Args:
            comparison_results: Results from compare_methods()
            
        Returns:
            Name of recommended method
        """
        scores = {}
        
        for method_name, results in comparison_results.items():
            score = 0
            
            # Penalize DC bias (mean should be close to 0)
            if abs(results['mean_value']) < 1.0:
                score += 30
            elif abs(results['mean_value']) < 10.0:
                score += 20
            elif abs(results['mean_value']) < 100.0:
                score += 10
            
            # Reward full range usage
            if results['uses_full_range']:
                score += 20
            
            # Reward more unique values (better quantization)
            if results['unique_values'] > 20000:
                score += 20
            elif results['unique_values'] > 10000:
                score += 15
            elif results['unique_values'] > 5000:
                score += 10
            
            # Consider performance (faster is better)
            if results['mean_time_ms'] < 1.0:
                score += 20
            elif results['mean_time_ms'] < 2.0:
                score += 15
            elif results['mean_time_ms'] < 5.0:
                score += 10
            
            # Stability bonus (low std deviation in timing)
            if results['std_time_ms'] < 0.1:
                score += 10
            elif results['std_time_ms'] < 0.5:
                score += 5
            
            scores[method_name] = score
        
        # Find method with highest score
        optimal = max(scores, key=scores.get)
        
        return optimal


def demonstrate_converters():
    """Demonstrate PCM16 converter comparison"""
    print("PCM16 Converter Comparison")
    print("=" * 50)
    
    # Generate test signal
    sample_rate = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Complex test signal
    test_signal = np.sin(2 * np.pi * 440 * t) * 0.5  # Main tone
    test_signal += 0.3 * np.sin(2 * np.pi * 880 * t)  # Harmonic
    test_signal += 0.1 * np.random.randn(len(t))  # Noise
    
    print("\nComparing conversion methods...")
    results = PCM16Converter.compare_methods(test_signal)
    
    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  Range: [{metrics['min_value']}, {metrics['max_value']}]")
        print(f"  Mean: {metrics['mean_value']:.2f}")
        print(f"  Unique Values: {metrics['unique_values']}")
        print(f"  Performance: {metrics['mean_time_ms']:.3f} ± {metrics['std_time_ms']:.3f} ms")
        print(f"  Uses Full Range: {metrics['uses_full_range']}")
    
    print("\n" + "=" * 50)
    optimal = PCM16Converter.recommend_optimal_method(results)
    print(f"Recommended Method: {optimal}")


if __name__ == "__main__":
    demonstrate_converters()