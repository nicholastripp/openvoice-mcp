#!/usr/bin/env python3
"""
DC Bias Analyzer for PCM16 Conversion
Part of Task 1.4 - Audio Format Validation

This module specifically analyzes DC bias introduced during Float32 to PCM16 conversion,
identifying sources of DC offset and providing correction methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class DCBiasMetrics:
    """DC bias measurement results"""
    mean_dc_offset: float
    dc_offset_samples: float
    dc_offset_percent: float
    dc_drift: float
    dc_stability: str
    requires_correction: bool
    correction_factor: float


class DCBiasAnalyzer:
    """Analyze and correct DC bias in PCM16 conversion"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.dc_threshold_percent = 0.001  # 0.001% is our target
    
    def analyze_dc_bias(self, pcm16_data: np.ndarray) -> DCBiasMetrics:
        """
        Comprehensive DC bias analysis
        
        Args:
            pcm16_data: PCM16 audio data
            
        Returns:
            DCBiasMetrics with detailed analysis
        """
        # Calculate mean DC offset
        mean_dc = np.mean(pcm16_data)
        
        # Convert to percentage of full scale
        dc_percent = (mean_dc / 32768) * 100
        
        # Analyze DC drift over time
        window_size = min(1000, len(pcm16_data) // 10)
        if window_size > 0:
            windows = len(pcm16_data) // window_size
            dc_values = []
            for i in range(windows):
                window_data = pcm16_data[i*window_size:(i+1)*window_size]
                dc_values.append(np.mean(window_data))
            
            if len(dc_values) > 1:
                dc_drift = np.std(dc_values)
            else:
                dc_drift = 0.0
        else:
            dc_drift = 0.0
        
        # Determine stability
        if dc_drift < 1.0:
            stability = "stable"
        elif dc_drift < 10.0:
            stability = "moderate"
        else:
            stability = "unstable"
        
        # Determine if correction is needed
        requires_correction = abs(dc_percent) > self.dc_threshold_percent
        
        # Calculate correction factor
        correction_factor = -mean_dc if requires_correction else 0.0
        
        return DCBiasMetrics(
            mean_dc_offset=mean_dc,
            dc_offset_samples=mean_dc,
            dc_offset_percent=dc_percent,
            dc_drift=dc_drift,
            dc_stability=stability,
            requires_correction=requires_correction,
            correction_factor=correction_factor
        )
    
    def measure_conversion_dc_bias(self, conversion_method: Callable,
                                  test_duration: float = 1.0) -> Dict[str, DCBiasMetrics]:
        """
        Measure DC bias introduced by a conversion method
        
        Args:
            conversion_method: Function that converts float32 to pcm16
            test_duration: Duration of test signals in seconds
            
        Returns:
            Dictionary of test results
        """
        results = {}
        samples = int(self.sample_rate * test_duration)
        
        # Test 1: Zero signal (should produce zero DC)
        zero_signal = np.zeros(samples)
        zero_pcm = conversion_method(zero_signal)
        results['zero_signal'] = self.analyze_dc_bias(zero_pcm)
        
        # Test 2: Small positive DC offset
        small_dc = np.ones(samples) * 0.01
        small_pcm = conversion_method(small_dc)
        results['small_positive_dc'] = self.analyze_dc_bias(small_pcm)
        
        # Test 3: Small negative DC offset
        neg_dc = np.ones(samples) * -0.01
        neg_pcm = conversion_method(neg_dc)
        results['small_negative_dc'] = self.analyze_dc_bias(neg_pcm)
        
        # Test 4: Sine wave (should have zero DC)
        t = np.linspace(0, test_duration, samples)
        sine = np.sin(2 * np.pi * 1000 * t)
        sine_pcm = conversion_method(sine)
        results['sine_wave'] = self.analyze_dc_bias(sine_pcm)
        
        # Test 5: Complex signal (speech-like)
        complex_signal = sine * 0.5
        complex_signal += 0.3 * np.sin(2 * np.pi * 200 * t)
        complex_signal += 0.1 * np.random.randn(samples)
        complex_pcm = conversion_method(complex_signal)
        results['complex_signal'] = self.analyze_dc_bias(complex_pcm)
        
        return results
    
    def compare_scaling_factors(self) -> Dict[str, Dict]:
        """
        Compare DC bias between different scaling factors
        
        Returns:
            Comparison results for 32767 vs 32768 scaling
        """
        samples = self.sample_rate
        results = {}
        
        # Test signals
        test_signals = {
            'zero': np.zeros(samples),
            'positive_full': np.ones(samples) * 0.999,
            'negative_full': np.ones(samples) * -0.999,
            'sine': np.sin(2 * np.pi * 1000 * np.linspace(0, 1, samples)),
            'square': np.sign(np.sin(2 * np.pi * 100 * np.linspace(0, 1, samples)))
        }
        
        # Method 1: Scale by 32767 (symmetric)
        def scale_32767(x):
            return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Method 2: Scale by 32768 (asymmetric)
        def scale_32768(x):
            x_clipped = np.clip(x, -1.0, 1.0)
            # Traditional approach - can cause slight negative bias
            return (x_clipped * 32768).astype(np.int16)
        
        # Method 3: Hybrid (different scaling for positive/negative)
        def scale_hybrid(x):
            x_clipped = np.clip(x, -1.0, 1.0)
            result = np.zeros_like(x_clipped, dtype=np.int16)
            negative_mask = x_clipped < 0
            result[negative_mask] = (x_clipped[negative_mask] * 32768).astype(np.int16)
            result[~negative_mask] = (x_clipped[~negative_mask] * 32767).astype(np.int16)
            return result
        
        methods = {
            '32767_symmetric': scale_32767,
            '32768_asymmetric': scale_32768,
            'hybrid_approach': scale_hybrid
        }
        
        for method_name, method in methods.items():
            method_results = {}
            
            for signal_name, signal in test_signals.items():
                pcm16 = method(signal)
                metrics = self.analyze_dc_bias(pcm16)
                method_results[signal_name] = {
                    'dc_offset': metrics.mean_dc_offset,
                    'dc_percent': metrics.dc_offset_percent,
                    'requires_correction': metrics.requires_correction
                }
            
            # Calculate average DC bias across all signals
            avg_dc = np.mean([r['dc_offset'] for r in method_results.values()])
            avg_percent = np.mean([abs(r['dc_percent']) for r in method_results.values()])
            
            results[method_name] = {
                'signals': method_results,
                'average_dc_offset': avg_dc,
                'average_dc_percent': avg_percent,
                'recommended': avg_percent < self.dc_threshold_percent
            }
        
        return results
    
    def apply_dc_correction(self, pcm16_data: np.ndarray, 
                          metrics: Optional[DCBiasMetrics] = None) -> np.ndarray:
        """
        Apply DC bias correction to PCM16 data
        
        Args:
            pcm16_data: Input PCM16 data
            metrics: Pre-calculated metrics (optional)
            
        Returns:
            Corrected PCM16 data
        """
        if metrics is None:
            metrics = self.analyze_dc_bias(pcm16_data)
        
        if not metrics.requires_correction:
            return pcm16_data
        
        # Apply correction
        corrected = pcm16_data.astype(np.float32) + metrics.correction_factor
        
        # Ensure we stay within int16 bounds
        corrected = np.clip(corrected, -32768, 32767)
        
        return corrected.astype(np.int16)
    
    def generate_dc_test_report(self, conversion_method: Callable) -> Dict:
        """
        Generate comprehensive DC bias test report
        
        Args:
            conversion_method: Conversion function to test
            
        Returns:
            Detailed test report
        """
        report = {
            'method_name': conversion_method.__name__ if hasattr(conversion_method, '__name__') else 'unknown',
            'dc_measurements': self.measure_conversion_dc_bias(conversion_method),
            'scaling_comparison': self.compare_scaling_factors(),
            'recommendations': []
        }
        
        # Analyze results and generate recommendations
        zero_dc = report['dc_measurements']['zero_signal'].dc_offset_percent
        sine_dc = report['dc_measurements']['sine_wave'].dc_offset_percent
        
        if abs(zero_dc) < self.dc_threshold_percent:
            report['recommendations'].append(
                f"✓ Zero signal DC bias is excellent ({zero_dc:.6f}%)"
            )
        else:
            report['recommendations'].append(
                f"⚠ Zero signal has DC bias of {zero_dc:.6f}% (target: <{self.dc_threshold_percent}%)"
            )
        
        if abs(sine_dc) < self.dc_threshold_percent:
            report['recommendations'].append(
                f"✓ Sine wave DC bias is excellent ({sine_dc:.6f}%)"
            )
        else:
            report['recommendations'].append(
                f"⚠ Sine wave has DC bias of {sine_dc:.6f}% (target: <{self.dc_threshold_percent}%)"
            )
        
        # Overall assessment
        all_good = all(
            abs(m.dc_offset_percent) < self.dc_threshold_percent
            for m in report['dc_measurements'].values()
        )
        
        if all_good:
            report['overall_assessment'] = "PASS - DC bias within acceptable limits"
        else:
            report['overall_assessment'] = "NEEDS IMPROVEMENT - DC bias exceeds threshold"
        
        return report


def demonstrate_dc_bias():
    """Demonstrate DC bias analysis"""
    print("DC Bias Analyzer Demonstration")
    print("=" * 50)
    
    analyzer = DCBiasAnalyzer()
    
    # Test the current implementation method
    def current_method(audio: np.ndarray) -> np.ndarray:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    print("\nTesting current PCM16 conversion method (32767 scaling)...")
    results = analyzer.measure_conversion_dc_bias(current_method)
    
    for signal_type, metrics in results.items():
        print(f"\n{signal_type}:")
        print(f"  DC Offset: {metrics.mean_dc_offset:.2f} samples")
        print(f"  DC Percent: {metrics.dc_offset_percent:.6f}%")
        print(f"  DC Stability: {metrics.dc_stability}")
        print(f"  Needs Correction: {metrics.requires_correction}")
    
    print("\n" + "=" * 50)
    print("Comparing scaling factors...")
    comparison = analyzer.compare_scaling_factors()
    
    for method, results in comparison.items():
        print(f"\n{method}:")
        print(f"  Average DC Offset: {results['average_dc_offset']:.4f} samples")
        print(f"  Average DC Percent: {results['average_dc_percent']:.6f}%")
        print(f"  Recommended: {'Yes' if results['recommended'] else 'No'}")


if __name__ == "__main__":
    demonstrate_dc_bias()