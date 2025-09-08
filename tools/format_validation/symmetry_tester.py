#!/usr/bin/env python3
"""
Symmetry Tester for PCM16 Conversion
Part of Task 1.4 - Audio Format Validation

This module tests clipping symmetry in PCM16 conversion to ensure equal headroom
in positive and negative directions, preventing asymmetric distortion.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class SymmetryMetrics:
    """Symmetry test results"""
    positive_peak: int
    negative_peak: int
    positive_headroom: int
    negative_headroom: int
    symmetry_ratio: float
    asymmetry_samples: int
    is_symmetric: bool
    symmetry_score: float


class SymmetryTester:
    """Test and validate clipping symmetry in PCM16 conversion"""
    
    def __init__(self):
        self.symmetry_threshold = 0.01  # 1% tolerance for symmetry
        self.int16_max = 32767
        self.int16_min = -32768
    
    def test_clipping_symmetry(self, conversion_method: Callable) -> SymmetryMetrics:
        """
        Test if conversion method provides symmetric clipping
        
        Args:
            conversion_method: Function that converts float32 to pcm16
            
        Returns:
            SymmetryMetrics with detailed analysis
        """
        # Test at various amplitude levels
        test_amplitudes = [0.5, 0.9, 0.99, 0.999, 1.0, 1.1]
        
        positive_peaks = []
        negative_peaks = []
        
        for amplitude in test_amplitudes:
            # Test positive signal
            positive_signal = np.ones(1000) * amplitude
            positive_pcm = conversion_method(positive_signal)
            positive_peaks.append(np.max(positive_pcm))
            
            # Test negative signal
            negative_signal = np.ones(1000) * -amplitude
            negative_pcm = conversion_method(negative_signal)
            negative_peaks.append(np.min(negative_pcm))
        
        # Find actual peaks achieved
        max_positive = max(positive_peaks)
        max_negative = min(negative_peaks)
        
        # Calculate headroom
        positive_headroom = self.int16_max - max_positive
        negative_headroom = max_negative - self.int16_min
        
        # Calculate symmetry ratio
        if max_negative != 0:
            symmetry_ratio = abs(max_positive) / abs(max_negative)
        else:
            symmetry_ratio = float('inf')
        
        # Count asymmetry (difference in absolute peaks)
        asymmetry = abs(max_positive) - abs(max_negative)
        
        # Determine if symmetric (within threshold)
        is_symmetric = abs(symmetry_ratio - 1.0) < self.symmetry_threshold
        
        # Calculate symmetry score (0-100)
        if symmetry_ratio == float('inf'):
            score = 0.0
        else:
            deviation = abs(symmetry_ratio - 1.0)
            score = max(0, 100 * (1 - deviation))
        
        return SymmetryMetrics(
            positive_peak=max_positive,
            negative_peak=max_negative,
            positive_headroom=positive_headroom,
            negative_headroom=negative_headroom,
            symmetry_ratio=symmetry_ratio,
            asymmetry_samples=asymmetry,
            is_symmetric=is_symmetric,
            symmetry_score=score
        )
    
    def test_edge_cases(self, conversion_method: Callable) -> Dict[str, Dict]:
        """
        Test edge cases for symmetry
        
        Args:
            conversion_method: Conversion function to test
            
        Returns:
            Dictionary of edge case results
        """
        results = {}
        
        # Edge case 1: Exactly ±1.0
        pos_one = np.ones(100) * 1.0
        neg_one = np.ones(100) * -1.0
        pos_pcm = conversion_method(pos_one)
        neg_pcm = conversion_method(neg_one)
        
        results['unity_values'] = {
            'positive_result': int(np.max(pos_pcm)),
            'negative_result': int(np.min(neg_pcm)),
            'symmetric': abs(np.max(pos_pcm)) == abs(np.min(neg_pcm))
        }
        
        # Edge case 2: Very small values (test quantization symmetry)
        small_val = 1.0 / 32768  # Smallest representable value
        pos_small = np.ones(100) * small_val
        neg_small = np.ones(100) * -small_val
        pos_small_pcm = conversion_method(pos_small)
        neg_small_pcm = conversion_method(neg_small)
        
        results['minimum_values'] = {
            'positive_result': int(np.mean(pos_small_pcm)),
            'negative_result': int(np.mean(neg_small_pcm)),
            'symmetric': abs(np.mean(pos_small_pcm)) == abs(np.mean(neg_small_pcm))
        }
        
        # Edge case 3: Clipping behavior
        over_pos = np.ones(100) * 1.5
        over_neg = np.ones(100) * -1.5
        over_pos_pcm = conversion_method(over_pos)
        over_neg_pcm = conversion_method(over_neg)
        
        results['clipping_behavior'] = {
            'positive_clipped_to': int(np.max(over_pos_pcm)),
            'negative_clipped_to': int(np.min(over_neg_pcm)),
            'uses_full_range': (np.min(over_neg_pcm) == self.int16_min) or 
                               (np.max(over_pos_pcm) == self.int16_max)
        }
        
        # Edge case 4: Zero crossing symmetry
        t = np.linspace(0, 1, 1000)
        sine = np.sin(2 * np.pi * t)
        sine_pcm = conversion_method(sine)
        
        # Count positive and negative samples
        pos_count = np.sum(sine_pcm > 0)
        neg_count = np.sum(sine_pcm < 0)
        zero_count = np.sum(sine_pcm == 0)
        
        results['zero_crossing'] = {
            'positive_samples': pos_count,
            'negative_samples': neg_count,
            'zero_samples': zero_count,
            'balanced': abs(pos_count - neg_count) < 10  # Allow small difference
        }
        
        return results
    
    def compare_symmetry_methods(self) -> Dict[str, SymmetryMetrics]:
        """
        Compare symmetry of different scaling methods
        
        Returns:
            Comparison of symmetry metrics for different methods
        """
        methods = {}
        
        # Method 1: Symmetric scaling (32767)
        def symmetric_32767(x):
            return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
        methods['symmetric_32767'] = symmetric_32767
        
        # Method 2: Asymmetric scaling (32768)
        def asymmetric_32768(x):
            return (np.clip(x, -1.0, 1.0) * 32768).astype(np.int16)
        methods['asymmetric_32768'] = asymmetric_32768
        
        # Method 3: Hybrid approach
        def hybrid_scaling(x):
            x = np.clip(x, -1.0, 1.0)
            result = np.zeros_like(x, dtype=np.int16)
            negative = x < 0
            result[negative] = (x[negative] * 32768).astype(np.int16)
            result[~negative] = (x[~negative] * 32767).astype(np.int16)
            return result
        methods['hybrid_scaling'] = hybrid_scaling
        
        # Method 4: Clamped symmetric
        def clamped_symmetric(x):
            scaled = np.clip(x, -1.0, 1.0) * 32767
            return np.clip(scaled, -32768, 32767).astype(np.int16)
        methods['clamped_symmetric'] = clamped_symmetric
        
        results = {}
        for name, method in methods.items():
            results[name] = self.test_clipping_symmetry(method)
        
        return results
    
    def visualize_symmetry(self, conversion_method: Callable) -> Dict:
        """
        Generate data for symmetry visualization
        
        Args:
            conversion_method: Conversion function to analyze
            
        Returns:
            Visualization data
        """
        # Generate transfer function data
        input_values = np.linspace(-1.5, 1.5, 1000)
        output_values = []
        
        for val in input_values:
            signal = np.array([val])
            pcm = conversion_method(signal)
            output_values.append(pcm[0])
        
        output_values = np.array(output_values)
        
        # Find key points
        positive_clip_idx = np.where(output_values == np.max(output_values))[0]
        negative_clip_idx = np.where(output_values == np.min(output_values))[0]
        
        if len(positive_clip_idx) > 0:
            positive_clip_point = input_values[positive_clip_idx[0]]
        else:
            positive_clip_point = None
        
        if len(negative_clip_idx) > 0:
            negative_clip_point = input_values[negative_clip_idx[-1]]
        else:
            negative_clip_point = None
        
        return {
            'input_values': input_values.tolist(),
            'output_values': output_values.tolist(),
            'positive_clip_point': positive_clip_point,
            'negative_clip_point': negative_clip_point,
            'max_output': int(np.max(output_values)),
            'min_output': int(np.min(output_values))
        }
    
    def generate_symmetry_report(self, conversion_method: Callable) -> Dict:
        """
        Generate comprehensive symmetry test report
        
        Args:
            conversion_method: Conversion function to test
            
        Returns:
            Detailed test report
        """
        metrics = self.test_clipping_symmetry(conversion_method)
        edge_cases = self.test_edge_cases(conversion_method)
        
        report = {
            'method_name': conversion_method.__name__ if hasattr(conversion_method, '__name__') else 'unknown',
            'symmetry_metrics': {
                'positive_peak': metrics.positive_peak,
                'negative_peak': metrics.negative_peak,
                'symmetry_ratio': metrics.symmetry_ratio,
                'is_symmetric': metrics.is_symmetric,
                'symmetry_score': metrics.symmetry_score
            },
            'edge_cases': edge_cases,
            'recommendations': []
        }
        
        # Generate recommendations
        if metrics.is_symmetric:
            report['recommendations'].append(
                f"✓ Clipping is symmetric (ratio: {metrics.symmetry_ratio:.4f})"
            )
        else:
            report['recommendations'].append(
                f"⚠ Clipping is asymmetric (ratio: {metrics.symmetry_ratio:.4f}, target: 1.0)"
            )
        
        if edge_cases['unity_values']['symmetric']:
            report['recommendations'].append("✓ Unity values (±1.0) are handled symmetrically")
        else:
            report['recommendations'].append("⚠ Unity values show asymmetry")
        
        if edge_cases['zero_crossing']['balanced']:
            report['recommendations'].append("✓ Zero crossings are balanced")
        else:
            report['recommendations'].append("⚠ Zero crossing imbalance detected")
        
        # Overall assessment
        if metrics.symmetry_score >= 95:
            report['overall_assessment'] = "EXCELLENT - Near-perfect symmetry"
        elif metrics.symmetry_score >= 90:
            report['overall_assessment'] = "GOOD - Acceptable symmetry"
        elif metrics.symmetry_score >= 80:
            report['overall_assessment'] = "FAIR - Some asymmetry present"
        else:
            report['overall_assessment'] = "POOR - Significant asymmetry detected"
        
        return report


def demonstrate_symmetry():
    """Demonstrate symmetry testing"""
    print("Symmetry Tester Demonstration")
    print("=" * 50)
    
    tester = SymmetryTester()
    
    # Test current implementation
    def current_method(audio: np.ndarray) -> np.ndarray:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    print("\nTesting current PCM16 conversion method (32767 scaling)...")
    metrics = tester.test_clipping_symmetry(current_method)
    
    print(f"\nSymmetry Metrics:")
    print(f"  Positive Peak: {metrics.positive_peak}")
    print(f"  Negative Peak: {metrics.negative_peak}")
    print(f"  Symmetry Ratio: {metrics.symmetry_ratio:.4f}")
    print(f"  Is Symmetric: {metrics.is_symmetric}")
    print(f"  Symmetry Score: {metrics.symmetry_score:.1f}/100")
    
    print("\n" + "=" * 50)
    print("Edge Case Testing...")
    edge_cases = tester.test_edge_cases(current_method)
    
    for case_name, results in edge_cases.items():
        print(f"\n{case_name}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Method Comparison...")
    comparison = tester.compare_symmetry_methods()
    
    for method_name, metrics in comparison.items():
        print(f"\n{method_name}:")
        print(f"  Symmetry Ratio: {metrics.symmetry_ratio:.4f}")
        print(f"  Symmetry Score: {metrics.symmetry_score:.1f}/100")
        print(f"  Assessment: {'PASS' if metrics.is_symmetric else 'FAIL'}")


if __name__ == "__main__":
    demonstrate_symmetry()