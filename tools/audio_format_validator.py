#!/usr/bin/env python3
"""
Audio Format Validator for HA Realtime Voice Assistant
Task 1.4 - PCM16 Conversion Validation and Optimization

This tool validates and optimizes the Float32 to PCM16 conversion process,
ensuring symmetric clipping, zero DC bias, and proper bit depth preservation
for optimal OpenAI Realtime API compatibility.

Usage:
    python tools/audio_format_validator.py --full-test
    python tools/audio_format_validator.py --quick-test
    python tools/audio_format_validator.py --openai-test
    python tools/audio_format_validator.py --compare-methods
"""

import sys
import os
import argparse
import numpy as np
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from math import gcd
import struct

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import audio pipeline components
try:
    from audio.capture import AudioCapture
    from config import load_config
    from utils.logger import setup_logging, get_logger
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    warnings.warn("Pipeline components not available - some features will be limited")

# Import scipy for signal processing
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some analysis features will be limited")

# Import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization disabled")


@dataclass
class ConversionMetrics:
    """Metrics for a single conversion method"""
    method_name: str
    dc_bias: float
    dc_bias_percent: float
    positive_peak: int
    negative_peak: int
    symmetry_ratio: float
    clipping_count: int
    clipping_ratio: float
    quantization_noise: float
    thd: float
    snr: float
    round_trip_error: float
    effective_bits: float
    headroom_db: float
    processing_time: float


@dataclass
class TestSignal:
    """Test signal definition"""
    name: str
    signal: np.ndarray
    sample_rate: int
    description: str
    expected_issues: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    test_duration: float
    conversion_methods: List[ConversionMetrics]
    test_signals: List[str]
    optimal_method: str
    recommendations: List[str]
    dc_bias_analysis: Dict[str, Any]
    symmetry_analysis: Dict[str, Any]
    quality_analysis: Dict[str, Any]
    openai_compatibility: Dict[str, Any]
    conclusion: str


class AudioFormatValidator:
    """Main audio format validation class"""
    
    def __init__(self, sample_rate: int = 24000, verbose: bool = True):
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.conversion_methods = self._define_conversion_methods()
        self.test_signals = {}
        self.test_results = {}
        
        # Setup logging
        if PIPELINE_AVAILABLE:
            setup_logging()
            self.logger = get_logger(__name__)
        else:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
    
    def _define_conversion_methods(self) -> Dict[str, Callable]:
        """Define various PCM16 conversion methods to test"""
        methods = {}
        
        # Method 1: Current implementation (symmetric, 32767)
        def current_method(audio: np.ndarray) -> np.ndarray:
            audio = np.clip(audio, -1.0, 1.0)
            return (audio * 32767).astype(np.int16)
        methods['current_32767'] = current_method
        
        # Method 2: Alternative with 32768 (asymmetric)
        def asymmetric_method(audio: np.ndarray) -> np.ndarray:
            audio = np.clip(audio, -1.0, 1.0)
            # Scale negative values by 32768, positive by 32767
            result = np.zeros_like(audio, dtype=np.int16)
            negative_mask = audio < 0
            result[negative_mask] = (audio[negative_mask] * 32768).astype(np.int16)
            result[~negative_mask] = (audio[~negative_mask] * 32767).astype(np.int16)
            return result
        methods['asymmetric_32768'] = asymmetric_method
        
        # Method 3: Traditional 32768 (simple multiplication)
        def traditional_method(audio: np.ndarray) -> np.ndarray:
            audio = np.clip(audio, -1.0, 1.0)
            return (audio * 32768).astype(np.int16)
        methods['traditional_32768'] = traditional_method
        
        # Method 4: Clamped with explicit clipping
        def clamped_method(audio: np.ndarray) -> np.ndarray:
            audio = np.clip(audio, -1.0, 1.0)
            scaled = audio * 32767
            return np.clip(scaled, -32768, 32767).astype(np.int16)
        methods['clamped_32767'] = clamped_method
        
        # Method 5: Dithered conversion (for low-level signals)
        def dithered_method(audio: np.ndarray) -> np.ndarray:
            audio = np.clip(audio, -1.0, 1.0)
            # Add triangular dither
            dither = np.random.triangular(-0.5, 0, 0.5, size=audio.shape) / 32768
            dithered = audio + dither
            dithered = np.clip(dithered, -1.0, 1.0)
            return (dithered * 32767).astype(np.int16)
        methods['dithered_32767'] = dithered_method
        
        return methods
    
    def generate_test_signals(self, duration: float = 1.0) -> Dict[str, TestSignal]:
        """Generate comprehensive test signals"""
        signals = {}
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # 1. Pure sine waves at different frequencies
        for freq in [440, 1000, 3000]:
            signal = np.sin(2 * np.pi * freq * t)
            signals[f'sine_{freq}hz'] = TestSignal(
                name=f'sine_{freq}hz',
                signal=signal,
                sample_rate=self.sample_rate,
                description=f'Pure {freq}Hz sine wave'
            )
        
        # 2. DC offset tests
        for offset in [-0.1, 0.05, 0.1]:
            signal = np.ones(samples) * offset
            signals[f'dc_offset_{offset}'] = TestSignal(
                name=f'dc_offset_{offset}',
                signal=signal,
                sample_rate=self.sample_rate,
                description=f'DC offset of {offset}',
                expected_issues=['dc_bias']
            )
        
        # 3. Full scale signals
        signals['full_scale_positive'] = TestSignal(
            name='full_scale_positive',
            signal=np.ones(samples) * 0.999,
            sample_rate=self.sample_rate,
            description='Near full scale positive'
        )
        
        signals['full_scale_negative'] = TestSignal(
            name='full_scale_negative',
            signal=np.ones(samples) * -0.999,
            sample_rate=self.sample_rate,
            description='Near full scale negative'
        )
        
        # 4. Quiet signal (test quantization)
        quiet_signal = np.sin(2 * np.pi * 1000 * t) * 0.001
        signals['quiet_signal'] = TestSignal(
            name='quiet_signal',
            signal=quiet_signal,
            sample_rate=self.sample_rate,
            description='Very quiet signal (0.1% amplitude)',
            expected_issues=['quantization_noise']
        )
        
        # 5. White noise
        signals['white_noise'] = TestSignal(
            name='white_noise',
            signal=np.random.randn(samples) * 0.5,
            sample_rate=self.sample_rate,
            description='White noise at 50% amplitude'
        )
        
        # 6. Impulse
        impulse = np.zeros(samples)
        impulse[samples // 2] = 1.0
        signals['impulse'] = TestSignal(
            name='impulse',
            signal=impulse,
            sample_rate=self.sample_rate,
            description='Single sample impulse'
        )
        
        # 7. Complex speech-like signal
        speech_like = np.sin(2 * np.pi * 200 * t)  # Fundamental
        speech_like += 0.5 * np.sin(2 * np.pi * 400 * t)  # Harmonic
        speech_like += 0.3 * np.sin(2 * np.pi * 800 * t)  # Harmonic
        speech_like += 0.1 * np.random.randn(samples)  # Noise
        speech_like *= 0.7  # Scale to reasonable level
        signals['speech_like'] = TestSignal(
            name='speech_like',
            signal=speech_like,
            sample_rate=self.sample_rate,
            description='Complex speech-like signal'
        )
        
        # 8. Clipping test signal
        clipping_test = np.sin(2 * np.pi * 1000 * t) * 1.2  # Intentionally exceed ±1
        signals['clipping_test'] = TestSignal(
            name='clipping_test',
            signal=clipping_test,
            sample_rate=self.sample_rate,
            description='Over-range signal to test clipping',
            expected_issues=['clipping']
        )
        
        self.test_signals = signals
        return signals
    
    def analyze_conversion_method(self, method: Callable, signal: np.ndarray, 
                                 method_name: str) -> ConversionMetrics:
        """Analyze a single conversion method with a test signal"""
        start_time = time.time()
        
        # Perform conversion
        pcm16_data = method(signal)
        
        # Calculate DC bias
        dc_bias = np.mean(pcm16_data)
        dc_bias_percent = (dc_bias / 32768) * 100
        
        # Find peaks
        positive_peak = np.max(pcm16_data)
        negative_peak = np.min(pcm16_data)
        
        # Calculate symmetry ratio
        symmetry_ratio = abs(positive_peak) / abs(negative_peak) if negative_peak != 0 else float('inf')
        
        # Count clipping
        clipping_count = np.sum((pcm16_data == 32767) | (pcm16_data == -32768))
        clipping_ratio = clipping_count / len(pcm16_data)
        
        # Convert back to float for quality metrics
        reconstructed = pcm16_data.astype(np.float32) / 32768.0
        
        # Calculate round-trip error
        round_trip_error = np.sqrt(np.mean((signal[:len(reconstructed)] - reconstructed) ** 2))
        
        # Calculate THD if scipy available
        if SCIPY_AVAILABLE and np.any(signal != 0):
            thd = self._calculate_thd(reconstructed)
            snr = self._calculate_snr(signal[:len(reconstructed)], reconstructed)
        else:
            thd = 0.0
            snr = 0.0
        
        # Calculate quantization noise
        quantization_noise = np.std(signal[:len(reconstructed)] - reconstructed)
        
        # Calculate effective bits
        if quantization_noise > 0:
            effective_bits = -np.log2(quantization_noise)
        else:
            effective_bits = 16.0
        
        # Calculate headroom
        max_amplitude = max(abs(positive_peak), abs(negative_peak))
        if max_amplitude > 0:
            headroom_db = 20 * np.log10(32768 / max_amplitude)
        else:
            headroom_db = float('inf')
        
        processing_time = time.time() - start_time
        
        return ConversionMetrics(
            method_name=method_name,
            dc_bias=dc_bias,
            dc_bias_percent=dc_bias_percent,
            positive_peak=positive_peak,
            negative_peak=negative_peak,
            symmetry_ratio=symmetry_ratio,
            clipping_count=clipping_count,
            clipping_ratio=clipping_ratio,
            quantization_noise=quantization_noise,
            thd=thd,
            snr=snr,
            round_trip_error=round_trip_error,
            effective_bits=effective_bits,
            headroom_db=headroom_db,
            processing_time=processing_time
        )
    
    def _calculate_thd(self, signal: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        if not SCIPY_AVAILABLE:
            return 0.0
        
        # Perform FFT
        fft_vals = np.abs(fft(signal))
        fft_vals = fft_vals[:len(fft_vals) // 2]
        
        # Find fundamental frequency (highest peak)
        fundamental_idx = np.argmax(fft_vals[1:]) + 1
        fundamental_power = fft_vals[fundamental_idx] ** 2
        
        # Sum harmonic powers (2x, 3x, 4x, 5x fundamental)
        harmonic_power = 0
        for harmonic in range(2, 6):
            idx = fundamental_idx * harmonic
            if idx < len(fft_vals):
                harmonic_power += fft_vals[idx] ** 2
        
        # Calculate THD
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power) * 100
        else:
            thd = 0.0
        
        return thd
    
    def _calculate_snr(self, original: np.ndarray, converted: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(original ** 2)
        noise = original - converted
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
    
    def test_dc_bias(self) -> Dict[str, Any]:
        """Comprehensive DC bias testing"""
        results = {}
        
        # Test with pure DC signals
        dc_levels = [-0.5, -0.1, 0.0, 0.1, 0.5]
        
        for method_name, method in self.conversion_methods.items():
            method_results = []
            
            for dc_level in dc_levels:
                # Create DC signal
                signal = np.ones(self.sample_rate) * dc_level
                
                # Convert and measure
                pcm16 = method(signal)
                measured_dc = np.mean(pcm16)
                expected_dc = dc_level * 32767  # Expected if perfect
                error = measured_dc - expected_dc
                
                method_results.append({
                    'input_dc': dc_level,
                    'expected_output': expected_dc,
                    'measured_output': measured_dc,
                    'error': error,
                    'error_percent': (error / 32768) * 100 if expected_dc != 0 else 0
                })
            
            results[method_name] = method_results
        
        return results
    
    def test_symmetry(self) -> Dict[str, Any]:
        """Test clipping symmetry"""
        results = {}
        
        # Test signals at various amplitudes
        amplitudes = [0.5, 0.9, 0.99, 0.999, 1.0, 1.1]
        
        for method_name, method in self.conversion_methods.items():
            method_results = []
            
            for amplitude in amplitudes:
                # Create positive and negative signals
                positive_signal = np.ones(1000) * amplitude
                negative_signal = np.ones(1000) * -amplitude
                
                # Convert
                positive_pcm = method(positive_signal)
                negative_pcm = method(negative_signal)
                
                # Measure peaks
                pos_peak = np.max(positive_pcm)
                neg_peak = np.min(negative_pcm)
                
                # Calculate symmetry
                symmetry = abs(pos_peak) - abs(neg_peak)
                
                method_results.append({
                    'amplitude': amplitude,
                    'positive_peak': pos_peak,
                    'negative_peak': neg_peak,
                    'asymmetry': symmetry,
                    'symmetric': abs(symmetry) <= 1
                })
            
            results[method_name] = method_results
        
        return results
    
    def test_bit_depth_preservation(self) -> Dict[str, Any]:
        """Test bit depth preservation and quantization"""
        results = {}
        
        # Generate signals at different bit depths
        bit_depths = [8, 12, 14, 16]
        
        for method_name, method in self.conversion_methods.items():
            method_results = []
            
            for bits in bit_depths:
                # Create signal with limited bit depth
                levels = 2 ** bits
                step = 2.0 / levels
                signal = np.linspace(-1.0 + step/2, 1.0 - step/2, levels)
                signal = np.tile(signal, 10)  # Repeat for better statistics
                
                # Add small noise to test quantization
                signal += np.random.randn(len(signal)) * (step / 10)
                signal = np.clip(signal, -1.0, 1.0)
                
                # Convert and measure
                pcm16 = method(signal)
                
                # Count unique values (effective levels)
                unique_values = len(np.unique(pcm16))
                expected_values = min(levels, 65536)
                
                # Calculate effective bit depth
                if unique_values > 1:
                    effective_bits = np.log2(unique_values)
                else:
                    effective_bits = 0
                
                method_results.append({
                    'input_bits': bits,
                    'expected_levels': expected_values,
                    'measured_levels': unique_values,
                    'effective_bits': effective_bits,
                    'preservation_ratio': unique_values / expected_values
                })
            
            results[method_name] = method_results
        
        return results
    
    def test_openai_compatibility(self) -> Dict[str, Any]:
        """Test OpenAI Realtime API compatibility"""
        results = {
            'format_check': {},
            'endianness_check': {},
            'size_check': {},
            'range_check': {}
        }
        
        for method_name, method in self.conversion_methods.items():
            # Generate test signal
            t = np.linspace(0, 1, self.sample_rate)
            signal = np.sin(2 * np.pi * 440 * t) * 0.5
            
            # Convert
            pcm16 = method(signal)
            
            # Check format requirements
            results['format_check'][method_name] = {
                'dtype': str(pcm16.dtype),
                'is_int16': pcm16.dtype == np.int16,
                'itemsize': pcm16.itemsize,
                'correct_size': pcm16.itemsize == 2
            }
            
            # Check endianness
            bytes_data = pcm16.tobytes()
            # Verify little-endian by checking known value
            test_val = np.array([1000], dtype=np.int16)
            test_bytes = test_val.tobytes()
            is_little_endian = test_bytes == struct.pack('<h', 1000)
            
            results['endianness_check'][method_name] = {
                'system_endianness': sys.byteorder,
                'is_little_endian': is_little_endian,
                'bytes_sample': bytes_data[:10].hex()
            }
            
            # Check size
            expected_size = len(signal) * 2  # 2 bytes per sample
            actual_size = len(bytes_data)
            
            results['size_check'][method_name] = {
                'expected_bytes': expected_size,
                'actual_bytes': actual_size,
                'size_correct': expected_size == actual_size
            }
            
            # Check range
            results['range_check'][method_name] = {
                'min_value': int(np.min(pcm16)),
                'max_value': int(np.max(pcm16)),
                'within_int16': np.min(pcm16) >= -32768 and np.max(pcm16) <= 32767
            }
        
        return results
    
    def run_comprehensive_test(self) -> ValidationReport:
        """Run all validation tests"""
        start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print("Audio Format Validator - Comprehensive Test")
            print("=" * 60)
        
        # Generate test signals
        if self.verbose:
            print("\n1. Generating test signals...")
        self.generate_test_signals()
        
        # Test each method with each signal
        if self.verbose:
            print("\n2. Testing conversion methods...")
        
        all_results = []
        for method_name, method in self.conversion_methods.items():
            if self.verbose:
                print(f"   Testing {method_name}...")
            
            method_metrics = []
            for signal_name, test_signal in self.test_signals.items():
                metrics = self.analyze_conversion_method(
                    method, test_signal.signal, method_name
                )
                method_metrics.append(metrics)
            
            # Average metrics across all signals
            avg_metrics = self._average_metrics(method_metrics, method_name)
            all_results.append(avg_metrics)
        
        # Run specialized tests
        if self.verbose:
            print("\n3. Running DC bias analysis...")
        dc_bias_results = self.test_dc_bias()
        
        if self.verbose:
            print("\n4. Running symmetry analysis...")
        symmetry_results = self.test_symmetry()
        
        if self.verbose:
            print("\n5. Testing bit depth preservation...")
        bit_depth_results = self.test_bit_depth_preservation()
        
        if self.verbose:
            print("\n6. Checking OpenAI compatibility...")
        openai_results = self.test_openai_compatibility()
        
        # Determine optimal method
        optimal_method = self._determine_optimal_method(all_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_results, dc_bias_results, symmetry_results, 
            bit_depth_results, openai_results
        )
        
        # Create report
        duration = time.time() - start_time
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            test_duration=duration,
            conversion_methods=all_results,
            test_signals=list(self.test_signals.keys()),
            optimal_method=optimal_method,
            recommendations=recommendations,
            dc_bias_analysis=dc_bias_results,
            symmetry_analysis=symmetry_results,
            quality_analysis=bit_depth_results,
            openai_compatibility=openai_results,
            conclusion=self._generate_conclusion(optimal_method, all_results)
        )
        
        return report
    
    def _average_metrics(self, metrics_list: List[ConversionMetrics], 
                        method_name: str) -> ConversionMetrics:
        """Average metrics across multiple test signals"""
        if not metrics_list:
            return None
        
        # Calculate averages
        avg = ConversionMetrics(
            method_name=method_name,
            dc_bias=np.mean([m.dc_bias for m in metrics_list]),
            dc_bias_percent=np.mean([m.dc_bias_percent for m in metrics_list]),
            positive_peak=int(np.mean([m.positive_peak for m in metrics_list])),
            negative_peak=int(np.mean([m.negative_peak for m in metrics_list])),
            symmetry_ratio=np.mean([m.symmetry_ratio for m in metrics_list]),
            clipping_count=int(np.mean([m.clipping_count for m in metrics_list])),
            clipping_ratio=np.mean([m.clipping_ratio for m in metrics_list]),
            quantization_noise=np.mean([m.quantization_noise for m in metrics_list]),
            thd=np.mean([m.thd for m in metrics_list]),
            snr=np.mean([m.snr for m in metrics_list]),
            round_trip_error=np.mean([m.round_trip_error for m in metrics_list]),
            effective_bits=np.mean([m.effective_bits for m in metrics_list]),
            headroom_db=np.mean([m.headroom_db for m in metrics_list]),
            processing_time=np.mean([m.processing_time for m in metrics_list])
        )
        
        return avg
    
    def _determine_optimal_method(self, results: List[ConversionMetrics]) -> str:
        """Determine the optimal conversion method based on metrics"""
        scores = {}
        
        for metrics in results:
            score = 0
            
            # DC bias (lower is better, critical)
            if abs(metrics.dc_bias_percent) < 0.001:
                score += 25
            elif abs(metrics.dc_bias_percent) < 0.01:
                score += 15
            elif abs(metrics.dc_bias_percent) < 0.1:
                score += 5
            
            # Symmetry (closer to 1.0 is better)
            if 0.99 <= metrics.symmetry_ratio <= 1.01:
                score += 25
            elif 0.95 <= metrics.symmetry_ratio <= 1.05:
                score += 15
            elif 0.9 <= metrics.symmetry_ratio <= 1.1:
                score += 5
            
            # SNR (higher is better)
            if metrics.snr > 90:
                score += 20
            elif metrics.snr > 60:
                score += 15
            elif metrics.snr > 40:
                score += 10
            
            # Round-trip error (lower is better)
            if metrics.round_trip_error < 0.0001:
                score += 15
            elif metrics.round_trip_error < 0.001:
                score += 10
            elif metrics.round_trip_error < 0.01:
                score += 5
            
            # Effective bits (higher is better)
            if metrics.effective_bits > 15:
                score += 10
            elif metrics.effective_bits > 14:
                score += 7
            elif metrics.effective_bits > 12:
                score += 3
            
            # No clipping bonus
            if metrics.clipping_ratio == 0:
                score += 5
            
            scores[metrics.method_name] = score
        
        # Find method with highest score
        optimal = max(scores, key=scores.get)
        
        if self.verbose:
            print("\n7. Method Scores:")
            for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"   {method}: {score}/100")
        
        return optimal
    
    def _generate_recommendations(self, metrics: List[ConversionMetrics],
                                 dc_bias: Dict, symmetry: Dict,
                                 bit_depth: Dict, openai: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Find best method
        best_dc_bias = min(metrics, key=lambda x: abs(x.dc_bias_percent))
        best_symmetry = min(metrics, key=lambda x: abs(x.symmetry_ratio - 1.0))
        
        # DC bias recommendation
        if abs(best_dc_bias.dc_bias_percent) < 0.001:
            recommendations.append(
                f"✓ DC bias is excellent with {best_dc_bias.method_name} "
                f"({best_dc_bias.dc_bias_percent:.4f}%). No changes needed."
            )
        else:
            recommendations.append(
                f"⚠ Consider using {best_dc_bias.method_name} to minimize DC bias "
                f"(current: {best_dc_bias.dc_bias_percent:.4f}%)"
            )
        
        # Symmetry recommendation
        if 0.99 <= best_symmetry.symmetry_ratio <= 1.01:
            recommendations.append(
                f"✓ Clipping symmetry is excellent with {best_symmetry.method_name} "
                f"(ratio: {best_symmetry.symmetry_ratio:.3f})"
            )
        else:
            recommendations.append(
                f"⚠ Improve clipping symmetry by using {best_symmetry.method_name} "
                f"(ratio: {best_symmetry.symmetry_ratio:.3f})"
            )
        
        # OpenAI compatibility
        all_compatible = all(
            openai['format_check'][m]['is_int16'] and
            openai['endianness_check'][m]['is_little_endian'] and
            openai['size_check'][m]['size_correct'] and
            openai['range_check'][m]['within_int16']
            for m in openai['format_check']
        )
        
        if all_compatible:
            recommendations.append("✓ All methods are OpenAI Realtime API compatible")
        else:
            recommendations.append("⚠ Some methods have OpenAI compatibility issues")
        
        # Current implementation assessment
        current_method_metrics = next((m for m in metrics if m.method_name == 'current_32767'), None)
        if current_method_metrics:
            if abs(current_method_metrics.dc_bias_percent) < 0.001 and \
               0.99 <= current_method_metrics.symmetry_ratio <= 1.01:
                recommendations.append(
                    "✓ Current implementation (32767 scaling) is optimal. No changes required."
                )
            else:
                recommendations.append(
                    "⚠ Current implementation could be improved for better audio quality"
                )
        
        return recommendations
    
    def _generate_conclusion(self, optimal_method: str, 
                           metrics: List[ConversionMetrics]) -> str:
        """Generate overall conclusion"""
        optimal_metrics = next((m for m in metrics if m.method_name == optimal_method), None)
        
        if optimal_method == 'current_32767':
            conclusion = (
                f"The current implementation using 32767 scaling is OPTIMAL. "
                f"It provides excellent DC bias performance ({optimal_metrics.dc_bias_percent:.4f}%), "
                f"perfect symmetry (ratio: {optimal_metrics.symmetry_ratio:.3f}), "
                f"and high signal quality (SNR: {optimal_metrics.snr:.1f}dB). "
                f"No changes to the PCM16 conversion are recommended."
            )
        else:
            conclusion = (
                f"Consider switching to {optimal_method} for improved audio quality. "
                f"This method offers better DC bias ({optimal_metrics.dc_bias_percent:.4f}%), "
                f"symmetry (ratio: {optimal_metrics.symmetry_ratio:.3f}), "
                f"and overall signal quality (SNR: {optimal_metrics.snr:.1f}dB)."
            )
        
        return conclusion
    
    def save_report(self, report: ValidationReport, output_path: str = None):
        """Save validation report to JSON file"""
        if output_path is None:
            output_path = "reports/pcm16_conversion_analysis.json"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dict
        report_dict = asdict(report)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\nReport saved to: {output_path}")
    
    def plot_results(self, report: ValidationReport):
        """Generate visualization plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available - skipping visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PCM16 Conversion Analysis Results', fontsize=16)
        
        # Extract data for plotting
        methods = [m.method_name for m in report.conversion_methods]
        dc_bias = [abs(m.dc_bias_percent) for m in report.conversion_methods]
        symmetry = [abs(m.symmetry_ratio - 1.0) for m in report.conversion_methods]
        snr = [m.snr for m in report.conversion_methods]
        thd = [m.thd for m in report.conversion_methods]
        error = [m.round_trip_error for m in report.conversion_methods]
        bits = [m.effective_bits for m in report.conversion_methods]
        
        # Plot 1: DC Bias
        axes[0, 0].bar(methods, dc_bias)
        axes[0, 0].set_title('DC Bias (%)')
        axes[0, 0].set_ylabel('Absolute DC Bias (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Symmetry Deviation
        axes[0, 1].bar(methods, symmetry)
        axes[0, 1].set_title('Symmetry Deviation from 1.0')
        axes[0, 1].set_ylabel('Absolute Deviation')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: SNR
        axes[0, 2].bar(methods, snr)
        axes[0, 2].set_title('Signal-to-Noise Ratio (dB)')
        axes[0, 2].set_ylabel('SNR (dB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: THD
        axes[1, 0].bar(methods, thd)
        axes[1, 0].set_title('Total Harmonic Distortion (%)')
        axes[1, 0].set_ylabel('THD (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Round-trip Error
        axes[1, 1].bar(methods, error)
        axes[1, 1].set_title('Round-trip Error')
        axes[1, 1].set_ylabel('RMS Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Effective Bits
        axes[1, 2].bar(methods, bits)
        axes[1, 2].set_title('Effective Bit Depth')
        axes[1, 2].set_ylabel('Bits')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].axhline(y=16, color='r', linestyle='--', label='16-bit target')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = "reports/pcm16_conversion_analysis.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"Plots saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Audio Format Validator')
    parser.add_argument('--full-test', action='store_true',
                       help='Run comprehensive validation tests')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick validation test')
    parser.add_argument('--openai-test', action='store_true',
                       help='Test OpenAI compatibility only')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Compare all conversion methods')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    parser.add_argument('--output', type=str,
                       default='reports/pcm16_conversion_analysis.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Create validator
    validator = AudioFormatValidator(verbose=args.verbose)
    
    if args.full_test or not any([args.quick_test, args.openai_test, args.compare_methods]):
        # Run full test by default
        print("Running comprehensive audio format validation...")
        report = validator.run_comprehensive_test()
        validator.save_report(report, args.output)
        validator.plot_results(report)
        
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"\nOptimal Method: {report.optimal_method}")
        print(f"\nConclusion: {report.conclusion}")
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    elif args.quick_test:
        print("Running quick validation test...")
        validator.generate_test_signals(duration=0.1)
        results = []
        for method_name, method in validator.conversion_methods.items():
            signal = validator.test_signals['sine_1000hz'].signal
            metrics = validator.analyze_conversion_method(method, signal, method_name)
            results.append(metrics)
            print(f"\n{method_name}:")
            print(f"  DC Bias: {metrics.dc_bias_percent:.4f}%")
            print(f"  Symmetry: {metrics.symmetry_ratio:.3f}")
            print(f"  SNR: {metrics.snr:.1f} dB")
    
    elif args.openai_test:
        print("Testing OpenAI compatibility...")
        validator.generate_test_signals(duration=0.1)
        results = validator.test_openai_compatibility()
        
        for method in results['format_check']:
            print(f"\n{method}:")
            print(f"  Format OK: {results['format_check'][method]['is_int16']}")
            print(f"  Endianness OK: {results['endianness_check'][method]['is_little_endian']}")
            print(f"  Size OK: {results['size_check'][method]['size_correct']}")
            print(f"  Range OK: {results['range_check'][method]['within_int16']}")
    
    elif args.compare_methods:
        print("Comparing conversion methods...")
        validator.generate_test_signals(duration=0.5)
        report = validator.run_comprehensive_test()
        
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        
        for metrics in report.conversion_methods:
            print(f"\n{metrics.method_name}:")
            print(f"  DC Bias: {metrics.dc_bias_percent:.6f}%")
            print(f"  Symmetry Ratio: {metrics.symmetry_ratio:.4f}")
            print(f"  SNR: {metrics.snr:.1f} dB")
            print(f"  THD: {metrics.thd:.4f}%")
            print(f"  Round-trip Error: {metrics.round_trip_error:.6f}")
            print(f"  Effective Bits: {metrics.effective_bits:.2f}")


if __name__ == "__main__":
    main()