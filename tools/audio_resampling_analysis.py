#!/usr/bin/env python3
"""
Audio Resampling Quality Analysis Tool

This tool compares different resampling methods to identify the optimal solution
for voice audio quality in the HA Realtime Voice Assistant pipeline.

Resampling Methods Tested:
- scipy.signal.resample (FFT-based)
- scipy.signal.resample_poly (polyphase filtering)
- librosa.resample (Kaiser window)
- soxr (SoX resampler)
- resampy.resample (band-limited sinc)

Usage:
    python tools/audio_resampling_analysis.py --run-all
    python tools/audio_resampling_analysis.py --method scipy_fft --test quality
    python tools/audio_resampling_analysis.py --benchmark-performance
    python tools/audio_resampling_analysis.py --generate-report
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
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import existing analysis infrastructure
from audio_analysis.metrics import AudioMetrics
from audio_analysis.visualization import AudioVisualizer

# Core resampling libraries
try:
    from scipy import signal
    from scipy.fft import rfft, rfftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - core functionality limited")

# Optional resampling libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available - skipping librosa resampling tests")

try:
    import soxr
    SOXR_AVAILABLE = True
except ImportError:
    SOXR_AVAILABLE = False
    warnings.warn("soxr not available - skipping soxr resampling tests")

try:
    import resampy
    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False
    warnings.warn("resampy not available - skipping resampy resampling tests")


@dataclass
class ResamplingMethod:
    """Configuration for a resampling method"""
    name: str
    display_name: str
    library: str
    available: bool
    parameters: Dict[str, Any]
    description: str


@dataclass
class QualityMetrics:
    """Quality metrics for resampled audio"""
    method: str
    thd: float  # Total Harmonic Distortion (%)
    snr: float  # Signal-to-Noise Ratio (dB)
    frequency_response: Dict[str, float]  # Frequency band responses
    aliasing_rejection: float  # dB
    phase_response_variance: float  # Radians
    passband_ripple: float  # dB
    stopband_attenuation: float  # dB
    dc_offset: float  # Normalized
    correlation: float  # With original signal


@dataclass
class PerformanceMetrics:
    """Performance metrics for resampling method"""
    method: str
    chunk_size: int
    processing_time_ms: float  # Average milliseconds per chunk
    cpu_usage_percent: float  # CPU usage during processing
    memory_mb: float  # Peak memory usage
    latency_ms: float  # End-to-end latency
    throughput_samples_per_sec: float  # Samples processed per second


@dataclass
class TestConfiguration:
    """Test configuration parameters"""
    chunk_sizes: List[int]  # Samples
    sample_rate_pairs: List[Tuple[int, int]]  # (orig_sr, target_sr)
    test_signals: List[str]  # Signal types
    amplitude_levels: List[float]  # Normalized amplitudes
    iterations: int  # Performance test iterations


class ResamplingAnalyzer:
    """Main analyzer for comparing audio resampling methods"""
    
    def __init__(self):
        """Initialize the resampling analyzer"""
        self.metrics_calculator = AudioMetrics()
        self.visualizer = AudioVisualizer() if hasattr(self, '_check_viz_available') else None
        
        # Define available resampling methods
        self.methods = self._initialize_methods()
        
        # Test configuration
        self.config = TestConfiguration(
            chunk_sizes=[1200, 2400, 4800],  # 50ms, 100ms, 200ms at 24kHz
            sample_rate_pairs=[
                (48000, 24000),  # Main pipeline resampling
                (48000, 16000),  # Wake word resampling
                (44100, 24000),  # Alternative input rate
            ],
            test_signals=[
                'tone_440',
                'tone_1000',
                'tone_3000',
                'tone_8000',
                'speech_male',
                'speech_female',
                'white_noise',
                'pink_noise',
                'multi_tone',
                'frequency_sweep'
            ],
            amplitude_levels=[0.1, 0.5, 0.9],
            iterations=100
        )
        
        # Results storage
        self.quality_results: List[QualityMetrics] = []
        self.performance_results: List[PerformanceMetrics] = []
    
    def _initialize_methods(self) -> Dict[str, ResamplingMethod]:
        """Initialize resampling method configurations"""
        methods = {}
        
        # scipy FFT-based resampling (current method)
        methods['scipy_fft'] = ResamplingMethod(
            name='scipy_fft',
            display_name='SciPy FFT (Current)',
            library='scipy.signal.resample',
            available=SCIPY_AVAILABLE,
            parameters={},
            description='FFT-based resampling without explicit anti-aliasing'
        )
        
        # scipy polyphase resampling
        methods['scipy_poly'] = ResamplingMethod(
            name='scipy_poly',
            display_name='SciPy Polyphase',
            library='scipy.signal.resample_poly',
            available=SCIPY_AVAILABLE,
            parameters={'window': 'hamming'},
            description='Polyphase filtering with configurable window'
        )
        
        # librosa Kaiser window resampling
        methods['librosa'] = ResamplingMethod(
            name='librosa',
            display_name='Librosa Kaiser',
            library='librosa.resample',
            available=LIBROSA_AVAILABLE,
            parameters={'res_type': 'kaiser_best'},
            description='Kaiser window method optimized for quality'
        )
        
        # soxr high-quality resampling
        methods['soxr'] = ResamplingMethod(
            name='soxr',
            display_name='SoX Resampler',
            library='soxr',
            available=SOXR_AVAILABLE,
            parameters={'quality': 'HQ'},
            description='High-quality SoX resampler used in audio production'
        )
        
        # resampy band-limited sinc
        methods['resampy'] = ResamplingMethod(
            name='resampy',
            display_name='Resampy Sinc',
            library='resampy',
            available=RESAMPY_AVAILABLE,
            parameters={'filter': 'kaiser_best'},
            description='Band-limited sinc interpolation'
        )
        
        return methods
    
    def resample_scipy_fft(self, audio_data: np.ndarray, 
                          orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using scipy FFT method (current implementation)"""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not available")
        
        new_length = int(len(audio_data) * target_sr / orig_sr)
        return signal.resample(audio_data, new_length)
    
    def resample_scipy_poly(self, audio_data: np.ndarray,
                           orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using scipy polyphase method"""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not available")
        
        # Calculate up and down factors
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        
        # Use polyphase resampling with Hamming window
        return signal.resample_poly(audio_data, up, down, window='hamming')
    
    def resample_librosa(self, audio_data: np.ndarray,
                        orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using librosa Kaiser window method"""
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa not available")
        
        return librosa.resample(
            y=audio_data,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type='kaiser_best'
        )
    
    def resample_soxr(self, audio_data: np.ndarray,
                     orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using soxr high-quality resampler"""
        if not SOXR_AVAILABLE:
            raise RuntimeError("soxr not available")
        
        # soxr expects audio in shape (samples,) or (samples, channels)
        resampled = soxr.resample(
            audio_data,
            orig_sr,
            target_sr,
            quality='HQ'  # High quality mode
        )
        return resampled
    
    def resample_resampy(self, audio_data: np.ndarray,
                        orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using resampy band-limited sinc"""
        if not RESAMPY_AVAILABLE:
            raise RuntimeError("resampy not available")
        
        return resampy.resample(
            audio_data,
            orig_sr,
            target_sr,
            filter='kaiser_best'
        )
    
    def generate_test_signal(self, signal_type: str, duration: float,
                           sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
        """Generate test signals for analysis"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        if signal_type == 'tone_440':
            signal = amplitude * np.sin(2 * np.pi * 440 * t)
        elif signal_type == 'tone_1000':
            signal = amplitude * np.sin(2 * np.pi * 1000 * t)
        elif signal_type == 'tone_3000':
            signal = amplitude * np.sin(2 * np.pi * 3000 * t)
        elif signal_type == 'tone_8000':
            signal = amplitude * np.sin(2 * np.pi * 8000 * t)
        elif signal_type == 'white_noise':
            signal = amplitude * np.random.randn(samples)
        elif signal_type == 'pink_noise':
            # Simple pink noise approximation
            white = np.random.randn(samples)
            # Apply 1/f filter
            fft = rfft(white)
            freqs = rfftfreq(samples, 1/sample_rate)
            freqs[0] = 1  # Avoid division by zero
            fft = fft / np.sqrt(freqs)
            signal = amplitude * np.real(np.fft.irfft(fft, samples))
        elif signal_type == 'multi_tone':
            # Multiple harmonically related tones
            signal = amplitude * (
                0.5 * np.sin(2 * np.pi * 440 * t) +
                0.3 * np.sin(2 * np.pi * 880 * t) +
                0.2 * np.sin(2 * np.pi * 1320 * t)
            )
        elif signal_type == 'frequency_sweep':
            # Logarithmic sweep from 20Hz to 12kHz
            f0, f1 = 20, 12000
            sweep = amplitude * signal.chirp(t, f0, duration, f1, method='logarithmic')
            signal = sweep
        elif signal_type.startswith('speech_'):
            # Synthesize simple speech-like signal
            # Mix of formants typical for vowels
            f1, f2, f3 = 700, 1220, 2600  # Formants for 'a' sound
            signal = amplitude * (
                0.6 * np.sin(2 * np.pi * f1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t) +
                0.1 * np.sin(2 * np.pi * f3 * t)
            )
            # Add some amplitude modulation for naturalness
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
            signal *= envelope
        else:
            # Default to silence
            signal = np.zeros(samples)
        
        return signal.astype(np.float32)
    
    def benchmark_quality(self, method: str, signal: np.ndarray,
                         orig_sr: int, target_sr: int) -> QualityMetrics:
        """Measure quality metrics for a resampling method"""
        # Get the resampling function
        resample_func = getattr(self, f'resample_{method}')
        
        # Perform resampling
        resampled = resample_func(signal, orig_sr, target_sr)
        
        # Calculate quality metrics
        thd = self.metrics_calculator.calculate_thd(resampled, target_sr)
        snr = self.metrics_calculator.calculate_snr(resampled, signal[:len(resampled)])
        
        # Frequency response analysis
        freq_response = self.analyze_frequency_response(
            signal, resampled, orig_sr, target_sr
        )
        
        # Aliasing rejection (measure energy above Nyquist)
        aliasing = self.measure_aliasing(resampled, target_sr)
        
        # Phase response
        phase_var = self.measure_phase_response(signal, resampled, orig_sr, target_sr)
        
        # Passband and stopband characteristics
        passband_ripple, stopband_atten = self.measure_filter_response(
            method, orig_sr, target_sr
        )
        
        # DC offset
        dc_offset = np.mean(resampled)
        
        # Correlation with original (downsampled for comparison)
        correlation = self.calculate_correlation(signal, resampled, orig_sr, target_sr)
        
        return QualityMetrics(
            method=method,
            thd=thd,
            snr=snr,
            frequency_response=freq_response,
            aliasing_rejection=aliasing,
            phase_response_variance=phase_var,
            passband_ripple=passband_ripple,
            stopband_attenuation=stopband_atten,
            dc_offset=dc_offset,
            correlation=correlation
        )
    
    def benchmark_performance(self, method: str, chunk_size: int,
                            iterations: int = 100) -> PerformanceMetrics:
        """Measure performance metrics for a resampling method"""
        # Generate test chunk
        orig_sr = 48000
        target_sr = 24000
        test_signal = self.generate_test_signal('white_noise', 
                                               chunk_size / orig_sr, 
                                               orig_sr)
        
        # Get resampling function
        resample_func = getattr(self, f'resample_{method}')
        
        # Warm up
        for _ in range(10):
            _ = resample_func(test_signal, orig_sr, target_sr)
        
        # Measure processing time
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = resample_func(test_signal, orig_sr, target_sr)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time_ms = np.mean(times)
        
        # Calculate throughput
        samples_per_sec = chunk_size / (avg_time_ms / 1000)
        
        # Estimate CPU usage (simplified)
        cpu_usage = (avg_time_ms / (chunk_size / orig_sr * 1000)) * 100
        
        # Memory usage (simplified estimate)
        import sys
        memory_mb = sys.getsizeof(test_signal) / (1024 * 1024)
        
        return PerformanceMetrics(
            method=method,
            chunk_size=chunk_size,
            processing_time_ms=avg_time_ms,
            cpu_usage_percent=min(cpu_usage, 100),
            memory_mb=memory_mb,
            latency_ms=avg_time_ms,
            throughput_samples_per_sec=samples_per_sec
        )
    
    def analyze_frequency_response(self, original: np.ndarray, resampled: np.ndarray,
                                  orig_sr: int, target_sr: int) -> Dict[str, float]:
        """Analyze frequency response of resampling"""
        # Define frequency bands for voice analysis
        bands = {
            '20-80Hz': (20, 80),      # Sub-bass
            '80-250Hz': (80, 250),    # Bass
            '250-500Hz': (250, 500),  # Low-mid
            '500-1kHz': (500, 1000),  # Mid
            '1-2kHz': (1000, 2000),   # Upper-mid
            '2-4kHz': (2000, 4000),   # Presence
            '4-8kHz': (4000, 8000),   # Brilliance
            '8-12kHz': (8000, 12000)  # Air
        }
        
        # Calculate FFT of resampled signal
        fft = rfft(resampled)
        freqs = rfftfreq(len(resampled), 1/target_sr)
        magnitude = np.abs(fft)
        
        # Calculate response for each band
        response = {}
        for band_name, (low, high) in bands.items():
            # Find frequency indices
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_energy = np.mean(magnitude[mask])
                response[band_name] = float(20 * np.log10(band_energy + 1e-10))
            else:
                response[band_name] = -60.0  # Below noise floor
        
        return response
    
    def measure_aliasing(self, resampled: np.ndarray, target_sr: int) -> float:
        """Measure aliasing artifacts above Nyquist frequency"""
        # Calculate FFT
        fft = rfft(resampled)
        freqs = rfftfreq(len(resampled), 1/target_sr)
        magnitude = np.abs(fft)
        
        # Nyquist frequency
        nyquist = target_sr / 2
        
        # Measure energy above 0.9 * Nyquist (allowing for transition band)
        high_freq_mask = freqs > (0.9 * nyquist)
        
        if np.any(high_freq_mask):
            alias_energy = np.mean(magnitude[high_freq_mask])
            total_energy = np.mean(magnitude)
            
            if total_energy > 0:
                # Return aliasing rejection in dB
                return float(-20 * np.log10(alias_energy / total_energy + 1e-10))
            else:
                return 60.0  # Good rejection
        else:
            return 60.0  # No high frequencies to measure
    
    def measure_phase_response(self, original: np.ndarray, resampled: np.ndarray,
                              orig_sr: int, target_sr: int) -> float:
        """Measure phase response consistency"""
        # Calculate phase of FFTs
        fft_resampled = rfft(resampled)
        phase = np.angle(fft_resampled)
        
        # Calculate phase variance (simplified)
        # Unwrap phase to avoid discontinuities
        unwrapped = np.unwrap(phase)
        
        # Calculate variance of phase derivative (group delay variation)
        phase_diff = np.diff(unwrapped)
        variance = np.var(phase_diff)
        
        return float(variance)
    
    def measure_filter_response(self, method: str, orig_sr: int, 
                               target_sr: int) -> Tuple[float, float]:
        """Measure passband ripple and stopband attenuation"""
        # Generate impulse response
        impulse = np.zeros(1000)
        impulse[500] = 1.0
        
        # Get resampling function
        resample_func = getattr(self, f'resample_{method}')
        
        try:
            # Resample impulse to get filter response
            response = resample_func(impulse, orig_sr, target_sr)
            
            # Calculate frequency response
            fft = rfft(response)
            magnitude = np.abs(fft)
            freqs = rfftfreq(len(response), 1/target_sr)
            
            # Passband (0 to 0.4 * Nyquist)
            passband_mask = freqs < (0.4 * target_sr / 2)
            if np.any(passband_mask):
                passband_mag = magnitude[passband_mask]
                passband_ripple = float(np.max(passband_mag) - np.min(passband_mag))
            else:
                passband_ripple = 0.0
            
            # Stopband (0.6 * Nyquist to Nyquist)
            stopband_mask = freqs > (0.6 * target_sr / 2)
            if np.any(stopband_mask):
                stopband_mag = magnitude[stopband_mask]
                stopband_atten = float(-20 * np.log10(np.max(stopband_mag) + 1e-10))
            else:
                stopband_atten = 60.0
            
            return passband_ripple, stopband_atten
            
        except Exception:
            # Return default values if measurement fails
            return 0.0, 40.0
    
    def calculate_correlation(self, original: np.ndarray, resampled: np.ndarray,
                            orig_sr: int, target_sr: int) -> float:
        """Calculate correlation between original and resampled signals"""
        # Downsample original to match resampled length for comparison
        if len(original) > len(resampled):
            # Simple decimation for comparison
            factor = len(original) // len(resampled)
            downsampled_orig = original[::factor][:len(resampled)]
        else:
            downsampled_orig = original
        
        # Ensure same length
        min_len = min(len(downsampled_orig), len(resampled))
        downsampled_orig = downsampled_orig[:min_len]
        resampled = resampled[:min_len]
        
        # Calculate correlation coefficient
        if len(downsampled_orig) > 0:
            correlation = np.corrcoef(downsampled_orig, resampled)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def run_comprehensive_test(self):
        """Run comprehensive testing of all resampling methods"""
        print("\n" + "="*60)
        print("Audio Resampling Quality Analysis")
        print("="*60)
        
        # Test each available method
        for method_name, method_info in self.methods.items():
            if not method_info.available:
                print(f"\nSkipping {method_info.display_name} - library not available")
                continue
            
            print(f"\nTesting {method_info.display_name}...")
            print(f"  Library: {method_info.library}")
            print(f"  Description: {method_info.description}")
            
            # Quality tests
            print("  Running quality tests...")
            for signal_type in ['tone_1000', 'multi_tone', 'frequency_sweep']:
                for orig_sr, target_sr in [(48000, 24000), (48000, 16000)]:
                    test_signal = self.generate_test_signal(
                        signal_type, 0.5, orig_sr, amplitude=0.5
                    )
                    
                    try:
                        metrics = self.benchmark_quality(
                            method_name, test_signal, orig_sr, target_sr
                        )
                        self.quality_results.append(metrics)
                        
                        # Print summary
                        print(f"    {signal_type} {orig_sr}→{target_sr}Hz: "
                             f"THD={metrics.thd:.2f}%, SNR={metrics.snr:.1f}dB, "
                             f"Aliasing={metrics.aliasing_rejection:.1f}dB")
                    except Exception as e:
                        print(f"    Error testing {signal_type}: {e}")
            
            # Performance tests
            print("  Running performance tests...")
            for chunk_size in [1200, 2400, 4800]:
                try:
                    perf = self.benchmark_performance(
                        method_name, chunk_size, iterations=50
                    )
                    self.performance_results.append(perf)
                    
                    print(f"    Chunk {chunk_size}: "
                         f"Time={perf.processing_time_ms:.2f}ms, "
                         f"CPU={perf.cpu_usage_percent:.1f}%, "
                         f"Throughput={perf.throughput_samples_per_sec/1e6:.2f}M samples/s")
                except Exception as e:
                    print(f"    Error benchmarking chunk {chunk_size}: {e}")
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        if not self.quality_results or not self.performance_results:
            print("No results to report. Run tests first.")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        quality_df = pd.DataFrame([asdict(r) for r in self.quality_results])
        perf_df = pd.DataFrame([asdict(r) for r in self.performance_results])
        
        # Create summary
        summary = {}
        for method_name in self.methods.keys():
            if not self.methods[method_name].available:
                continue
            
            method_quality = quality_df[quality_df['method'] == method_name]
            method_perf = perf_df[perf_df['method'] == method_name]
            
            if not method_quality.empty and not method_perf.empty:
                summary[method_name] = {
                    'avg_thd': method_quality['thd'].mean(),
                    'avg_snr': method_quality['snr'].mean(),
                    'avg_aliasing_rejection': method_quality['aliasing_rejection'].mean(),
                    'avg_processing_time_ms': method_perf['processing_time_ms'].mean(),
                    'avg_cpu_usage': method_perf['cpu_usage_percent'].mean(),
                    'quality_score': (
                        (100 - method_quality['thd'].mean()) * 0.3 +
                        method_quality['snr'].mean() * 0.3 +
                        method_quality['aliasing_rejection'].mean() * 0.4
                    ),
                    'performance_score': (
                        (100 - method_perf['cpu_usage_percent'].mean()) * 0.5 +
                        (1000 / method_perf['processing_time_ms'].mean()) * 0.5
                    )
                }
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df['overall_score'] = (
            summary_df['quality_score'] * 0.7 + 
            summary_df['performance_score'] * 0.3
        )
        
        return summary_df.sort_values('overall_score', ascending=False)
    
    def save_results(self, output_dir: Path):
        """Save all results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        quality_data = [asdict(r) for r in self.quality_results]
        perf_data = [asdict(r) for r in self.performance_results]
        
        with open(output_dir / 'quality_results.json', 'w') as f:
            json.dump(quality_data, f, indent=2)
        
        with open(output_dir / 'performance_results.json', 'w') as f:
            json.dump(perf_data, f, indent=2)
        
        # Generate and save comparison report
        summary_df = self.generate_comparison_report()
        summary_df.to_csv(output_dir / 'comparison_summary.csv')
        
        # Generate recommendation
        if not summary_df.empty:
            best_method = summary_df.index[0]
            recommendation = {
                'recommended_method': best_method,
                'method_info': asdict(self.methods[best_method]),
                'quality_metrics': {
                    'thd': float(summary_df.loc[best_method, 'avg_thd']),
                    'snr': float(summary_df.loc[best_method, 'avg_snr']),
                    'aliasing_rejection': float(summary_df.loc[best_method, 'avg_aliasing_rejection'])
                },
                'performance_metrics': {
                    'processing_time_ms': float(summary_df.loc[best_method, 'avg_processing_time_ms']),
                    'cpu_usage_percent': float(summary_df.loc[best_method, 'avg_cpu_usage'])
                },
                'overall_score': float(summary_df.loc[best_method, 'overall_score'])
            }
            
            with open(output_dir / 'recommendation.json', 'w') as f:
                json.dump(recommendation, f, indent=2)
            
            print(f"\n{'='*60}")
            print("RECOMMENDATION")
            print(f"{'='*60}")
            print(f"Best Method: {self.methods[best_method].display_name}")
            print(f"Overall Score: {recommendation['overall_score']:.2f}")
            print(f"THD: {recommendation['quality_metrics']['thd']:.2f}%")
            print(f"SNR: {recommendation['quality_metrics']['snr']:.1f}dB")
            print(f"Aliasing Rejection: {recommendation['quality_metrics']['aliasing_rejection']:.1f}dB")
            print(f"Processing Time: {recommendation['performance_metrics']['processing_time_ms']:.2f}ms")
            print(f"CPU Usage: {recommendation['performance_metrics']['cpu_usage_percent']:.1f}%")
        
        print(f"\nResults saved to {output_dir}")
    
    def _check_viz_available(self) -> bool:
        """Check if visualization is available"""
        try:
            import matplotlib
            return True
        except ImportError:
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Audio Resampling Quality Analysis Tool'
    )
    
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all tests and generate report'
    )
    
    parser.add_argument(
        '--method',
        choices=['scipy_fft', 'scipy_poly', 'librosa', 'soxr', 'resampy'],
        help='Test specific resampling method'
    )
    
    parser.add_argument(
        '--test',
        choices=['quality', 'performance', 'both'],
        default='both',
        help='Type of test to run'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/resampling'),
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate report from existing results'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResamplingAnalyzer()
    
    if args.run_all:
        # Run comprehensive test
        analyzer.run_comprehensive_test()
        analyzer.save_results(args.output_dir)
    elif args.method:
        # Test specific method
        if not analyzer.methods[args.method].available:
            print(f"Method {args.method} not available - library not installed")
            sys.exit(1)
        
        if args.test in ['quality', 'both']:
            print(f"Testing quality for {args.method}...")
            test_signal = analyzer.generate_test_signal('multi_tone', 0.5, 48000)
            metrics = analyzer.benchmark_quality(args.method, test_signal, 48000, 24000)
            print(f"Quality metrics: {metrics}")
        
        if args.test in ['performance', 'both']:
            print(f"Testing performance for {args.method}...")
            perf = analyzer.benchmark_performance(args.method, 2400)
            print(f"Performance metrics: {perf}")
    elif args.generate_report:
        # Load and generate report from existing results
        if (args.output_dir / 'quality_results.json').exists():
            with open(args.output_dir / 'quality_results.json') as f:
                quality_data = json.load(f)
                analyzer.quality_results = [QualityMetrics(**d) for d in quality_data]
        
        if (args.output_dir / 'performance_results.json').exists():
            with open(args.output_dir / 'performance_results.json') as f:
                perf_data = json.load(f)
                analyzer.performance_results = [PerformanceMetrics(**d) for d in perf_data]
        
        summary = analyzer.generate_comparison_report()
        print("\nComparison Summary:")
        print(summary)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()