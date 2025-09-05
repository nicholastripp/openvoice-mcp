"""
Performance benchmarking module for resampling methods

Provides comprehensive performance testing including:
- Processing time and latency measurements
- CPU and memory usage profiling
- Raspberry Pi specific optimizations
- Real-time capability assessment
- Batch vs streaming performance
"""

import time
import numpy as np
import psutil
import os
import gc
import platform
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, asdict
import warnings
import json
from pathlib import Path


@dataclass
class PerformanceMeasurement:
    """Single performance measurement"""
    method: str
    chunk_size: int
    sample_rate_from: int
    sample_rate_to: int
    processing_time_ms: float
    cpu_percent: float
    memory_mb: float
    cache_misses: int
    context_switches: int


@dataclass
class StreamingPerformance:
    """Streaming/real-time performance metrics"""
    method: str
    chunks_processed: int
    total_time_ms: float
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    latency_std_ms: float
    dropped_chunks: int
    xruns: int  # Buffer underruns/overruns


@dataclass
class PlatformInfo:
    """System platform information"""
    cpu_model: str
    cpu_cores: int
    cpu_freq_mhz: float
    ram_total_mb: float
    platform: str
    architecture: str
    is_raspberry_pi: bool
    pi_model: str


class PerformanceBenchmark:
    """Performance benchmarking for audio resampling methods"""
    
    def __init__(self):
        """Initialize performance benchmark"""
        self.platform_info = self._detect_platform()
        self.process = psutil.Process(os.getpid())
        
        # Performance thresholds for real-time capability
        self.realtime_thresholds = {
            'latency_ms': 10.0,  # Max acceptable latency
            'cpu_percent': 50.0,  # Max CPU usage for one core
            'jitter_ms': 2.0      # Max latency variation
        }
        
        # Raspberry Pi specific settings
        if self.platform_info.is_raspberry_pi:
            self._optimize_for_pi()
    
    def _detect_platform(self) -> PlatformInfo:
        """Detect system platform and capabilities"""
        cpu_info = {
            'cpu_model': platform.processor() or 'Unknown',
            'cpu_cores': psutil.cpu_count(logical=False) or 1,
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'ram_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'platform': platform.system(),
            'architecture': platform.machine(),
            'is_raspberry_pi': False,
            'pi_model': ''
        }
        
        # Detect Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'Raspberry Pi' in model:
                    cpu_info['is_raspberry_pi'] = True
                    cpu_info['pi_model'] = model
        except FileNotFoundError:
            pass
        
        # Alternative Pi detection
        if not cpu_info['is_raspberry_pi']:
            if 'arm' in cpu_info['architecture'].lower() or 'aarch64' in cpu_info['architecture'].lower():
                if os.path.exists('/boot/config.txt') or os.path.exists('/boot/firmware/config.txt'):
                    cpu_info['is_raspberry_pi'] = True
                    cpu_info['pi_model'] = 'Raspberry Pi (Generic ARM)'
        
        return PlatformInfo(**cpu_info)
    
    def _optimize_for_pi(self):
        """Apply Raspberry Pi specific optimizations"""
        # Set process priority
        try:
            os.nice(-5)  # Increase priority slightly
        except PermissionError:
            pass
        
        # Set CPU affinity to avoid core switching
        try:
            # Use cores 2-3 on Pi 4, avoiding core 0 (system tasks)
            if self.platform_info.cpu_cores >= 4:
                self.process.cpu_affinity([2, 3])
            elif self.platform_info.cpu_cores >= 2:
                self.process.cpu_affinity([1])
        except (AttributeError, OSError):
            pass
        
        print(f"Optimized for {self.platform_info.pi_model}")
    
    def measure_processing_time(self, resample_func: Callable,
                              audio_data: np.ndarray,
                              orig_sr: int, target_sr: int,
                              iterations: int = 100,
                              warmup: int = 10) -> Dict[str, float]:
        """
        Measure processing time with statistical analysis
        
        Args:
            resample_func: Resampling function to benchmark
            audio_data: Test audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            iterations: Number of test iterations
            warmup: Number of warmup iterations
            
        Returns:
            Timing statistics
        """
        # Warmup runs
        for _ in range(warmup):
            _ = resample_func(audio_data, orig_sr, target_sr)
            gc.collect()
        
        # Measurement runs
        times = []
        for _ in range(iterations):
            gc.collect()  # Consistent GC state
            gc.disable()  # Prevent GC during measurement
            
            start = time.perf_counter_ns()
            _ = resample_func(audio_data, orig_sr, target_sr)
            end = time.perf_counter_ns()
            
            gc.enable()
            times.append((end - start) / 1e6)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99))
        }
    
    def measure_cpu_memory(self, resample_func: Callable,
                          audio_data: np.ndarray,
                          orig_sr: int, target_sr: int,
                          duration_sec: float = 5.0) -> Dict[str, float]:
        """
        Measure CPU and memory usage during resampling
        
        Args:
            resample_func: Resampling function to benchmark
            audio_data: Test audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            duration_sec: Duration of measurement
            
        Returns:
            CPU and memory statistics
        """
        # Baseline measurements
        self.process.cpu_percent()  # Reset counter
        time.sleep(0.1)
        baseline_cpu = self.process.cpu_percent()
        baseline_memory = self.process.memory_info().rss / (1024 * 1024)
        
        # Start monitoring
        cpu_samples = []
        memory_samples = []
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration_sec:
            # Perform resampling
            _ = resample_func(audio_data, orig_sr, target_sr)
            iterations += 1
            
            # Sample metrics
            cpu_samples.append(self.process.cpu_percent())
            memory_samples.append(self.process.memory_info().rss / (1024 * 1024))
            
            # Small delay to allow CPU measurement
            time.sleep(0.01)
        
        cpu_samples = np.array(cpu_samples)
        memory_samples = np.array(memory_samples)
        
        return {
            'cpu_mean_percent': float(np.mean(cpu_samples)),
            'cpu_max_percent': float(np.max(cpu_samples)),
            'cpu_std_percent': float(np.std(cpu_samples)),
            'memory_mean_mb': float(np.mean(memory_samples)),
            'memory_max_mb': float(np.max(memory_samples)),
            'memory_delta_mb': float(np.max(memory_samples) - baseline_memory),
            'iterations_per_sec': float(iterations / duration_sec)
        }
    
    def benchmark_streaming(self, resample_func: Callable,
                          chunk_size: int, orig_sr: int, target_sr: int,
                          duration_sec: float = 10.0,
                          target_latency_ms: float = 50.0) -> StreamingPerformance:
        """
        Benchmark streaming/real-time performance
        
        Args:
            resample_func: Resampling function
            chunk_size: Size of audio chunks
            orig_sr: Original sample rate
            target_sr: Target sample rate
            duration_sec: Test duration
            target_latency_ms: Target processing latency
            
        Returns:
            Streaming performance metrics
        """
        # Calculate timing constraints
        chunk_duration_ms = (chunk_size / orig_sr) * 1000
        max_processing_time = min(target_latency_ms, chunk_duration_ms * 0.8)
        
        # Generate test chunks
        num_chunks = int(duration_sec * orig_sr / chunk_size)
        
        latencies = []
        dropped = 0
        xruns = 0
        
        for i in range(num_chunks):
            # Generate chunk (simulate real-time input)
            chunk = np.random.randn(chunk_size).astype(np.float32) * 0.5
            
            # Measure processing time
            start = time.perf_counter()
            try:
                _ = resample_func(chunk, orig_sr, target_sr)
            except Exception:
                dropped += 1
                continue
            
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            
            # Check for deadline miss (xrun)
            if latency_ms > max_processing_time:
                xruns += 1
            
            # Simulate real-time pacing
            sleep_time = max(0, (chunk_duration_ms - latency_ms) / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        latencies = np.array(latencies) if latencies else np.array([0])
        
        return StreamingPerformance(
            method=resample_func.__name__,
            chunks_processed=num_chunks - dropped,
            total_time_ms=float(np.sum(latencies)),
            avg_latency_ms=float(np.mean(latencies)),
            max_latency_ms=float(np.max(latencies)),
            min_latency_ms=float(np.min(latencies)),
            latency_std_ms=float(np.std(latencies)),
            dropped_chunks=dropped,
            xruns=xruns
        )
    
    def benchmark_batch_sizes(self, resample_func: Callable,
                            batch_sizes: List[int],
                            orig_sr: int, target_sr: int) -> Dict[int, Dict[str, float]]:
        """
        Benchmark performance across different batch sizes
        
        Args:
            resample_func: Resampling function
            batch_sizes: List of batch sizes to test
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Performance metrics for each batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            # Generate test data
            audio_data = np.random.randn(batch_size).astype(np.float32) * 0.5
            
            # Measure performance
            timing = self.measure_processing_time(
                resample_func, audio_data, orig_sr, target_sr, iterations=50
            )
            
            # Calculate efficiency metrics
            samples_per_ms = batch_size / timing['mean_ms']
            realtime_factor = (batch_size / orig_sr * 1000) / timing['mean_ms']
            
            results[batch_size] = {
                **timing,
                'samples_per_ms': samples_per_ms,
                'realtime_factor': realtime_factor,  # >1 means faster than realtime
                'efficiency_score': samples_per_ms / (timing['std_ms'] + 1)  # Stability weighted
            }
        
        return results
    
    def profile_cache_efficiency(self, resample_func: Callable,
                                audio_data: np.ndarray,
                                orig_sr: int, target_sr: int,
                                iterations: int = 100) -> Dict[str, float]:
        """
        Profile cache efficiency and memory access patterns
        
        Args:
            resample_func: Resampling function
            audio_data: Test audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            iterations: Number of iterations
            
        Returns:
            Cache efficiency metrics
        """
        results = {}
        
        # Test with data that fits in cache
        small_data = audio_data[:1024]  # L1/L2 cache size
        small_timing = self.measure_processing_time(
            resample_func, small_data, orig_sr, target_sr, iterations
        )
        
        # Test with data that exceeds cache
        large_data = np.tile(audio_data, 10)  # Force cache misses
        large_timing = self.measure_processing_time(
            resample_func, large_data, orig_sr, target_sr, iterations // 10
        )
        
        # Normalized timing per sample
        small_per_sample = small_timing['mean_ms'] / len(small_data)
        large_per_sample = large_timing['mean_ms'] / len(large_data)
        
        # Cache efficiency ratio (lower is better)
        cache_penalty = large_per_sample / small_per_sample if small_per_sample > 0 else 1.0
        
        results['cache_fit_ms_per_sample'] = small_per_sample
        results['cache_miss_ms_per_sample'] = large_per_sample
        results['cache_penalty_ratio'] = cache_penalty
        results['cache_sensitive'] = cache_penalty > 1.5  # Significant penalty
        
        return results
    
    def evaluate_realtime_capability(self, resample_func: Callable,
                                   chunk_size: int,
                                   orig_sr: int, target_sr: int) -> Dict[str, any]:
        """
        Evaluate if method is suitable for real-time processing
        
        Args:
            resample_func: Resampling function
            chunk_size: Typical chunk size
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Real-time capability assessment
        """
        # Generate test chunk
        test_chunk = np.random.randn(chunk_size).astype(np.float32) * 0.5
        
        # Measure timing
        timing = self.measure_processing_time(
            resample_func, test_chunk, orig_sr, target_sr, iterations=100
        )
        
        # Measure CPU
        cpu_memory = self.measure_cpu_memory(
            resample_func, test_chunk, orig_sr, target_sr, duration_sec=2.0
        )
        
        # Calculate real-time constraints
        chunk_duration_ms = (chunk_size / orig_sr) * 1000
        deadline_ms = chunk_duration_ms * 0.8  # 80% of chunk duration
        
        # Evaluation criteria
        meets_latency = timing['p95_ms'] < self.realtime_thresholds['latency_ms']
        meets_deadline = timing['p95_ms'] < deadline_ms
        meets_cpu = cpu_memory['cpu_mean_percent'] < self.realtime_thresholds['cpu_percent']
        meets_jitter = timing['std_ms'] < self.realtime_thresholds['jitter_ms']
        
        # Overall assessment
        is_realtime_capable = all([meets_latency, meets_deadline, meets_cpu, meets_jitter])
        
        # Calculate headroom
        latency_headroom = max(0, (deadline_ms - timing['p95_ms']) / deadline_ms * 100)
        cpu_headroom = max(0, (self.realtime_thresholds['cpu_percent'] - cpu_memory['cpu_mean_percent']))
        
        return {
            'is_realtime_capable': is_realtime_capable,
            'meets_latency': meets_latency,
            'meets_deadline': meets_deadline,
            'meets_cpu': meets_cpu,
            'meets_jitter': meets_jitter,
            'latency_p95_ms': timing['p95_ms'],
            'deadline_ms': deadline_ms,
            'latency_headroom_percent': latency_headroom,
            'cpu_usage_percent': cpu_memory['cpu_mean_percent'],
            'cpu_headroom_percent': cpu_headroom,
            'jitter_ms': timing['std_ms'],
            'realtime_factor': chunk_duration_ms / timing['mean_ms'],
            'recommendation': self._get_realtime_recommendation(
                is_realtime_capable, latency_headroom, cpu_headroom
            )
        }
    
    def _get_realtime_recommendation(self, capable: bool, 
                                    latency_headroom: float,
                                    cpu_headroom: float) -> str:
        """Generate recommendation based on real-time evaluation"""
        if not capable:
            return "Not suitable for real-time processing"
        elif latency_headroom > 50 and cpu_headroom > 30:
            return "Excellent for real-time with significant headroom"
        elif latency_headroom > 25 and cpu_headroom > 15:
            return "Good for real-time processing"
        elif latency_headroom > 10 and cpu_headroom > 5:
            return "Marginal for real-time, may struggle under load"
        else:
            return "Borderline real-time capability, optimize settings"
    
    def compare_methods(self, methods: Dict[str, Callable],
                      chunk_size: int = 2400,
                      orig_sr: int = 48000,
                      target_sr: int = 24000) -> pd.DataFrame:
        """
        Compare multiple resampling methods
        
        Args:
            methods: Dictionary of method names and functions
            chunk_size: Test chunk size
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Comparison DataFrame
        """
        import pandas as pd
        
        comparison = []
        test_data = np.random.randn(chunk_size).astype(np.float32) * 0.5
        
        for name, func in methods.items():
            print(f"Benchmarking {name}...")
            
            # Timing
            timing = self.measure_processing_time(
                func, test_data, orig_sr, target_sr, iterations=100
            )
            
            # CPU/Memory
            resources = self.measure_cpu_memory(
                func, test_data, orig_sr, target_sr, duration_sec=3.0
            )
            
            # Real-time capability
            realtime = self.evaluate_realtime_capability(
                func, chunk_size, orig_sr, target_sr
            )
            
            comparison.append({
                'method': name,
                'latency_mean_ms': timing['mean_ms'],
                'latency_p95_ms': timing['p95_ms'],
                'jitter_ms': timing['std_ms'],
                'cpu_percent': resources['cpu_mean_percent'],
                'memory_mb': resources['memory_mean_mb'],
                'realtime_capable': realtime['is_realtime_capable'],
                'realtime_factor': realtime['realtime_factor'],
                'efficiency_score': 1000 / (timing['mean_ms'] * resources['cpu_mean_percent'] + 1)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('efficiency_score', ascending=False)
        
        return df
    
    def generate_report(self, results: Dict, output_path: Path):
        """
        Generate performance benchmark report
        
        Args:
            results: Benchmark results
            output_path: Path to save report
        """
        report = {
            'platform': asdict(self.platform_info),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'thresholds': self.realtime_thresholds
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report saved to {output_path}")


def run_raspberry_pi_optimization_test(resample_func: Callable) -> Dict[str, float]:
    """
    Run Raspberry Pi specific optimization tests
    
    Args:
        resample_func: Resampling function to test
        
    Returns:
        Pi-specific performance metrics
    """
    benchmark = PerformanceBenchmark()
    
    if not benchmark.platform_info.is_raspberry_pi:
        return {'error': 'Not running on Raspberry Pi'}
    
    results = {}
    
    # Test different chunk sizes for Pi
    pi_chunk_sizes = [600, 1200, 2400, 4800]  # 25ms, 50ms, 100ms, 200ms at 24kHz
    
    for chunk_size in pi_chunk_sizes:
        test_data = np.random.randn(chunk_size).astype(np.float32) * 0.5
        
        timing = benchmark.measure_processing_time(
            resample_func, test_data, 48000, 24000, iterations=50
        )
        
        results[f'chunk_{chunk_size}'] = {
            'latency_ms': timing['mean_ms'],
            'jitter_ms': timing['std_ms'],
            'suitable': timing['p95_ms'] < (chunk_size / 48000 * 1000 * 0.5)  # <50% of chunk duration
        }
    
    # Find optimal chunk size
    suitable_chunks = [k for k, v in results.items() if v.get('suitable', False)]
    if suitable_chunks:
        # Pick largest suitable chunk for efficiency
        results['optimal_chunk'] = suitable_chunks[-1]
    else:
        results['optimal_chunk'] = 'none_suitable'
    
    return results