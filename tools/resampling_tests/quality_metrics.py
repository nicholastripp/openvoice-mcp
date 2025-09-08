"""
Quality metrics module for resampling analysis

Provides detailed quality measurements specifically for resampling evaluation:
- Spectral distortion analysis
- Aliasing measurement
- Phase coherence testing
- Transient response evaluation
- Intermodulation distortion
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings

try:
    from scipy import signal
    from scipy.fft import rfft, rfftfreq, fft, ifft
    from scipy.signal import windows
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - quality metrics limited")


class ResamplingQualityMetrics:
    """Advanced quality metrics for resampling evaluation"""
    
    def __init__(self):
        """Initialize quality metrics calculator"""
        self.window_functions = {
            'blackman': windows.blackman,
            'hamming': windows.hamming,
            'hann': windows.hann,
            'kaiser': lambda N: windows.kaiser(N, beta=8.6)
        }
    
    def spectral_distortion(self, original: np.ndarray, resampled: np.ndarray,
                           orig_sr: int, target_sr: int) -> float:
        """
        Calculate spectral distortion between original and resampled signals
        
        Returns:
            Spectral distortion in dB
        """
        # Compute magnitude spectra
        orig_fft = rfft(original, n=8192)
        orig_mag = np.abs(orig_fft)
        
        resamp_fft = rfft(resampled, n=8192)
        resamp_mag = np.abs(resamp_fft)
        
        # Normalize to same length for comparison
        min_len = min(len(orig_mag), len(resamp_mag))
        orig_mag = orig_mag[:min_len]
        resamp_mag = resamp_mag[:min_len]
        
        # Calculate spectral distortion
        eps = 1e-10
        distortion = np.sqrt(np.mean((20 * np.log10(resamp_mag + eps) - 
                                      20 * np.log10(orig_mag + eps)) ** 2))
        
        return float(distortion)
    
    def measure_aliasing_artifacts(self, signal: np.ndarray, sample_rate: int,
                                  original_rate: int) -> Dict[str, float]:
        """
        Comprehensive aliasing measurement
        
        Returns:
            Dictionary with aliasing metrics
        """
        # Apply window to reduce spectral leakage
        windowed = signal * windows.blackman(len(signal))
        
        # Compute spectrum
        spectrum = rfft(windowed, n=len(signal) * 4)
        freqs = rfftfreq(len(signal) * 4, 1/sample_rate)
        magnitude = np.abs(spectrum)
        
        # Nyquist frequencies
        nyquist_new = sample_rate / 2
        nyquist_orig = original_rate / 2
        
        # Measure energy in different regions
        results = {}
        
        # Energy in passband (0 to 0.4 * Nyquist)
        passband_mask = freqs < (0.4 * nyquist_new)
        passband_energy = np.sum(magnitude[passband_mask] ** 2)
        
        # Energy in transition band (0.4 to 0.6 * Nyquist)
        transition_mask = (freqs >= 0.4 * nyquist_new) & (freqs < 0.6 * nyquist_new)
        transition_energy = np.sum(magnitude[transition_mask] ** 2)
        
        # Energy in stopband (above 0.6 * Nyquist)
        stopband_mask = freqs >= (0.6 * nyquist_new)
        stopband_energy = np.sum(magnitude[stopband_mask] ** 2)
        
        # Calculate aliasing metrics
        total_energy = passband_energy + transition_energy + stopband_energy
        
        if total_energy > 0:
            results['passband_ratio'] = float(passband_energy / total_energy)
            results['transition_ratio'] = float(transition_energy / total_energy)
            results['stopband_ratio'] = float(stopband_energy / total_energy)
            results['aliasing_db'] = float(-10 * np.log10(stopband_energy / passband_energy + 1e-10))
        else:
            results['passband_ratio'] = 1.0
            results['transition_ratio'] = 0.0
            results['stopband_ratio'] = 0.0
            results['aliasing_db'] = 60.0
        
        # Find spurious peaks above Nyquist/2
        peak_threshold = np.max(magnitude) * 0.1
        spurious_peaks = magnitude[stopband_mask] > peak_threshold
        results['spurious_peak_count'] = int(np.sum(spurious_peaks))
        
        return results
    
    def phase_coherence(self, original: np.ndarray, resampled: np.ndarray,
                       orig_sr: int, target_sr: int) -> Dict[str, float]:
        """
        Analyze phase coherence and group delay
        
        Returns:
            Phase coherence metrics
        """
        # Compute phase spectra
        orig_fft = fft(original)
        resamp_fft = fft(resampled)
        
        orig_phase = np.angle(orig_fft[:len(orig_fft)//2])
        resamp_phase = np.angle(resamp_fft[:len(resamp_fft)//2])
        
        # Normalize lengths
        min_len = min(len(orig_phase), len(resamp_phase))
        orig_phase = orig_phase[:min_len]
        resamp_phase = resamp_phase[:min_len]
        
        # Unwrap phases
        orig_unwrapped = np.unwrap(orig_phase)
        resamp_unwrapped = np.unwrap(resamp_phase)
        
        # Calculate phase difference
        phase_diff = resamp_unwrapped - orig_unwrapped
        
        # Group delay (derivative of phase)
        group_delay_orig = np.diff(orig_unwrapped)
        group_delay_resamp = np.diff(resamp_unwrapped)
        
        results = {
            'phase_rmse': float(np.sqrt(np.mean(phase_diff ** 2))),
            'phase_max_deviation': float(np.max(np.abs(phase_diff))),
            'group_delay_variation': float(np.std(group_delay_resamp - group_delay_orig[:len(group_delay_resamp)])),
            'phase_linearity': float(np.corrcoef(np.arange(len(resamp_unwrapped)), resamp_unwrapped)[0, 1])
        }
        
        return results
    
    def transient_response(self, method_func, orig_sr: int, target_sr: int) -> Dict[str, float]:
        """
        Test transient response using impulse and step signals
        
        Args:
            method_func: Resampling function to test
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Transient response metrics
        """
        results = {}
        
        # Generate impulse
        impulse = np.zeros(1000)
        impulse[500] = 1.0
        
        # Resample impulse
        try:
            impulse_response = method_func(impulse, orig_sr, target_sr)
            
            # Find main peak
            main_peak_idx = np.argmax(np.abs(impulse_response))
            main_peak_value = impulse_response[main_peak_idx]
            
            # Pre-ringing (energy before main peak)
            pre_ring = impulse_response[:main_peak_idx]
            pre_ring_energy = np.sum(pre_ring ** 2)
            
            # Post-ringing (energy after main peak)
            post_ring = impulse_response[main_peak_idx+1:]
            post_ring_energy = np.sum(post_ring ** 2)
            
            results['impulse_peak_preservation'] = float(np.abs(main_peak_value))
            results['pre_ringing_db'] = float(-10 * np.log10(pre_ring_energy + 1e-10))
            results['post_ringing_db'] = float(-10 * np.log10(post_ring_energy + 1e-10))
            
            # Generate step function
            step = np.ones(1000)
            step[:500] = 0
            
            # Resample step
            step_response = method_func(step, orig_sr, target_sr)
            
            # Find transition region
            transition_start = int(400 * target_sr / orig_sr)
            transition_end = int(600 * target_sr / orig_sr)
            transition = step_response[transition_start:transition_end]
            
            # Measure overshoot and rise time
            if len(transition) > 0:
                overshoot = np.max(transition) - 1.0
                undershoot = np.min(transition)
                
                # Rise time (10% to 90%)
                sorted_trans = np.sort(transition)
                val_10 = sorted_trans[int(len(sorted_trans) * 0.1)]
                val_90 = sorted_trans[int(len(sorted_trans) * 0.9)]
                rise_samples = np.sum((transition >= val_10) & (transition <= val_90))
                
                results['step_overshoot'] = float(overshoot)
                results['step_undershoot'] = float(undershoot)
                results['rise_time_samples'] = int(rise_samples)
            else:
                results['step_overshoot'] = 0.0
                results['step_undershoot'] = 0.0
                results['rise_time_samples'] = 0
                
        except Exception as e:
            # Return default values on error
            results = {
                'impulse_peak_preservation': 0.0,
                'pre_ringing_db': -40.0,
                'post_ringing_db': -40.0,
                'step_overshoot': 0.0,
                'step_undershoot': 0.0,
                'rise_time_samples': 0
            }
        
        return results
    
    def intermodulation_distortion(self, signal: np.ndarray, sample_rate: int) -> float:
        """
        Measure intermodulation distortion using two-tone test
        
        Returns:
            IMD in percentage
        """
        if len(signal) < sample_rate // 10:
            return 0.0
        
        # Apply window
        windowed = signal * windows.blackman(len(signal))
        
        # Compute spectrum
        spectrum = rfft(windowed)
        freqs = rfftfreq(len(windowed), 1/sample_rate)
        magnitude = np.abs(spectrum)
        
        # For a two-tone test at f1 and f2, IMD products appear at:
        # 2*f1 - f2, 2*f2 - f1, f1 + f2, f1 - f2, etc.
        
        # Find the two strongest peaks (assumed to be test tones)
        peak_indices = signal.argpartition(-2)[-2:]
        
        if len(peak_indices) < 2:
            return 0.0
        
        # Simple IMD estimate: ratio of distortion products to fundamentals
        fundamental_energy = np.sum(magnitude[peak_indices] ** 2)
        total_energy = np.sum(magnitude ** 2)
        distortion_energy = total_energy - fundamental_energy
        
        if fundamental_energy > 0:
            imd = np.sqrt(distortion_energy / fundamental_energy) * 100
            return float(imd)
        else:
            return 0.0
    
    def signal_to_noise_floor(self, signal: np.ndarray, sample_rate: int) -> float:
        """
        Measure signal-to-noise floor ratio
        
        Returns:
            SNR to noise floor in dB
        """
        # Find quiet portions (assumed to be noise)
        rms = np.sqrt(np.mean(signal ** 2))
        noise_threshold = rms * 0.1
        
        # Identify noise regions
        noise_mask = np.abs(signal) < noise_threshold
        
        if np.sum(noise_mask) > len(signal) // 10:
            noise = signal[noise_mask]
            noise_rms = np.sqrt(np.mean(noise ** 2))
            
            if noise_rms > 0:
                snr = 20 * np.log10(rms / noise_rms)
                return float(snr)
        
        # If no clear noise floor, return high SNR
        return 90.0
    
    def frequency_warping(self, original: np.ndarray, resampled: np.ndarray,
                         orig_sr: int, target_sr: int, test_freqs: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Detect frequency warping or shifts
        
        Args:
            original: Original signal
            resampled: Resampled signal
            orig_sr: Original sample rate
            target_sr: Target sample rate
            test_freqs: Specific frequencies to test
            
        Returns:
            Frequency warping metrics
        """
        if test_freqs is None:
            # Default test frequencies for voice range
            test_freqs = [100, 200, 500, 1000, 2000, 4000, 8000]
        
        results = {}
        
        for freq in test_freqs:
            if freq > target_sr / 2:
                continue  # Skip frequencies above Nyquist
            
            # Find peak near expected frequency in spectrum
            spectrum = rfft(resampled)
            freqs = rfftfreq(len(resampled), 1/target_sr)
            
            # Search window around expected frequency (±10%)
            freq_min = freq * 0.9
            freq_max = freq * 1.1
            mask = (freqs >= freq_min) & (freqs <= freq_max)
            
            if np.any(mask):
                # Find actual peak frequency
                masked_spectrum = np.abs(spectrum[mask])
                masked_freqs = freqs[mask]
                peak_idx = np.argmax(masked_spectrum)
                actual_freq = masked_freqs[peak_idx]
                
                # Calculate frequency shift
                shift = actual_freq - freq
                shift_percent = (shift / freq) * 100
                
                results[f'shift_{int(freq)}Hz'] = float(shift_percent)
        
        # Overall warping metric
        if results:
            results['max_warping_percent'] = float(np.max(np.abs(list(results.values()))))
            results['mean_warping_percent'] = float(np.mean(np.abs(list(results.values()))))
        else:
            results['max_warping_percent'] = 0.0
            results['mean_warping_percent'] = 0.0
        
        return results
    
    def perceptual_quality_metrics(self, original: np.ndarray, resampled: np.ndarray,
                                 orig_sr: int, target_sr: int) -> Dict[str, float]:
        """
        Calculate perceptual quality metrics relevant to speech
        
        Returns:
            Perceptual quality metrics
        """
        results = {}
        
        # Weighted frequency bands for speech
        speech_bands = {
            'low_energy': (50, 250, 0.1),      # Low frequency energy
            'fundamental': (250, 500, 0.3),     # Fundamental frequency range
            'formant1': (500, 1500, 0.3),      # First formant region
            'formant2': (1500, 3000, 0.2),     # Second formant region
            'sibilance': (3000, 8000, 0.1)     # Sibilance and clarity
        }
        
        # Calculate weighted spectral difference
        spectrum_orig = rfft(original)
        spectrum_resamp = rfft(resampled)
        freqs_orig = rfftfreq(len(original), 1/orig_sr)
        freqs_resamp = rfftfreq(len(resampled), 1/target_sr)
        
        weighted_error = 0.0
        
        for band_name, (low_freq, high_freq, weight) in speech_bands.items():
            # Original band energy
            mask_orig = (freqs_orig >= low_freq) & (freqs_orig < high_freq)
            if np.any(mask_orig):
                energy_orig = np.mean(np.abs(spectrum_orig[mask_orig]) ** 2)
            else:
                energy_orig = 0.0
            
            # Resampled band energy
            mask_resamp = (freqs_resamp >= low_freq) & (freqs_resamp < high_freq)
            if np.any(mask_resamp):
                energy_resamp = np.mean(np.abs(spectrum_resamp[mask_resamp]) ** 2)
            else:
                energy_resamp = 0.0
            
            # Band error
            if energy_orig > 0:
                band_error = np.abs(energy_resamp - energy_orig) / energy_orig
            else:
                band_error = 0.0
            
            results[f'{band_name}_preservation'] = float(1.0 - band_error)
            weighted_error += band_error * weight
        
        # Overall perceptual score (0-100)
        results['perceptual_score'] = float(max(0, (1.0 - weighted_error) * 100))
        
        # Clarity metric (high frequency preservation)
        hf_mask_orig = freqs_orig > 2000
        hf_mask_resamp = freqs_resamp > 2000
        
        if np.any(hf_mask_orig) and np.any(hf_mask_resamp):
            hf_orig = np.mean(np.abs(spectrum_orig[hf_mask_orig]))
            hf_resamp = np.mean(np.abs(spectrum_resamp[hf_mask_resamp]))
            results['clarity_index'] = float(hf_resamp / (hf_orig + 1e-10))
        else:
            results['clarity_index'] = 0.0
        
        return results


def calculate_comprehensive_metrics(original: np.ndarray, resampled: np.ndarray,
                                  orig_sr: int, target_sr: int,
                                  method_func=None) -> Dict[str, any]:
    """
    Calculate all quality metrics for a resampling operation
    
    Args:
        original: Original signal
        resampled: Resampled signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        method_func: Optional resampling function for transient tests
        
    Returns:
        Dictionary with all quality metrics
    """
    metrics = ResamplingQualityMetrics()
    
    results = {
        'spectral_distortion_db': metrics.spectral_distortion(
            original, resampled, orig_sr, target_sr
        ),
        'aliasing': metrics.measure_aliasing_artifacts(
            resampled, target_sr, orig_sr
        ),
        'phase_coherence': metrics.phase_coherence(
            original, resampled, orig_sr, target_sr
        ),
        'imd_percent': metrics.intermodulation_distortion(
            resampled, target_sr
        ),
        'snr_noise_floor_db': metrics.signal_to_noise_floor(
            resampled, target_sr
        ),
        'frequency_warping': metrics.frequency_warping(
            original, resampled, orig_sr, target_sr
        ),
        'perceptual': metrics.perceptual_quality_metrics(
            original, resampled, orig_sr, target_sr
        )
    }
    
    # Add transient response if method function provided
    if method_func is not None:
        results['transient'] = metrics.transient_response(
            method_func, orig_sr, target_sr
        )
    
    return results