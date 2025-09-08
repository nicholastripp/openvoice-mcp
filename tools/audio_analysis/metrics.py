"""
Audio metrics calculation module

Provides comprehensive audio quality metrics including:
- RMS, Peak, Peak-to-Average ratio
- Total Harmonic Distortion (THD)
- Signal-to-Noise Ratio (SNR)
- Clipping detection
- DC offset analysis
- Frequency response analysis
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq, rfft, rfftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some metrics will be limited")


class AudioMetrics:
    """Calculate comprehensive audio quality metrics"""
    
    def __init__(self):
        """Initialize audio metrics calculator"""
        self.reference_level = 1.0  # Full scale digital
    
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) level
        
        Args:
            audio_data: Audio samples
            
        Returns:
            RMS level (0.0 to 1.0)
        """
        if len(audio_data) == 0:
            return 0.0
        
        return float(np.sqrt(np.mean(audio_data ** 2)))
    
    def calculate_peak(self, audio_data: np.ndarray) -> float:
        """
        Calculate peak amplitude
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Peak amplitude (0.0 to 1.0)
        """
        if len(audio_data) == 0:
            return 0.0
        
        return float(np.max(np.abs(audio_data)))
    
    def calculate_peak_to_average(self, audio_data: np.ndarray) -> float:
        """
        Calculate peak-to-average ratio (crest factor)
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Peak-to-average ratio in dB
        """
        rms = self.calculate_rms(audio_data)
        peak = self.calculate_peak(audio_data)
        
        if rms == 0:
            return 0.0
        
        # Crest factor in dB
        return float(20 * np.log10(peak / rms))
    
    def calculate_thd(self, audio_data: np.ndarray, sample_rate: int, 
                     fundamental_freq: Optional[float] = None) -> float:
        """
        Calculate Total Harmonic Distortion (THD)
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            fundamental_freq: Expected fundamental frequency (auto-detect if None)
            
        Returns:
            THD as percentage
        """
        if not SCIPY_AVAILABLE or len(audio_data) < sample_rate // 10:
            return 0.0
        
        try:
            # Apply window to reduce spectral leakage
            window = signal.windows.blackman(len(audio_data))
            windowed = audio_data * window
            
            # Compute FFT
            fft_data = np.abs(rfft(windowed))
            freqs = rfftfreq(len(windowed), 1/sample_rate)
            
            # Find fundamental frequency if not provided
            if fundamental_freq is None:
                # Find peak in reasonable range (50Hz to 2kHz for voice)
                mask = (freqs > 50) & (freqs < 2000)
                if np.any(mask):
                    peak_idx = np.argmax(fft_data[mask])
                    fundamental_idx = np.where(mask)[0][peak_idx]
                    fundamental_freq = freqs[fundamental_idx]
                else:
                    return 0.0
            else:
                # Find closest frequency bin to specified fundamental
                fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
            
            # Get fundamental amplitude
            fundamental_amplitude = fft_data[fundamental_idx]
            
            if fundamental_amplitude < 1e-10:
                return 0.0
            
            # Calculate harmonics (2nd through 10th)
            harmonic_sum_squared = 0.0
            for harmonic in range(2, 11):
                harmonic_freq = fundamental_freq * harmonic
                if harmonic_freq > sample_rate / 2:
                    break
                
                # Find closest bin to harmonic frequency
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                
                # Allow some tolerance for frequency bin selection
                if harmonic_idx < len(fft_data):
                    harmonic_amplitude = fft_data[harmonic_idx]
                    harmonic_sum_squared += harmonic_amplitude ** 2
            
            # Calculate THD as percentage
            thd = 100.0 * np.sqrt(harmonic_sum_squared) / fundamental_amplitude
            
            return float(min(thd, 100.0))  # Cap at 100%
            
        except Exception as e:
            warnings.warn(f"THD calculation failed: {e}")
            return 0.0
    
    def calculate_snr(self, audio_data: np.ndarray, noise_floor: Optional[float] = None) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR)
        
        Args:
            audio_data: Audio samples
            noise_floor: Known noise floor (will estimate if None)
            
        Returns:
            SNR in dB
        """
        if len(audio_data) < 100:
            return 0.0
        
        # Estimate noise floor from quietest 10% of samples if not provided
        if noise_floor is None:
            sorted_abs = np.sort(np.abs(audio_data))
            noise_samples = sorted_abs[:len(sorted_abs)//10]
            noise_floor = np.mean(noise_samples) if len(noise_samples) > 0 else 1e-10
        
        # Calculate signal power (RMS of active signal)
        signal_threshold = noise_floor * 3  # Signal must be 3x noise floor
        signal_samples = audio_data[np.abs(audio_data) > signal_threshold]
        
        if len(signal_samples) == 0:
            return 0.0
        
        signal_power = np.sqrt(np.mean(signal_samples ** 2))
        
        # Avoid log of zero
        if noise_floor < 1e-10:
            noise_floor = 1e-10
        
        # SNR in dB
        snr = 20 * np.log10(signal_power / noise_floor)
        
        return float(max(0, min(snr, 120)))  # Reasonable bounds
    
    def analyze_clipping(self, audio_data: np.ndarray, 
                        threshold: float = 0.99) -> Tuple[int, float]:
        """
        Analyze audio clipping
        
        Args:
            audio_data: Audio samples
            threshold: Clipping threshold (default 0.99)
            
        Returns:
            (clipping_count, clipping_ratio)
        """
        if len(audio_data) == 0:
            return 0, 0.0
        
        # Count samples above threshold
        clipped = np.abs(audio_data) > threshold
        clipping_count = int(np.sum(clipped))
        clipping_ratio = float(clipping_count / len(audio_data))
        
        return clipping_count, clipping_ratio
    
    def calculate_dc_offset(self, audio_data: np.ndarray) -> float:
        """
        Calculate DC offset (bias)
        
        Args:
            audio_data: Audio samples
            
        Returns:
            DC offset value
        """
        if len(audio_data) == 0:
            return 0.0
        
        return float(np.mean(audio_data))
    
    def calculate_frequency_response(self, audio_data: np.ndarray, 
                                    sample_rate: int,
                                    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Calculate frequency response in specified bands
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            freq_bands: Dictionary of band names to (low_freq, high_freq) tuples
            
        Returns:
            Dictionary of band names to power levels in dB
        """
        if not SCIPY_AVAILABLE or len(audio_data) < 1024:
            return {}
        
        # Default frequency bands if not specified
        if freq_bands is None:
            freq_bands = {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "low_mid": (250, 500),
                "mid": (500, 2000),
                "upper_mid": (2000, 4000),
                "presence": (4000, 6000),
                "brilliance": (6000, 12000),
                "air": (12000, 20000)
            }
        
        try:
            # Apply window
            window = signal.windows.hann(len(audio_data))
            windowed = audio_data * window
            
            # Compute power spectral density
            freqs, psd = signal.welch(
                windowed,
                sample_rate,
                nperseg=min(len(windowed), 1024),
                scaling='density'
            )
            
            # Calculate power in each band
            response = {}
            for band_name, (low_freq, high_freq) in freq_bands.items():
                # Find frequency bins in range
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.any(mask):
                    # Calculate average power in band
                    band_power = np.mean(psd[mask])
                    
                    # Convert to dB (with reference to full scale)
                    if band_power > 1e-10:
                        response[band_name] = float(10 * np.log10(band_power))
                    else:
                        response[band_name] = -100.0
                else:
                    response[band_name] = -100.0
            
            return response
            
        except Exception as e:
            warnings.warn(f"Frequency response calculation failed: {e}")
            return {}
    
    def calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """
        Calculate dynamic range
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Dynamic range in dB
        """
        if len(audio_data) < 100:
            return 0.0
        
        # Sort absolute values
        sorted_abs = np.sort(np.abs(audio_data))
        
        # Get 95th percentile (loud parts) and 5th percentile (quiet parts)
        loud_level = sorted_abs[int(len(sorted_abs) * 0.95)]
        quiet_level = sorted_abs[int(len(sorted_abs) * 0.05)]
        
        if quiet_level < 1e-10:
            quiet_level = 1e-10
        
        # Dynamic range in dB
        dynamic_range = 20 * np.log10(loud_level / quiet_level)
        
        return float(max(0, min(dynamic_range, 120)))
    
    def detect_feedback(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        Detect potential acoustic feedback (howling)
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            True if feedback detected
        """
        if not SCIPY_AVAILABLE or len(audio_data) < sample_rate:
            return False
        
        try:
            # Compute FFT
            fft_data = np.abs(rfft(audio_data))
            
            # Look for narrow peaks that are much higher than surrounding frequencies
            # This is characteristic of feedback
            peak_prominence = np.max(fft_data) / np.median(fft_data)
            
            # Feedback typically shows as very prominent narrow peaks
            return peak_prominence > 100
            
        except Exception:
            return False
    
    def calculate_zero_crossing_rate(self, audio_data: np.ndarray) -> float:
        """
        Calculate zero-crossing rate (useful for detecting voice/unvoiced)
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Zero-crossing rate (crossings per sample)
        """
        if len(audio_data) < 2:
            return 0.0
        
        # Count sign changes
        signs = np.sign(audio_data)
        sign_changes = np.sum(signs[:-1] != signs[1:])
        
        # Rate per sample
        zcr = sign_changes / len(audio_data)
        
        return float(zcr)