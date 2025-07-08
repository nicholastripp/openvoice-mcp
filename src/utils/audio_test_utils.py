"""
Audio test utilities for generating test audio data
"""
import numpy as np
from typing import Optional
from utils.logger import get_logger


def generate_test_audio(
    duration_seconds: float = 2.0,
    sample_rate: int = 24000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    noise_level: float = 0.05
) -> bytes:
    """
    Generate test audio data in PCM16 format
    
    Args:
        duration_seconds: Duration of audio in seconds
        sample_rate: Sample rate in Hz
        frequency: Tone frequency in Hz
        amplitude: Amplitude (0.0 to 1.0)
        noise_level: Amount of noise to add (0.0 to 1.0)
        
    Returns:
        PCM16 audio data as bytes
    """
    logger = get_logger("AudioTestUtils")
    
    try:
        # Generate time array
        num_samples = int(sample_rate * duration_seconds)
        t = np.linspace(0, duration_seconds, num_samples, False)
        
        # Generate base tone
        audio_signal = np.sin(2 * np.pi * frequency * t) * amplitude
        
        # Add some harmonic complexity to make it more realistic
        audio_signal += np.sin(2 * np.pi * frequency * 2 * t) * (amplitude * 0.2)
        audio_signal += np.sin(2 * np.pi * frequency * 0.5 * t) * (amplitude * 0.1)
        
        # Add small amount of noise to make it more realistic
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, num_samples)
            audio_signal += noise
        
        # Apply envelope to avoid clicks
        envelope_samples = int(sample_rate * 0.01)  # 10ms fade
        if envelope_samples > 0:
            fade_in = np.linspace(0, 1, envelope_samples)
            fade_out = np.linspace(1, 0, envelope_samples)
            audio_signal[:envelope_samples] *= fade_in
            audio_signal[-envelope_samples:] *= fade_out
        
        # Clamp to valid range
        audio_signal = np.clip(audio_signal, -1.0, 1.0)
        
        # Convert to PCM16
        pcm16_data = (audio_signal * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = pcm16_data.tobytes()
        
        logger.info(f"Generated {len(audio_bytes)} bytes of test audio ({duration_seconds}s, {frequency}Hz)")
        
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Error generating test audio: {e}")
        raise


def generate_speech_like_audio(
    duration_seconds: float = 2.0,
    sample_rate: int = 24000,
    base_frequency: float = 150.0,
    amplitude: float = 0.4
) -> bytes:
    """
    Generate speech-like audio with varying frequency and amplitude
    
    Args:
        duration_seconds: Duration of audio in seconds
        sample_rate: Sample rate in Hz
        base_frequency: Base frequency in Hz (typical male voice ~150Hz, female ~250Hz)
        amplitude: Base amplitude (0.0 to 1.0)
        
    Returns:
        PCM16 audio data as bytes
    """
    logger = get_logger("AudioTestUtils")
    
    try:
        num_samples = int(sample_rate * duration_seconds)
        t = np.linspace(0, duration_seconds, num_samples, False)
        
        # Create frequency modulation to simulate speech patterns
        freq_modulation = base_frequency * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))  # 2Hz modulation
        
        # Create amplitude modulation to simulate speech envelope
        amp_modulation = amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))  # 3Hz modulation
        
        # Generate the base signal
        audio_signal = np.zeros(num_samples)
        
        # Add multiple harmonics with varying amplitudes
        for harmonic in range(1, 6):
            harmonic_freq = freq_modulation * harmonic
            harmonic_amp = amp_modulation / (harmonic * 1.5)  # Decrease amplitude for higher harmonics
            
            # Create phase that changes over time
            phase = np.cumsum(2 * np.pi * harmonic_freq / sample_rate)
            audio_signal += harmonic_amp * np.sin(phase)
        
        # Add formant-like filtering by adding resonances
        formant1 = 0.1 * np.sin(2 * np.pi * 800 * t)  # First formant around 800Hz
        formant2 = 0.05 * np.sin(2 * np.pi * 1200 * t)  # Second formant around 1200Hz
        audio_signal += formant1 + formant2
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.02, num_samples)
        audio_signal += noise
        
        # Apply speech-like envelope
        envelope_samples = int(sample_rate * 0.005)  # 5ms fade
        if envelope_samples > 0:
            fade_in = np.linspace(0, 1, envelope_samples)
            fade_out = np.linspace(1, 0, envelope_samples)
            audio_signal[:envelope_samples] *= fade_in
            audio_signal[-envelope_samples:] *= fade_out
        
        # Clamp to valid range
        audio_signal = np.clip(audio_signal, -1.0, 1.0)
        
        # Convert to PCM16
        pcm16_data = (audio_signal * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = pcm16_data.tobytes()
        
        logger.info(f"Generated {len(audio_bytes)} bytes of speech-like audio ({duration_seconds}s, {base_frequency}Hz base)")
        
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Error generating speech-like audio: {e}")
        raise


def validate_audio_content(audio_data: bytes, sample_rate: int = 24000) -> dict:
    """
    Validate and analyze audio content
    
    Args:
        audio_data: PCM16 audio data as bytes
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with audio analysis results
    """
    logger = get_logger("AudioTestUtils")
    
    try:
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate basic statistics
        duration_ms = len(audio_data) / 2 / sample_rate * 1000
        max_amplitude = np.max(np.abs(samples))
        avg_amplitude = np.mean(np.abs(samples))
        rms_amplitude = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        # Check for clipping
        clipping_samples = np.sum(np.abs(samples) >= 32767)
        clipping_percentage = (clipping_samples / len(samples)) * 100
        
        # Check for silence
        silence_threshold = 100  # Very low threshold
        silence_samples = np.sum(np.abs(samples) < silence_threshold)
        silence_percentage = (silence_samples / len(samples)) * 100
        
        # Basic frequency analysis (simplified)
        # Convert to float and apply window
        float_samples = samples.astype(np.float32) / 32767.0
        windowed = float_samples * np.hanning(len(float_samples))
        
        # FFT
        fft_result = np.fft.fft(windowed)
        frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)
        
        # Find dominant frequency
        magnitude = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])  # Only positive frequencies
        dominant_frequency = abs(frequencies[dominant_freq_idx])
        
        analysis = {
            'duration_ms': duration_ms,
            'sample_count': len(samples),
            'max_amplitude': int(max_amplitude),
            'avg_amplitude': float(avg_amplitude),
            'rms_amplitude': float(rms_amplitude),
            'clipping_percentage': clipping_percentage,
            'silence_percentage': silence_percentage,
            'dominant_frequency': dominant_frequency,
            'is_valid': True
        }
        
        # Determine if audio is valid
        if silence_percentage > 95:
            analysis['is_valid'] = False
            analysis['validation_error'] = "Audio is mostly silence"
        elif clipping_percentage > 10:
            analysis['is_valid'] = False
            analysis['validation_error'] = "Audio has excessive clipping"
        elif max_amplitude < 100:
            analysis['is_valid'] = False
            analysis['validation_error'] = "Audio amplitude too low"
        
        logger.debug(f"Audio analysis: {duration_ms:.1f}ms, max_amp={max_amplitude}, dominant_freq={dominant_frequency:.1f}Hz")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error validating audio content: {e}")
        return {
            'is_valid': False,
            'validation_error': str(e)
        }