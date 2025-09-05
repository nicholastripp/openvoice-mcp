#!/usr/bin/env python3
"""
Optimized Audio Pipeline for HA Realtime Voice Assistant
Provides separate optimized paths for OpenAI (24kHz) and Porcupine (16kHz)
Implements improvements from APM Phase 1 Tasks 1.1-1.4

This module avoids double resampling and provides optimal audio processing
for both wake word detection and speech-to-text.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from math import gcd
import warnings

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - falling back to basic resampling")


@dataclass
class AudioPipelineConfig:
    """Configuration for optimized audio pipeline"""
    device_sample_rate: int = 48000
    openai_sample_rate: int = 24000
    porcupine_sample_rate: int = 16000
    input_gain: float = 1.0
    wake_word_gain: float = 1.0
    use_polyphase: bool = True
    enable_soft_limiting: bool = True


class OptimizedAudioPipeline:
    """
    Optimized audio pipeline with separate paths for different services
    Implements findings from Task 1.2 (resampling) and Task 1.4 (PCM16 validation)
    """
    
    def __init__(self, config: AudioPipelineConfig = None):
        self.config = config or AudioPipelineConfig()
        
        # Pre-calculate resampling parameters for efficiency
        self._setup_resampling_params()
        
        # Statistics tracking
        self.stats = {
            'openai_processed': 0,
            'porcupine_processed': 0,
            'clipping_events': 0,
            'resampling_time_ms': []
        }
    
    def _setup_resampling_params(self):
        """Pre-calculate resampling parameters for both paths"""
        # OpenAI path (e.g., 48000 -> 24000)
        if self.config.device_sample_rate != self.config.openai_sample_rate:
            g = gcd(self.config.device_sample_rate, self.config.openai_sample_rate)
            self.openai_up = self.config.openai_sample_rate // g
            self.openai_down = self.config.device_sample_rate // g
            self.openai_ratio = self.config.openai_sample_rate / self.config.device_sample_rate
        else:
            self.openai_up = self.openai_down = 1
            self.openai_ratio = 1.0
        
        # Porcupine path (e.g., 48000 -> 16000)
        if self.config.device_sample_rate != self.config.porcupine_sample_rate:
            g = gcd(self.config.device_sample_rate, self.config.porcupine_sample_rate)
            self.porcupine_up = self.config.porcupine_sample_rate // g
            self.porcupine_down = self.config.device_sample_rate // g
            self.porcupine_ratio = self.config.porcupine_sample_rate / self.config.device_sample_rate
        else:
            self.porcupine_up = self.porcupine_down = 1
            self.porcupine_ratio = 1.0
    
    def process_for_openai(self, audio_data: np.ndarray, 
                          apply_gain: bool = True) -> bytes:
        """
        Process audio for OpenAI Realtime API
        
        Args:
            audio_data: Float32 audio at device sample rate
            apply_gain: Whether to apply input gain
            
        Returns:
            PCM16 bytes at 24kHz
        """
        # Apply gain if requested
        if apply_gain and self.config.input_gain != 1.0:
            audio_data = self._apply_gain(audio_data, self.config.input_gain)
        
        # Resample to 24kHz if needed
        if self.openai_ratio != 1.0:
            audio_data = self._resample_optimized(
                audio_data, 
                self.openai_up, 
                self.openai_down,
                'openai'
            )
        
        # Convert to PCM16 (validated in Task 1.4)
        pcm16 = self._convert_to_pcm16(audio_data)
        
        self.stats['openai_processed'] += 1
        return pcm16.tobytes()
    
    def process_for_porcupine(self, audio_data: np.ndarray,
                             apply_gain: bool = True,
                             apply_wake_word_gain: bool = True) -> np.ndarray:
        """
        Process audio for Porcupine wake word detection
        Optimized path that avoids double resampling
        
        Args:
            audio_data: Float32 audio at device sample rate
            apply_gain: Whether to apply input gain
            apply_wake_word_gain: Whether to apply wake word specific gain
            
        Returns:
            PCM16 numpy array at 16kHz
        """
        # Apply gains BEFORE PCM16 conversion for better quality
        total_gain = 1.0
        if apply_gain:
            total_gain *= self.config.input_gain
        if apply_wake_word_gain:
            total_gain *= self.config.wake_word_gain
        
        if total_gain != 1.0:
            audio_data = self._apply_gain(audio_data, total_gain)
        
        # Resample directly to 16kHz (avoiding intermediate 24kHz step)
        if self.porcupine_ratio != 1.0:
            audio_data = self._resample_optimized(
                audio_data,
                self.porcupine_up,
                self.porcupine_down,
                'porcupine'
            )
        
        # Convert to PCM16
        pcm16 = self._convert_to_pcm16(audio_data)
        
        self.stats['porcupine_processed'] += 1
        return pcm16
    
    def process_dual_path(self, audio_data: np.ndarray,
                         apply_gain: bool = True) -> Tuple[bytes, np.ndarray]:
        """
        Process audio for both OpenAI and Porcupine in parallel
        Most efficient when both outputs are needed
        
        Args:
            audio_data: Float32 audio at device sample rate
            apply_gain: Whether to apply input gain
            
        Returns:
            Tuple of (openai_bytes, porcupine_array)
        """
        # Apply common input gain once
        if apply_gain and self.config.input_gain != 1.0:
            audio_data = self._apply_gain(audio_data, self.config.input_gain)
        
        # Process for OpenAI (24kHz)
        if self.openai_ratio != 1.0:
            openai_audio = self._resample_optimized(
                audio_data.copy(),  # Copy to avoid modifying original
                self.openai_up,
                self.openai_down,
                'openai'
            )
        else:
            openai_audio = audio_data.copy()
        
        openai_pcm16 = self._convert_to_pcm16(openai_audio)
        
        # Process for Porcupine (16kHz) with additional wake word gain
        porcupine_audio = audio_data
        if self.config.wake_word_gain != 1.0:
            porcupine_audio = self._apply_gain(porcupine_audio, self.config.wake_word_gain)
        
        if self.porcupine_ratio != 1.0:
            porcupine_audio = self._resample_optimized(
                porcupine_audio,
                self.porcupine_up,
                self.porcupine_down,
                'porcupine'
            )
        
        porcupine_pcm16 = self._convert_to_pcm16(porcupine_audio)
        
        self.stats['openai_processed'] += 1
        self.stats['porcupine_processed'] += 1
        
        return openai_pcm16.tobytes(), porcupine_pcm16
    
    def _resample_optimized(self, audio: np.ndarray, up: int, down: int,
                          target: str) -> np.ndarray:
        """
        Optimized resampling using polyphase method (from Task 1.2)
        
        Args:
            audio: Input audio
            up: Upsampling factor
            down: Downsampling factor
            target: Target system ('openai' or 'porcupine')
            
        Returns:
            Resampled audio
        """
        if not SCIPY_AVAILABLE:
            # Fallback to simple decimation/interpolation
            new_length = int(len(audio) * up / down)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
        
        # Use polyphase resampling for efficiency (Task 1.2 finding)
        if self.config.use_polyphase and up != down:
            # Use Hamming window for good frequency response
            resampled = signal.resample_poly(audio, up, down, window='hamming')
        else:
            # Fallback to FFT-based resampling
            new_length = int(len(audio) * up / down)
            resampled = signal.resample(audio, new_length)
        
        return resampled
    
    def _apply_gain(self, audio: np.ndarray, gain: float) -> np.ndarray:
        """
        Apply gain with soft limiting to prevent harsh clipping
        
        Args:
            audio: Input audio (float32, -1 to 1)
            gain: Gain factor
            
        Returns:
            Gained audio with soft limiting
        """
        # Apply gain
        audio = audio * gain
        
        # Soft limiting if enabled (from Task 1.3 findings)
        if self.config.enable_soft_limiting:
            # Apply tanh soft limiting for values above 0.9
            threshold = 0.9
            mask = np.abs(audio) > threshold
            
            if np.any(mask):
                # Track clipping events
                self.stats['clipping_events'] += 1
                
                # Soft limit using tanh compression
                over = np.abs(audio[mask]) - threshold
                compressed = threshold + (1 - threshold) * np.tanh(over / (1 - threshold))
                audio[mask] = np.sign(audio[mask]) * compressed
        
        # Hard clip as safety
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _convert_to_pcm16(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert float32 audio to PCM16 using validated method from Task 1.4
        
        Args:
            audio: Float32 audio (-1 to 1)
            
        Returns:
            PCM16 audio as int16 array
        """
        # Ensure input is clipped
        audio = np.clip(audio, -1.0, 1.0)
        
        # Use 32767 scaling (validated as optimal in Task 1.4)
        # This provides:
        # - Minimal DC bias (0.417%)
        # - Good symmetry
        # - Excellent SNR (>66 dB)
        pcm16 = (audio * 32767).astype(np.int16)
        
        return pcm16
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'openai_chunks_processed': self.stats['openai_processed'],
            'porcupine_chunks_processed': self.stats['porcupine_processed'],
            'clipping_events': self.stats['clipping_events'],
            'avg_resampling_time_ms': (
                np.mean(self.stats['resampling_time_ms']) 
                if self.stats['resampling_time_ms'] else 0
            )
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'openai_processed': 0,
            'porcupine_processed': 0,
            'clipping_events': 0,
            'resampling_time_ms': []
        }


def create_optimized_pipeline(device_rate: int = 48000,
                             input_gain: float = 1.0,
                             wake_word_gain: float = 1.0) -> OptimizedAudioPipeline:
    """
    Factory function to create optimized pipeline with common settings
    
    Args:
        device_rate: Device sample rate
        input_gain: Input gain factor
        wake_word_gain: Additional gain for wake word detection
        
    Returns:
        Configured OptimizedAudioPipeline instance
    """
    config = AudioPipelineConfig(
        device_sample_rate=device_rate,
        openai_sample_rate=24000,
        porcupine_sample_rate=16000,
        input_gain=input_gain,
        wake_word_gain=wake_word_gain,
        use_polyphase=True,
        enable_soft_limiting=True
    )
    
    return OptimizedAudioPipeline(config)