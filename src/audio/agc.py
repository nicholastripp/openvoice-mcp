"""
Automatic Gain Control (AGC) for audio input

This module provides automatic gain adjustment to maintain optimal audio levels,
preventing clipping while ensuring adequate volume for wake word detection.
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import time
from collections import deque

from config import AudioConfig
from utils.logger import get_logger


@dataclass
class AGCStats:
    """Statistics for AGC monitoring"""
    current_gain: float
    last_rms: float
    last_peak: float
    last_clipping_ratio: float
    total_adjustments: int
    clipping_events: int
    

class AutomaticGainControl:
    """
    Automatic Gain Control for audio input
    
    Automatically adjusts gain to:
    - Prevent clipping distortion
    - Maintain optimal audio levels
    - Smooth gain transitions
    """
    
    def __init__(self, config: AudioConfig, sample_rate: int = 48000):
        """
        Initialize AGC
        
        Args:
            config: Audio configuration with AGC settings
            sample_rate: Audio sample rate for timing calculations
        """
        self._logger = None  # Lazy initialization
        
        # AGC settings
        self.enabled = config.agc_enabled
        self.target_rms = config.agc_target_rms
        self.max_gain = config.agc_max_gain
        self.min_gain = config.agc_min_gain
        self.attack_time = config.agc_attack_time
        self.release_time = config.agc_release_time
        self.clipping_threshold = config.agc_clipping_threshold
        
        # Current state
        self.current_gain = config.input_volume  # Start with manual setting
        self.sample_rate = sample_rate
        
        # History tracking
        self.gain_history = deque(maxlen=100)  # Last 100 gain values
        self.rms_history = deque(maxlen=50)    # Last 50 RMS values
        self.clipping_history = deque(maxlen=50)  # Last 50 clipping measurements
        
        # Statistics
        self.stats = AGCStats(
            current_gain=self.current_gain,
            last_rms=0.0,
            last_peak=0.0,
            last_clipping_ratio=0.0,
            total_adjustments=0,
            clipping_events=0
        )
        
        # Timing
        self.last_update_time = time.time()
        self.frames_processed = 0
        
        # Log initialization message when logger is available
        self._log_init_message = True
        
    @property
    def logger(self):
        """Lazy logger initialization"""
        if self._logger is None:
            self._logger = get_logger("AGC")
            # Log initialization message on first access
            if hasattr(self, '_log_init_message') and self._log_init_message:
                if self.enabled:
                    self._logger.info(f"AGC initialized - target_rms: {self.target_rms}, "
                                   f"gain_range: [{self.min_gain}, {self.max_gain}], "
                                   f"clipping_threshold: {self.clipping_threshold:.1%}")
                else:
                    self._logger.info("AGC disabled - using fixed gain")
                self._log_init_message = False
        return self._logger
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio with automatic gain control
        
        Args:
            audio_data: Input audio data (float32, -1 to 1 range)
            
        Returns:
            Gain-adjusted audio data
        """
        if not self.enabled:
            # Simple fixed gain when AGC is disabled
            return audio_data * self.current_gain
        
        # Analyze audio characteristics
        rms, peak, clipping_ratio = self._analyze_audio(audio_data)
        
        # Update statistics
        self.stats.last_rms = rms
        self.stats.last_peak = peak
        self.stats.last_clipping_ratio = clipping_ratio
        
        # Calculate time since last update for smooth transitions
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Determine target gain
        target_gain = self._calculate_target_gain(rms, peak, clipping_ratio)
        
        # Smooth gain transition
        self.current_gain = self._smooth_gain_transition(target_gain, time_delta)
        
        # Apply gain with soft limiting
        output = self._apply_gain_with_limiting(audio_data, self.current_gain)
        
        # Update history
        self.gain_history.append(self.current_gain)
        self.rms_history.append(rms)
        self.clipping_history.append(clipping_ratio)
        self.frames_processed += 1
        
        # Log status periodically
        if self.frames_processed % 100 == 0:
            self._log_status()
        
        return output
    
    def _analyze_audio(self, audio_data: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze audio characteristics
        
        Returns:
            (rms, peak, clipping_ratio)
        """
        # RMS (Root Mean Square) - average power
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # Peak - maximum absolute value
        peak = np.max(np.abs(audio_data))
        
        # Clipping ratio - proportion of samples near maximum
        clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
        
        return rms, peak, clipping_ratio
    
    def _calculate_target_gain(self, rms: float, peak: float, clipping_ratio: float) -> float:
        """
        Calculate the target gain based on audio analysis
        """
        current = self.current_gain
        
        # Priority 1: Prevent clipping
        if clipping_ratio > self.clipping_threshold:
            # Reduce gain aggressively
            reduction_factor = 0.7  # 30% reduction
            if clipping_ratio > 0.1:  # Severe clipping
                reduction_factor = 0.5  # 50% reduction
            
            target = current * reduction_factor
            self.stats.clipping_events += 1
            self.logger.warning(f"Clipping detected ({clipping_ratio:.1%}) - reducing gain to {target:.2f}")
            
        # Priority 2: Check if we're too loud (even without clipping)
        elif peak > 0.9:
            # Mild reduction to create headroom
            target = current * 0.9
            
        # Priority 3: Check if we're too quiet
        elif rms < self.target_rms * 0.5 and peak < 0.7:
            # Increase gain slowly
            increase_factor = 1.1  # 10% increase
            if rms < self.target_rms * 0.2:  # Very quiet
                increase_factor = 1.2  # 20% increase
            
            target = current * increase_factor
            
        # Priority 4: Fine-tune to target RMS
        elif abs(rms - self.target_rms) > 0.05:
            # Adjust towards target RMS
            ratio = self.target_rms / max(rms, 0.001)  # Avoid division by zero
            # Limit adjustment speed
            ratio = np.clip(ratio, 0.9, 1.1)
            target = current * ratio
            
        else:
            # No adjustment needed
            target = current
        
        # Apply constraints
        target = np.clip(target, self.min_gain, self.max_gain)
        
        # Track adjustments
        if abs(target - current) > 0.01:
            self.stats.total_adjustments += 1
        
        return target
    
    def _smooth_gain_transition(self, target_gain: float, time_delta: float) -> float:
        """
        Smoothly transition to target gain using exponential smoothing
        """
        if target_gain == self.current_gain:
            return self.current_gain
        
        # Determine time constant based on direction
        if target_gain < self.current_gain:
            # Attack: Fast reduction (prevent clipping)
            time_constant = self.attack_time
        else:
            # Release: Slow increase (prevent pumping)
            time_constant = self.release_time
        
        # Calculate smoothing factor (exponential decay)
        alpha = 1.0 - np.exp(-time_delta / time_constant)
        
        # Apply smoothing
        new_gain = self.current_gain + alpha * (target_gain - self.current_gain)
        
        return new_gain
    
    def _apply_gain_with_limiting(self, audio_data: np.ndarray, gain: float) -> np.ndarray:
        """
        Apply gain with soft limiting to prevent harsh clipping
        """
        # Apply gain
        output = audio_data * gain
        
        # Soft limiting using tanh compression for values > 0.9
        threshold = 0.9
        mask = np.abs(output) > threshold
        
        if np.any(mask):
            # Apply soft limiting to values above threshold
            limited = np.sign(output[mask]) * (
                threshold + (1 - threshold) * np.tanh(
                    (np.abs(output[mask]) - threshold) / (1 - threshold)
                )
            )
            output[mask] = limited
        
        return output
    
    def _log_status(self):
        """Log AGC status periodically"""
        avg_rms = np.mean(self.rms_history) if self.rms_history else 0
        avg_clipping = np.mean(self.clipping_history) if self.clipping_history else 0
        
        self.logger.info(
            f"AGC Status - gain: {self.current_gain:.2f}, "
            f"avg_rms: {avg_rms:.3f}, "
            f"avg_clipping: {avg_clipping:.1%}, "
            f"adjustments: {self.stats.total_adjustments}"
        )
    
    def get_stats(self) -> AGCStats:
        """Get current AGC statistics"""
        self.stats.current_gain = self.current_gain
        return self.stats
    
    def reset(self):
        """Reset AGC to initial state"""
        self.current_gain = 1.0
        self.gain_history.clear()
        self.rms_history.clear()
        self.clipping_history.clear()
        self.stats = AGCStats(
            current_gain=self.current_gain,
            last_rms=0.0,
            last_peak=0.0,
            last_clipping_ratio=0.0,
            total_adjustments=0,
            clipping_events=0
        )
        self.logger.info("AGC reset to initial state")