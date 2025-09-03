"""
Pipeline stage capture module

Provides hooks to capture audio at each transformation stage in the pipeline
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any, Callable
from collections import deque
import time
from scipy import signal
import threading
from queue import Queue, Empty

from config import load_config, AudioConfig, WakeWordConfig
from audio.agc import AutomaticGainControl
from utils.logger import get_logger


class PipelineStageCapture:
    """
    Capture audio at specific pipeline stages
    
    This class provides hooks into the audio processing pipeline to capture
    audio data at each transformation stage for analysis.
    """
    
    def __init__(self, config: Any):
        """
        Initialize stage capture
        
        Args:
            config: Configuration object
        """
        self.logger = get_logger("PipelineStageCapture")
        self.config = config
        self.audio_config = config.audio
        
        # Initialize AGC for testing
        self.agc = AutomaticGainControl(self.audio_config, self.audio_config.sample_rate)
        
        # Storage for captured audio
        self.capture_buffers: Dict[str, deque] = {}
        self.capture_active: Dict[str, bool] = {}
        
        # Sample rates at different stages
        self.stage_sample_rates = {
            "raw_input": self.audio_config.sample_rate,
            "volume_adjusted": self.audio_config.sample_rate,
            "agc_processed": self.audio_config.sample_rate,
            "resampled_24k": 24000,
            "pcm16_converted": 24000,
            "wake_word_gain": 16000,  # Wake word uses 16kHz
            "highpass_filtered": 16000
        }
        
        self.logger.info("Pipeline stage capture initialized")
    
    async def capture(self, stage_name: str, duration: float) -> np.ndarray:
        """
        Capture audio at a specific pipeline stage
        
        Args:
            stage_name: Name of the pipeline stage
            duration: Capture duration in seconds
            
        Returns:
            Captured audio data
        """
        self.logger.info(f"Starting capture for stage: {stage_name}, duration: {duration}s")
        
        # Initialize capture buffer
        sample_rate = self.stage_sample_rates.get(stage_name, self.audio_config.sample_rate)
        buffer_size = int(sample_rate * duration)
        self.capture_buffers[stage_name] = deque(maxlen=buffer_size)
        self.capture_active[stage_name] = True
        
        # Start simulated audio capture
        await self._simulate_stage_capture(stage_name, duration)
        
        # Convert buffer to numpy array
        captured_audio = np.array(list(self.capture_buffers[stage_name]), dtype=np.float32)
        
        self.capture_active[stage_name] = False
        self.logger.info(f"Captured {len(captured_audio)} samples for stage: {stage_name}")
        
        return captured_audio
    
    async def _simulate_stage_capture(self, stage_name: str, duration: float) -> None:
        """
        Simulate audio capture for a specific stage
        
        This simulates the audio transformations that would occur at each stage
        of the pipeline. In a real implementation, this would hook into the
        actual audio processing pipeline.
        
        Args:
            stage_name: Name of the pipeline stage
            duration: Capture duration in seconds
        """
        sample_rate = self.stage_sample_rates.get(stage_name, self.audio_config.sample_rate)
        chunk_size = self.audio_config.chunk_size
        num_chunks = int((sample_rate * duration) / chunk_size)
        
        self.logger.debug(f"Simulating {num_chunks} chunks for {stage_name}")
        
        for chunk_idx in range(num_chunks):
            # Generate test audio (mix of tones and noise for testing)
            t = np.arange(chunk_size) / sample_rate + (chunk_idx * chunk_size / sample_rate)
            
            # Create a test signal with multiple components
            # 440Hz tone (A4 note) + 880Hz harmonic + some noise
            test_signal = (
                0.3 * np.sin(2 * np.pi * 440 * t) +  # Fundamental
                0.1 * np.sin(2 * np.pi * 880 * t) +  # 2nd harmonic
                0.05 * np.sin(2 * np.pi * 1320 * t) +  # 3rd harmonic
                0.02 * np.random.randn(chunk_size)  # Noise
            )
            
            # Apply transformations based on stage
            processed_audio = self._apply_stage_transformation(test_signal, stage_name)
            
            # Add to buffer
            self.capture_buffers[stage_name].extend(processed_audio)
            
            # Small delay to simulate real-time capture
            await asyncio.sleep(chunk_size / sample_rate)
    
    def _apply_stage_transformation(self, audio_data: np.ndarray, stage_name: str) -> np.ndarray:
        """
        Apply transformation for a specific pipeline stage
        
        Args:
            audio_data: Input audio data
            stage_name: Name of the pipeline stage
            
        Returns:
            Transformed audio data
        """
        if stage_name == "raw_input":
            # Stage 1: Raw input (no transformation)
            return audio_data
        
        elif stage_name == "volume_adjusted":
            # Stage 2: Apply input volume adjustment
            volume = self.audio_config.input_volume
            adjusted = audio_data * volume
            
            # Simulate clipping if volume too high
            if volume > 2.0:
                adjusted = np.clip(adjusted, -1.0, 1.0)
            
            return adjusted
        
        elif stage_name == "agc_processed":
            # Stage 3: Apply AGC processing
            # First apply volume adjustment
            volume_adjusted = audio_data * self.audio_config.input_volume
            
            # Then apply AGC
            if self.audio_config.agc_enabled:
                agc_processed = self.agc.process_audio(volume_adjusted)
            else:
                agc_processed = volume_adjusted
            
            return agc_processed
        
        elif stage_name == "resampled_24k":
            # Stage 4: Resample from device rate to 24kHz
            # First apply previous transformations
            processed = audio_data * self.audio_config.input_volume
            if self.audio_config.agc_enabled:
                processed = self.agc.process_audio(processed)
            
            # Resample if needed
            if self.audio_config.sample_rate != 24000:
                resampling_ratio = 24000 / self.audio_config.sample_rate
                new_length = int(len(processed) * resampling_ratio)
                resampled = signal.resample(processed, new_length)
            else:
                resampled = processed
            
            return resampled
        
        elif stage_name == "pcm16_converted":
            # Stage 5: Convert to PCM16 format
            # Apply all previous transformations
            processed = audio_data * self.audio_config.input_volume
            if self.audio_config.agc_enabled:
                processed = self.agc.process_audio(processed)
            
            # Resample
            if self.audio_config.sample_rate != 24000:
                resampling_ratio = 24000 / self.audio_config.sample_rate
                new_length = int(len(processed) * resampling_ratio)
                processed = signal.resample(processed, new_length)
            
            # Convert to PCM16
            processed = np.clip(processed, -1.0, 1.0)
            pcm16 = (processed * 32767).astype(np.int16)
            
            # Convert back to float for analysis
            return pcm16.astype(np.float32) / 32767
        
        elif stage_name == "wake_word_gain":
            # Stage 6: Apply wake word gain
            # Apply all previous transformations
            processed = audio_data * self.audio_config.input_volume
            if self.audio_config.agc_enabled:
                processed = self.agc.process_audio(processed)
            
            # Resample to 16kHz for wake word
            if self.audio_config.sample_rate != 16000:
                resampling_ratio = 16000 / self.audio_config.sample_rate
                new_length = int(len(processed) * resampling_ratio)
                processed = signal.resample(processed, new_length)
            
            # Apply wake word gain
            if hasattr(self.config, 'wake_word'):
                wake_gain = self.config.wake_word.audio_gain
            else:
                wake_gain = 1.0
            
            processed = processed * wake_gain
            
            # Simulate aggressive clipping if gain too high
            if wake_gain > 3.0:
                processed = np.clip(processed, -1.0, 1.0)
            
            return processed
        
        elif stage_name == "highpass_filtered":
            # Stage 7: Apply high-pass filter
            # Apply all previous transformations including wake word gain
            processed = audio_data * self.audio_config.input_volume
            if self.audio_config.agc_enabled:
                processed = self.agc.process_audio(processed)
            
            # Resample to 16kHz
            if self.audio_config.sample_rate != 16000:
                resampling_ratio = 16000 / self.audio_config.sample_rate
                new_length = int(len(processed) * resampling_ratio)
                processed = signal.resample(processed, new_length)
            
            # Apply wake word gain
            if hasattr(self.config, 'wake_word'):
                wake_gain = self.config.wake_word.audio_gain
                processed = processed * wake_gain
            
            # Apply high-pass filter
            if hasattr(self.config, 'wake_word') and hasattr(self.config.wake_word, 'highpass_filter_enabled'):
                if self.config.wake_word.highpass_filter_enabled:
                    cutoff = self.config.wake_word.highpass_filter_cutoff
                    nyquist = 16000 / 2
                    normalized_cutoff = cutoff / nyquist
                    
                    # Design filter
                    sos = signal.butter(4, normalized_cutoff, btype='highpass', output='sos')
                    
                    # Apply filter
                    processed = signal.sosfilt(sos, processed)
            
            return processed
        
        else:
            self.logger.warning(f"Unknown stage: {stage_name}")
            return audio_data
    
    def hook_into_pipeline(self, audio_capture_instance: Any) -> None:
        """
        Hook into the actual audio capture pipeline
        
        This would be used in a real implementation to intercept audio
        at various stages of the actual pipeline.
        
        Args:
            audio_capture_instance: Instance of AudioCapture class
        """
        # TODO: Implement actual hooks into the audio pipeline
        # This would involve modifying the AudioCapture class to support
        # stage callbacks or using monkey patching
        pass
    
    async def capture_parallel(self, stages: list, duration: float) -> Dict[str, np.ndarray]:
        """
        Capture multiple stages in parallel
        
        Args:
            stages: List of stage names to capture
            duration: Capture duration in seconds
            
        Returns:
            Dictionary of stage_name -> audio_data
        """
        self.logger.info(f"Starting parallel capture for stages: {stages}")
        
        # Create capture tasks
        tasks = [
            self.capture(stage, duration)
            for stage in stages
        ]
        
        # Run captures in parallel
        results = await asyncio.gather(*tasks)
        
        # Create results dictionary
        captured_data = {
            stage: audio
            for stage, audio in zip(stages, results)
        }
        
        self.logger.info(f"Parallel capture complete for {len(stages)} stages")
        
        return captured_data
    
    def analyze_stage_difference(self, stage1_audio: np.ndarray, 
                                stage2_audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze the difference between two pipeline stages
        
        Args:
            stage1_audio: Audio from first stage
            stage2_audio: Audio from second stage
            
        Returns:
            Dictionary of difference metrics
        """
        # Ensure same length for comparison
        min_len = min(len(stage1_audio), len(stage2_audio))
        audio1 = stage1_audio[:min_len]
        audio2 = stage2_audio[:min_len]
        
        # Calculate difference metrics
        metrics = {
            "gain_change_db": 20 * np.log10(
                np.sqrt(np.mean(audio2**2)) / (np.sqrt(np.mean(audio1**2)) + 1e-10)
            ),
            "peak_change_db": 20 * np.log10(
                np.max(np.abs(audio2)) / (np.max(np.abs(audio1)) + 1e-10)
            ),
            "correlation": np.corrcoef(audio1, audio2)[0, 1],
            "mse": np.mean((audio1 - audio2) ** 2),
            "snr_change": self._calculate_snr_change(audio1, audio2)
        }
        
        return metrics
    
    def _calculate_snr_change(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate SNR change between original and processed audio
        
        Args:
            original: Original audio
            processed: Processed audio
            
        Returns:
            SNR change in dB
        """
        # Estimate noise as difference between signals
        noise = processed - original
        
        # Calculate SNR
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return 60.0  # Very high SNR
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        return float(snr)