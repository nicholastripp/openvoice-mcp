"""
Audio playback for voice output
"""
import asyncio
import numpy as np
import sounddevice as sd
from scipy import signal
from typing import Optional, Any
from queue import Queue, Empty
import threading

from config import AudioConfig
from utils.logger import get_logger


class AudioPlayback:
    """
    Audio playback for voice output
    Handles resampling from OpenAI's 24kHz to device rate and queuing
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = get_logger("AudioPlayback")
        
        # Audio parameters
        self.device_sample_rate = config.sample_rate
        self.source_sample_rate = 24000  # OpenAI Realtime API output
        self.channels = config.channels
        self.chunk_size = config.chunk_size
        self.output_device = config.output_device
        self.volume = config.output_volume
        
        # State
        self.is_playing = False
        self.stream: Optional[sd.OutputStream] = None
        self.audio_queue = Queue(maxsize=100)  # Increased queue size to prevent underruns
        
        # Buffering for smooth playback - optimized for OpenAI's variable chunk sizes
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_buffer_size = int(self.device_sample_rate * 0.1)  # 100ms buffer minimum (increased)
        self.target_buffer_size = int(self.device_sample_rate * 0.25)  # 250ms target buffer (increased)
        
        # Resampling
        self.need_resampling = self.device_sample_rate != self.source_sample_rate
        if self.need_resampling:
            self.resampling_ratio = self.device_sample_rate / self.source_sample_rate
            self.logger.info(f"Will resample from {self.source_sample_rate}Hz to {self.device_sample_rate}Hz")
        
        # Threading
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Interruption support
        self.interrupt_requested = False
        
        # Playback smoothing
        self.underrun_count = 0
        self.last_underrun_warning = 0
    
    async def start(self) -> None:
        """Start audio playback system"""
        if self.is_playing:
            self.logger.warning("Audio playback already started")
            return
        
        try:
            # Get device info
            device_info = self._get_device_info()
            if device_info:
                self.logger.info(f"Using output device: {device_info['name']}")
            
            # Create audio stream with increased buffer size
            self.stream = sd.OutputStream(
                device=self.output_device if self.output_device != "default" else None,
                samplerate=self.device_sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                latency='low',
                extra_settings=sd.CoreAudioSettings(channel_map=None) if hasattr(sd, 'CoreAudioSettings') else None
            )
            
            # Start stream
            self.stream.start()
            self.is_playing = True
            
            # Start playback thread
            self.stop_event.clear()
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            self.logger.info("Audio playback started")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio playback: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop audio playback system"""
        if not self.is_playing:
            return
        
        self.is_playing = False
        
        # Stop playback thread
        self.stop_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Clear queue
        self.clear_queue()
        
        self.logger.info("Audio playback stopped")
    
    def play_audio(self, audio_data: bytes) -> None:
        """
        Queue audio data for playback
        
        Args:
            audio_data: PCM16 audio data at 24kHz
        """
        if not self.is_playing:
            self.logger.warning("Cannot play audio: playback not started")
            return
        
        try:
            # Convert PCM16 bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Process and queue
            processed_audio = self._process_audio_chunk(audio_float)
            self.audio_queue.put(processed_audio, block=False)
            
        except Exception as e:
            self.logger.error(f"Error queuing audio: {e}")
    
    def interrupt(self) -> None:
        """Interrupt current playback and clear queue"""
        self.interrupt_requested = True
        self.clear_queue()
        self.logger.debug("Audio playback interrupted")
    
    def clear_queue(self) -> None:
        """Clear the audio playback queue and buffer"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        # Also clear the internal buffer for clean start
        self.audio_buffer = np.array([], dtype=np.float32)
        self.underrun_count = 0
    
    def is_queue_empty(self) -> bool:
        """Check if playback queue is empty"""
        return self.audio_queue.empty()
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.audio_queue.qsize()
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        """
        Audio stream callback - called by sounddevice to fill output buffer
        
        Args:
            outdata: Output buffer to fill
            frames: Number of frames to write
            time: Time info
            status: Status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Fill with silence by default
        outdata.fill(0)
        
        if not self.is_playing:
            return
        
        # Check for interruption
        if self.interrupt_requested:
            self.interrupt_requested = False
            self.audio_buffer = np.array([], dtype=np.float32)  # Clear buffer on interrupt
            return
        
        try:
            # Fill buffer from queue if needed
            self._fill_buffer_from_queue()
            
            # Check if we have enough buffered audio
            if len(self.audio_buffer) >= frames:
                # Use buffered audio
                outdata[:, 0] = self.audio_buffer[:frames]
                self.audio_buffer = self.audio_buffer[frames:]
            elif len(self.audio_buffer) > 0:
                # Use what we have and pad with silence
                available_frames = len(self.audio_buffer)
                outdata[:available_frames, 0] = self.audio_buffer
                outdata[available_frames:, 0] = 0  # Silence for remaining frames
                self.audio_buffer = np.array([], dtype=np.float32)
                
                # Track underruns for debugging
                self.underrun_count += 1
                import time
                current_time = time.time()
                if current_time - self.last_underrun_warning > 3.0:  # Log every 3 seconds max
                    self.logger.warning(f"Audio underrun #{self.underrun_count}: needed {frames}, had {available_frames}, buffer_size={len(self.audio_buffer)}")
                    self.last_underrun_warning = current_time
            else:
                # No audio available - output silence
                # This is normal when audio stream first starts or between utterances
                pass
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
    
    def _playback_loop(self) -> None:
        """Background thread for audio processing (if needed for future features)"""
        self.logger.debug("Audio playback thread started")
        
        while not self.stop_event.is_set():
            # Currently just sleeping, but could be used for:
            # - Audio preprocessing
            # - Volume envelope processing
            # - Effects processing
            # - Queue management
            try:
                self.stop_event.wait(timeout=0.1)
            except Exception as e:
                self.logger.error(f"Error in playback loop: {e}")
        
        self.logger.debug("Audio playback thread stopped")
    
    def _fill_buffer_from_queue(self) -> None:
        """Fill the audio buffer from the queue for smooth playback"""
        try:
            # More aggressive buffer filling for variable chunk sizes
            # Fill buffer more aggressively when low, but respect target when adequate
            buffer_threshold = self.min_buffer_size if len(self.audio_buffer) < self.min_buffer_size else self.target_buffer_size
            
            while len(self.audio_buffer) < buffer_threshold and not self.audio_queue.empty():
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                except Empty:
                    break
        except Exception as e:
            self.logger.error(f"Error filling buffer from queue: {e}")
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio chunk: resample and apply volume
        
        Args:
            audio_data: Float32 audio data at 24kHz
            
        Returns:
            Float32 audio data at device sample rate
        """
        # Resample if needed
        if self.need_resampling:
            # Calculate new length
            new_length = int(len(audio_data) * self.resampling_ratio)
            
            # Resample using scipy
            resampled = signal.resample(audio_data, new_length)
        else:
            resampled = audio_data
        
        # Apply volume
        if self.volume != 1.0:
            resampled *= self.volume
        
        # Clamp to valid range
        resampled = np.clip(resampled, -1.0, 1.0)
        
        return resampled
    
    def _get_device_info(self) -> Optional[dict]:
        """Get information about the selected output device"""
        try:
            if self.output_device == "default":
                device_id = sd.default.device[1]  # Output device
            else:
                # Try to find device by name or use as index
                if isinstance(self.output_device, str) and not self.output_device.isdigit():
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if self.output_device.lower() in device['name'].lower():
                            device_id = i
                            break
                    else:
                        self.logger.warning(f"Device '{self.output_device}' not found, using default")
                        device_id = sd.default.device[1]
                else:
                    device_id = int(self.output_device)
            
            device_info = sd.query_devices(device_id)
            return device_info
            
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
            return None
    
    @staticmethod
    def list_devices() -> list:
        """
        List available audio output devices
        
        Returns:
            List of device information dictionaries
        """
        try:
            devices = sd.query_devices()
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    output_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_output_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return output_devices
            
        except Exception as e:
            logger = get_logger("AudioPlayback")
            logger.error(f"Error listing devices: {e}")
            return []
    
    @staticmethod
    def test_device(device_id: Any, duration: float = 1.0, frequency: float = 440.0) -> bool:
        """
        Test if a device works for playback
        
        Args:
            device_id: Device index or name
            duration: Test duration in seconds
            frequency: Test tone frequency in Hz
            
        Returns:
            True if device works, False otherwise
        """
        logger = get_logger("AudioPlayback")
        
        try:
            # Generate test tone
            sample_rate = 44100
            frames = int(sample_rate * duration)
            t = np.linspace(0, duration, frames, False)
            tone = 0.1 * np.sin(2 * np.pi * frequency * t)  # Low volume
            
            # Play test tone
            sd.play(
                tone,
                samplerate=sample_rate,
                device=device_id if device_id != "default" else None
            )
            sd.wait()  # Wait for playback to complete
            
            logger.info("Device test successful")
            return True
            
        except Exception as e:
            logger.error(f"Device test failed: {e}")
            return False