"""
Audio capture for voice input
"""
import asyncio
import numpy as np
import sounddevice as sd
from scipy import signal
from typing import Optional, Callable, Any
from queue import Queue, Empty
import threading

from config import AudioConfig
from utils.logger import get_logger


class AudioCapture:
    """
    Audio capture from microphone with real-time processing
    Handles resampling from device rate to OpenAI's required 24kHz
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = get_logger("AudioCapture")
        
        # Audio parameters
        self.device_sample_rate = config.sample_rate
        self.target_sample_rate = 24000  # OpenAI Realtime API requirement
        self.channels = config.channels
        self.chunk_size = config.chunk_size
        self.input_device = config.input_device
        self.volume = config.input_volume
        
        # State
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue = Queue()
        self.callback_handlers = []
        
        # Resampling
        self.need_resampling = self.device_sample_rate != self.target_sample_rate
        if self.need_resampling:
            self.resampling_ratio = self.target_sample_rate / self.device_sample_rate
            self.logger.info(f"Will resample from {self.device_sample_rate}Hz to {self.target_sample_rate}Hz")
        
        # Threading and async handling
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def start(self) -> None:
        """Start audio capture"""
        if self.is_recording:
            self.logger.warning("Audio capture already started")
            return
        
        try:
            # Store the current event loop for async callback handling
            self.event_loop = asyncio.get_event_loop()
            
            # Get device info
            device_info = self._get_device_info()
            if device_info:
                self.logger.info(f"Using input device: {device_info['name']}")
            
            # Create audio stream optimized for Raspberry Pi
            stream_params = {
                'device': self.input_device if self.input_device != "default" else None,
                'samplerate': self.device_sample_rate,
                'channels': self.channels,
                'dtype': np.float32,
                'blocksize': self.chunk_size,
                'callback': self._audio_callback,
                'latency': 'high'  # Higher latency for Pi stability
            }
            
            # Create stream with error handling
            try:
                self.stream = sd.InputStream(**stream_params)
                self.logger.debug(f"Created audio input stream with device: {stream_params['device']}")
            except Exception as e:
                self.logger.error(f"Failed to create audio input stream: {e}")
                # Try fallback with default device
                if stream_params['device'] is not None:
                    self.logger.info("Trying fallback to default audio input device")
                    stream_params['device'] = None
                    try:
                        self.stream = sd.InputStream(**stream_params)
                        self.logger.info("Successfully created input stream with default device")
                    except Exception as fallback_e:
                        self.logger.error(f"Fallback to default input device also failed: {fallback_e}")
                        raise RuntimeError(f"Audio input initialization failed: {e}. Fallback failed: {fallback_e}")
                else:
                    raise RuntimeError(f"Audio input initialization failed: {e}")
            
            # Start stream with error handling
            try:
                self.stream.start()
                self.is_recording = True
                self.logger.debug("Audio input stream started successfully")
            except Exception as e:
                self.logger.error(f"Failed to start audio input stream: {e}")
                if self.stream:
                    self.stream.close()
                    self.stream = None
                raise RuntimeError(f"Failed to start audio input stream: {e}")
            
            # Start processing thread
            try:
                self.stop_event.clear()
                self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
                self.processing_thread.start()
                self.logger.debug("Audio capture processing thread started")
            except Exception as e:
                self.logger.error(f"Failed to start capture processing thread: {e}")
                # Clean up stream
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                self.is_recording = False
                raise RuntimeError(f"Failed to start capture processing thread: {e}")
            
            self.logger.info(f"Raspberry Pi audio capture started successfully (device: {self.input_device}, rate: {self.device_sample_rate}Hz, latency: high)")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            # Ensure cleanup on any failure
            self.is_recording = False
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
            raise
    
    async def stop(self) -> None:
        """Stop audio capture"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Stop processing thread
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("Audio capture stopped")
    
    def add_callback(self, callback: Callable[[bytes], None]) -> None:
        """
        Add callback for processed audio data
        
        Args:
            callback: Function to call with audio data (PCM16, 24kHz, mono)
        """
        self.callback_handlers.append(callback)
    
    def remove_callback(self, callback: Callable[[bytes], None]) -> None:
        """Remove audio callback"""
        if callback in self.callback_handlers:
            self.callback_handlers.remove(callback)
    
    def get_volume_level(self) -> float:
        """
        Get current audio input level (0.0 to 1.0)
        
        Returns:
            Current volume level
        """
        # This would be implemented with a rolling buffer of recent audio
        # For now, return a placeholder
        return 0.0
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        """
        Audio stream callback - called by sounddevice on each audio chunk
        
        Args:
            indata: Audio input data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        if not self.is_recording:
            return
        
        try:
            # Copy data to prevent issues with the original buffer
            audio_data = indata.copy()
            
            # Apply volume adjustment
            if self.volume != 1.0:
                audio_data *= self.volume
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Add to processing queue
            self.audio_queue.put(audio_data, block=False)
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
    
    def _processing_loop(self) -> None:
        """Background thread for processing audio data"""
        self.logger.debug("Audio processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue (with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process the audio
                processed_audio = self._process_audio_chunk(audio_data)
                
                # Send to callbacks (handle both sync and async)
                for callback in self.callback_handlers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            # Handle async callback
                            if self.event_loop:
                                future = asyncio.run_coroutine_threadsafe(callback(processed_audio), self.event_loop)
                                # Don't wait for completion to avoid blocking audio processing
                            else:
                                self.logger.warning("Async callback registered but no event loop available")
                        else:
                            # Handle sync callback
                            callback(processed_audio)
                    except Exception as e:
                        self.logger.error(f"Error in audio callback: {e}")
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
        
        self.logger.debug("Audio processing thread stopped")
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> bytes:
        """
        Process audio chunk: resample and convert to PCM16
        
        Args:
            audio_data: Float32 audio data at device sample rate
            
        Returns:
            PCM16 audio data at 24kHz
        """
        # Apply input volume gain
        if self.volume != 1.0:
            audio_data = audio_data * self.volume
            # Log if clipping occurs
            if np.any(np.abs(audio_data) > 1.0):
                clipped_samples = np.sum(np.abs(audio_data) > 1.0)
                self.logger.debug(f"Audio clipping detected: {clipped_samples} samples clipped with gain {self.volume}")
        
        # Resample if needed
        if self.need_resampling:
            # Calculate new length
            new_length = int(len(audio_data) * self.resampling_ratio)
            
            # Resample using scipy
            resampled = signal.resample(audio_data, new_length)
        else:
            resampled = audio_data
        
        # Convert to PCM16
        # Clamp to [-1, 1] range
        resampled = np.clip(resampled, -1.0, 1.0)
        
        # Convert to int16
        pcm16_data = (resampled * 32767).astype(np.int16)
        
        return pcm16_data.tobytes()
    
    def _get_device_info(self) -> Optional[dict]:
        """Get information about the selected input device"""
        try:
            if self.input_device == "default":
                device_id = sd.default.device[0]  # Input device
            else:
                # Try to find device by name or use as index
                if isinstance(self.input_device, str) and not self.input_device.isdigit():
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if self.input_device.lower() in device['name'].lower():
                            device_id = i
                            break
                    else:
                        self.logger.warning(f"Device '{self.input_device}' not found, using default")
                        device_id = sd.default.device[0]
                else:
                    device_id = int(self.input_device)
            
            device_info = sd.query_devices(device_id)
            return device_info
            
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
            return None
    
    @staticmethod
    def list_devices() -> list:
        """
        List available audio input devices
        
        Returns:
            List of device information dictionaries
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return input_devices
            
        except Exception as e:
            logger = get_logger("AudioCapture")
            logger.error(f"Error listing devices: {e}")
            return []
    
    @staticmethod
    def test_device(device_id: Any, duration: float = 2.0) -> bool:
        """
        Test if a device works for recording
        
        Args:
            device_id: Device index or name
            duration: Test duration in seconds
            
        Returns:
            True if device works, False otherwise
        """
        logger = get_logger("AudioCapture")
        
        try:
            # Record for a short duration
            recording = sd.rec(
                frames=int(44100 * duration),
                samplerate=44100,
                channels=1,
                device=device_id if device_id != "default" else None,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Check if we got actual audio data
            if recording is not None and len(recording) > 0:
                max_amplitude = np.max(np.abs(recording))
                logger.info(f"Device test successful, max amplitude: {max_amplitude:.4f}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Device test failed: {e}")
            return False