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
        self.audio_queue = Queue(maxsize=150)  # Large queue for Pi stability
        
        # Enhanced buffering for smooth playback - optimized for Raspberry Pi hardware
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_buffer_size = int(self.device_sample_rate * 0.20)  # 200ms buffer minimum (increased)
        self.target_buffer_size = int(self.device_sample_rate * 0.50)  # 500ms target buffer (increased)
        self.max_buffer_size = int(self.device_sample_rate * 2.0)  # 2s maximum buffer to handle large OpenAI chunks
        
        # Resampling
        self.need_resampling = self.device_sample_rate != self.source_sample_rate
        if self.need_resampling:
            self.resampling_ratio = self.device_sample_rate / self.source_sample_rate
            self.logger.info(f"Will resample from {self.source_sample_rate}Hz to {self.device_sample_rate}Hz")
        self.logger.info(f"Enhanced Pi buffer configuration: min={self.min_buffer_size} samples ({self.min_buffer_size/self.device_sample_rate*1000:.0f}ms), target={self.target_buffer_size} samples ({self.target_buffer_size/self.device_sample_rate*1000:.0f}ms), max={self.max_buffer_size} samples ({self.max_buffer_size/self.device_sample_rate*1000:.0f}ms)")
        
        # Threading
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Interruption support
        self.interrupt_requested = False
        
        # Playback smoothing
        self.underrun_count = 0
        self.last_underrun_warning = 0
        self.consecutive_underruns = 0
        self.max_consecutive_underruns = 5  # Force completion after 5 consecutive underruns
        
        # Audio completion tracking
        self.completion_callbacks = []
        self.is_response_active = False
        self.silence_frames_count = 0
        self.silence_threshold = int(self.device_sample_rate * 0.1)  # 100ms of silence
        self.last_audio_time = 0
        self.completion_timeout = 3.0  # 3 seconds timeout for completion detection
        self.completion_timeout_task = None
        self.max_response_duration = 30.0  # 30 seconds maximum response duration
    
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
            
            # Create audio stream optimized for Raspberry Pi
            stream_params = {
                'device': self.output_device if self.output_device != "default" else None,
                'samplerate': self.device_sample_rate,
                'channels': self.channels,
                'dtype': np.float32,
                'blocksize': self.chunk_size,
                'callback': self._audio_callback,
                'latency': 'high'  # Higher latency for Pi stability
            }
            
            self.logger.debug("Using ALSA defaults with high latency for Raspberry Pi stability")
            
            # Create stream with error handling
            try:
                self.stream = sd.OutputStream(**stream_params)
                self.logger.debug(f"Created audio stream with device: {stream_params['device']}")
            except Exception as e:
                self.logger.error(f"Failed to create audio stream: {e}")
                # Try fallback with default device
                if stream_params['device'] is not None:
                    self.logger.info("Trying fallback to default audio device")
                    stream_params['device'] = None
                    try:
                        self.stream = sd.OutputStream(**stream_params)
                        self.logger.info("Successfully created stream with default device")
                    except Exception as fallback_e:
                        self.logger.error(f"Fallback to default device also failed: {fallback_e}")
                        raise RuntimeError(f"Audio initialization failed: {e}. Fallback failed: {fallback_e}")
                else:
                    raise RuntimeError(f"Audio initialization failed: {e}")
            
            # Start stream with error handling
            try:
                self.stream.start()
                self.is_playing = True
                self.logger.debug("Audio stream started successfully")
            except Exception as e:
                self.logger.error(f"Failed to start audio stream: {e}")
                if self.stream:
                    self.stream.close()
                    self.stream = None
                raise RuntimeError(f"Failed to start audio stream: {e}")
            
            # Start playback thread
            try:
                self.stop_event.clear()
                self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
                self.playback_thread.start()
                self.logger.debug("Audio playback thread started")
            except Exception as e:
                self.logger.error(f"Failed to start playback thread: {e}")
                # Clean up stream
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                self.is_playing = False
                raise RuntimeError(f"Failed to start playback thread: {e}")
            
            self.logger.info(f"Raspberry Pi audio playback started successfully (device: {self.output_device}, rate: {self.device_sample_rate}Hz, latency: high)")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio playback: {e}")
            # Ensure cleanup on any failure
            self.is_playing = False
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
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
            
            # Update last audio time when we get new data
            if self.is_response_active:
                import time
                self.last_audio_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error queuing audio: {e}")
    
    def interrupt(self) -> None:
        """Interrupt current playback and clear queue"""
        self.interrupt_requested = True
        self.clear_queue()
        self.is_response_active = False
        self.silence_frames_count = 0
        self.logger.debug("Audio playback interrupted")
    
    def add_completion_callback(self, callback) -> None:
        """Add a callback to be called when audio playback completes"""
        self.completion_callbacks.append(callback)
    
    def remove_completion_callback(self, callback) -> None:
        """Remove a completion callback"""
        if callback in self.completion_callbacks:
            self.completion_callbacks.remove(callback)
    
    def start_response(self) -> None:
        """Mark the start of a response for completion tracking"""
        self.is_response_active = True
        self.silence_frames_count = 0
        import time
        self.last_audio_time = time.time()
        
        # Start timeout task
        self._start_completion_timeout()
        
        self.logger.debug("Started response audio tracking with timeout protection")
    
    def end_response(self) -> None:
        """Mark the end of a response (OpenAI finished sending)"""
        self.logger.debug("OpenAI finished sending audio - monitoring for completion")
        # Don't set is_response_active to False here - let completion detection handle it
    
    def _notify_completion(self) -> None:
        """Notify all callbacks that audio playback has completed"""
        if self.is_response_active:
            self.logger.info("Audio playback completed - notifying callbacks")
            self.is_response_active = False
            
            # Cancel timeout task
            self._cancel_completion_timeout()
            
            # Notify all callbacks with comprehensive error handling
            for i, callback in enumerate(self.completion_callbacks):
                try:
                    self.logger.debug(f"Calling completion callback #{i+1}")
                    callback()
                    self.logger.debug(f"Completion callback #{i+1} completed successfully")
                except Exception as e:
                    self.logger.error(f"Error in completion callback #{i+1}: {e}")
                    # Continue with other callbacks even if one fails
                    continue
    
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
        
        # If we clear the queue, we're probably stopping playback
        if self.is_response_active:
            self.is_response_active = False
            self.silence_frames_count = 0
            self._cancel_completion_timeout()
    
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
                
                # Reset consecutive underrun count when we have enough audio
                self.consecutive_underruns = 0
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
                
                # Count silence frames for completion detection
                silence_frames = frames - available_frames
                self.silence_frames_count += silence_frames
                self.consecutive_underruns += 1  # Track consecutive underruns
                
                # If we have too many underruns, this may indicate completion
                if self.underrun_count >= 3 and self.audio_queue.empty():
                    self.logger.debug(f"Multiple underruns detected ({self.underrun_count}) with empty queue - may indicate completion")
                    self._check_completion()
                
                # If we have excessive underruns, force completion to prevent hanging
                if self.underrun_count >= 10:
                    self.logger.warning(f"Excessive underruns detected ({self.underrun_count}) - forcing completion to prevent hanging")
                    print(f"*** EXCESSIVE UNDERRUNS ({self.underrun_count}) - FORCING COMPLETION ***")
                    self._notify_completion()
                    return
            else:
                # No audio available - output silence
                # This is normal when audio stream first starts or between utterances
                self.silence_frames_count += frames
                
                # Track consecutive underruns when no audio available
                if self.is_response_active:
                    self.consecutive_underruns += 1
                    
                    # Force completion if too many consecutive underruns
                    if self.consecutive_underruns >= self.max_consecutive_underruns:
                        self.logger.warning(f"Too many consecutive underruns ({self.consecutive_underruns}) - forcing completion")
                        print(f"*** TOO MANY CONSECUTIVE UNDERRUNS ({self.consecutive_underruns}) - FORCING COMPLETION ***")
                        self._notify_completion()
                        return
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            # Force completion on callback error to prevent hanging
            if self.is_response_active:
                self.logger.warning("Audio callback error - forcing completion to prevent hanging")
                try:
                    self._notify_completion()
                except:
                    pass  # Ignore errors in error handler
    
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
        """Fill the audio buffer from the queue for smooth playback with enhanced underrun prevention"""
        try:
            # Enhanced buffer filling strategy
            buffer_threshold = self.target_buffer_size
            
            # If buffer is critically low, be more aggressive
            if len(self.audio_buffer) < self.min_buffer_size:
                buffer_threshold = self.target_buffer_size * 1.5  # 1.5x target when low
            
            # Check if we're approaching buffer overflow
            if len(self.audio_buffer) >= self.max_buffer_size:
                self.logger.warning(f"Audio buffer approaching maximum size ({len(self.audio_buffer)}/{self.max_buffer_size})")
                # Don't add more if we're at maximum to prevent memory issues
                return
            
            chunks_added = 0
            while len(self.audio_buffer) < buffer_threshold and not self.audio_queue.empty():
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    
                    # Check if adding this chunk would exceed maximum buffer size
                    if len(self.audio_buffer) + len(audio_chunk) > self.max_buffer_size:
                        # Only add part of the chunk to stay within limits
                        remaining_space = self.max_buffer_size - len(self.audio_buffer)
                        if remaining_space > 0:
                            audio_chunk = audio_chunk[:remaining_space]
                            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                            self.logger.warning(f"Truncated audio chunk to fit buffer limit (remaining space: {remaining_space})")
                        break
                    
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                    chunks_added += 1
                    
                    # Update last audio time when we get new data
                    if self.is_response_active:
                        import time
                        self.last_audio_time = time.time()
                        
                    # Prevent infinite loops
                    if chunks_added > 50:  # Safety limit
                        break
                        
                except Empty:
                    break
                    
            # Enhanced logging for buffer status
            if len(self.audio_buffer) < self.min_buffer_size and chunks_added == 0:
                self.logger.debug(f"Buffer critically low: {len(self.audio_buffer)} samples ({len(self.audio_buffer)/self.device_sample_rate*1000:.1f}ms), queue empty: {self.audio_queue.empty()}")
            elif chunks_added > 0:
                self.logger.debug(f"Added {chunks_added} chunks, buffer now: {len(self.audio_buffer)} samples ({len(self.audio_buffer)/self.device_sample_rate*1000:.1f}ms)")
                
        except Exception as e:
            self.logger.error(f"Error filling buffer from queue: {e}")
            # Try to recover by clearing potentially corrupted buffer
            if len(self.audio_buffer) == 0:
                self.audio_buffer = np.array([], dtype=np.float32)
    
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
    
    def _start_completion_timeout(self) -> None:
        """Start timeout task for audio completion detection"""
        import asyncio
        
        try:
            # Cancel any existing timeout task
            self._cancel_completion_timeout()
            
            # Create new timeout task
            loop = asyncio.get_event_loop()
            self.completion_timeout_task = loop.create_task(self._completion_timeout_handler())
            
        except Exception as e:
            self.logger.error(f"Error starting completion timeout: {e}")
    
    def _cancel_completion_timeout(self) -> None:
        """Cancel the completion timeout task"""
        if self.completion_timeout_task and not self.completion_timeout_task.done():
            self.completion_timeout_task.cancel()
            self.completion_timeout_task = None
    
    async def _completion_timeout_handler(self) -> None:
        """Handle timeout for audio completion detection"""
        try:
            await asyncio.sleep(self.max_response_duration)
            
            # If we reach here, audio completion was not detected within timeout
            if self.is_response_active:
                self.logger.error(f"Audio completion timeout after {self.max_response_duration}s - forcing completion")
                print(f"*** AUDIO COMPLETION TIMEOUT AFTER {self.max_response_duration}S - FORCING COMPLETION ***")
                
                # Force completion
                self._notify_completion()
                
        except asyncio.CancelledError:
            # Task was cancelled (normal completion)
            pass
        except Exception as e:
            self.logger.error(f"Error in completion timeout handler: {e}")
    
    def _check_completion(self) -> None:
        """Check if audio playback has completed with comprehensive error handling"""
        if not self.is_response_active:
            return
        
        try:
            # Check if we have enough silence to consider playback complete
            if (self.silence_frames_count >= self.silence_threshold and 
                self.audio_queue.empty() and 
                len(self.audio_buffer) == 0):
                self.logger.debug(f"Audio completion detected via silence ({self.silence_frames_count} frames)")
                self._notify_completion()
                return
                
            # Timeout-based completion detection as fallback
            import time
            current_time = time.time()
            if (current_time - self.last_audio_time > self.completion_timeout and 
                self.audio_queue.empty() and 
                len(self.audio_buffer) == 0):
                self.logger.debug(f"Audio completion detected via timeout ({self.completion_timeout}s)")
                self._notify_completion()
                return
            
            # Check for stuck state (no audio activity for too long)
            if current_time - self.last_audio_time > self.max_response_duration * 0.5:
                self.logger.warning(f"No audio activity for {current_time - self.last_audio_time:.1f}s - may be stuck")
                
        except Exception as e:
            self.logger.error(f"Error in completion check: {e}")
            # Force completion on error to prevent hanging
            try:
                self._notify_completion()
            except:
                pass
    
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