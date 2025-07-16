"""
Picovoice Porcupine wake word detection implementation
"""
import asyncio
import numpy as np
import threading
from typing import Callable, Optional, List, Dict
from queue import Queue, Empty
import time
import os
import concurrent.futures
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config import WakeWordConfig
from utils.logger import get_logger

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    pvporcupine = None


class PorcupineDetector:
    """
    Wake word detection using Picovoice Porcupine
    """
    
    # Mapping from config names to Porcupine built-in keywords
    # Based on Porcupine documentation and verified built-in keywords
    KEYWORD_MAPPING = {
        # Known working single-word keywords
        'alexa': 'alexa',
        'picovoice': 'picovoice',
        'americano': 'americano',
        'blueberry': 'blueberry',
        'bumblebee': 'bumblebee',
        'grapefruit': 'grapefruit',
        'grasshopper': 'grasshopper',
        'porcupine': 'porcupine',
        'terminator': 'terminator',
        'computer': 'computer',
        
        # Aliases for picovoice
        'hey_picovoice': 'picovoice',
        'ok_picovoice': 'picovoice',
        
        # NOTE: Multi-word wake words like 'hey google', 'ok google', 'hey siri'
        # may require special handling or might not be available as built-in keywords
        # 
        # NOTE: 'jarvis' and 'hey_jarvis' are NOT built-in Porcupine keywords
        # They require custom wake word creation at https://console.picovoice.ai/
    }
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.logger = get_logger("PorcupineDetector")
        
        if not PORCUPINE_AVAILABLE:
            raise ImportError("Porcupine not available. Install with: pip install pvporcupine")
        
        # Get access key from environment or config
        self.access_key = os.getenv('PICOVOICE_ACCESS_KEY', getattr(config, 'porcupine_access_key', None))
        if not self.access_key:
            raise ValueError("Picovoice access key not found. Set PICOVOICE_ACCESS_KEY environment variable or porcupine_access_key in config")
        
        # Audio parameters (Porcupine requirements)
        self.sample_rate = 16000  # Porcupine requires 16kHz
        self.frame_length = 512   # Porcupine frame size
        
        # State
        self.is_running = False
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self.audio_queue = Queue()
        self.detection_callbacks = []
        
        # Audio buffer for accumulating samples
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Detection parameters - validate wake word early
        try:
            # Get keywords or keyword paths
            result = self._get_keywords()
            if isinstance(result, tuple):
                self.keywords, self.keyword_paths = result
            else:
                self.keywords = result
                self.keyword_paths = None
            
            self.sensitivities = self._get_sensitivities()
        except ValueError as e:
            self.logger.error(f"Wake word configuration error: {e}")
            raise
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_cooldown = config.cooldown
        
        # Audio gain configuration from config
        self.audio_gain = config.audio_gain if hasattr(config, 'audio_gain') else 1.0
        
        # High-pass filter configuration
        self.highpass_filter_enabled = getattr(config, 'highpass_filter_enabled', False)
        self.highpass_filter_cutoff = getattr(config, 'highpass_filter_cutoff', 50.0)
    
    def __del__(self):
        """Cleanup Porcupine on object destruction"""
        if hasattr(self, 'porcupine') and self.porcupine:
            try:
                self.porcupine.delete()
                self.porcupine = None
            except Exception:
                pass  # Ignore errors during cleanup
    
    def _get_keywords(self):
        """Get list of keywords or keyword paths based on config"""
        model_name = self.config.model
        
        # Check if model is a custom wake word file (.ppn)
        if model_name.endswith('.ppn'):
            # Build path to wake word file in config/wake_words directory
            wake_words_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'wake_words')
            wake_words_dir = os.path.abspath(wake_words_dir)
            
            # Ensure wake_words directory exists
            os.makedirs(wake_words_dir, exist_ok=True)
            
            # Build full path to the .ppn file
            keyword_path = os.path.join(wake_words_dir, model_name)
            
            if not os.path.exists(keyword_path):
                raise ValueError(
                    f"Custom wake word file not found: {model_name}\n"
                    f"Expected location: config/wake_words/{model_name}\n"
                    f"Please:\n"
                    f"  1. Create your wake word at https://console.picovoice.ai/\n"
                    f"  2. Download the .ppn file\n"
                    f"  3. Place it in config/wake_words/{model_name}"
                )
            
            self.logger.info(f"Using custom wake word from: {keyword_path}")
            # Return tuple: (keywords=None, keyword_paths=[path])
            return (None, [keyword_path])
        
        # Handle built-in keywords
        if model_name in self.KEYWORD_MAPPING:
            keyword = self.KEYWORD_MAPPING[model_name]
            self.logger.info(f"Using wake word '{keyword}' (mapped from config '{model_name}')")
            return [keyword]
        elif model_name in self.KEYWORD_MAPPING.values():
            self.logger.info(f"Using wake word '{model_name}' (direct match)")
            return [model_name]
        else:
            # List available keywords for better error message
            available_configs = sorted(self.KEYWORD_MAPPING.keys())
            available_direct = sorted(set(self.KEYWORD_MAPPING.values()))
            
            error_msg = f"Invalid wake word: '{model_name}'"
            self.logger.error(error_msg)
            self.logger.error(f"Available config names: {', '.join(available_configs)}")
            self.logger.error(f"Available direct keywords: {', '.join(available_direct)}")
            self.logger.error("For custom wake words, create a .ppn file at https://console.picovoice.ai/")
            
            raise ValueError(
                f"{error_msg}\n"
                f"Available options:\n"
                f"  Config names: {', '.join(available_configs)}\n"
                f"  Direct keywords: {', '.join(available_direct)}\n"
                f"  For custom wake words:\n"
                f"    1. Create at https://console.picovoice.ai/\n"
                f"    2. Download the .ppn file\n"
                f"    3. Place in config/wake_words/\n"
                f"    4. Set model to the filename (e.g., 'my_wake_word.ppn')"
            )
    
    def _get_sensitivities(self) -> List[float]:
        """Get sensitivity values for each keyword"""
        # Porcupine sensitivity is 0.0-1.0, same as our config
        # Validate and clamp sensitivity to valid range
        sensitivity = self.config.sensitivity
        if sensitivity < 0.0 or sensitivity > 1.0:
            self.logger.warning(f"Sensitivity {sensitivity} is outside valid range [0.0, 1.0], clamping")
            sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Log the actual sensitivity being used
        self.logger.debug(f"Using sensitivity value: {sensitivity} (original: {self.config.sensitivity})")
        
        # Use the same sensitivity for all keywords or keyword paths
        if self.keyword_paths:
            return [sensitivity] * len(self.keyword_paths)
        else:
            return [sensitivity] * len(self.keywords)
    
    async def start(self) -> None:
        """Start wake word detection"""
        self.logger.debug("PorcupineDetector.start() called")
        if self.is_running:
            self.logger.warning("Wake word detector already running")
            return
        
        # Clean up any existing Porcupine instance before starting
        if hasattr(self, 'porcupine') and self.porcupine:
            self.logger.warning("Found existing Porcupine instance - cleaning up before restart")
            try:
                self.porcupine.delete()
            except Exception as e:
                self.logger.error(f"Error cleaning up existing Porcupine: {e}")
            self.porcupine = None
        
        try:
            # Check access key before initialization
            if not self.access_key:
                raise ValueError("Picovoice access key not found. Set PICOVOICE_ACCESS_KEY environment variable")
            
            # Log initialization status
            self.logger.info("Starting Porcupine wake word detector...")
            self.logger.info(f"Access key configured: {'Yes' if self.access_key else 'No'}")
            self.logger.info(f"Access key length: {len(self.access_key) if self.access_key else 0}")
            if self.keyword_paths:
                self.logger.info(f"Custom keyword paths: {self.keyword_paths}")
            else:
                self.logger.info(f"Keywords to detect: {self.keywords}")
            self.logger.info(f"Sensitivities: {self.sensitivities}")
            self.logger.debug(f"self.porcupine status before init: {self.porcupine is not None}")
            
            # Initialize Porcupine only if it doesn't exist
            if not self.porcupine:
                self.logger.info("Creating Porcupine instance (this may take a moment)...")
                self.logger.debug("Starting Porcupine initialization")
                
                # Create Porcupine in a separate thread to allow timeout
                loop = asyncio.get_event_loop()
                
                def create_porcupine():
                    self.logger.debug("Inside create_porcupine()")
                    if self.keyword_paths:
                        self.logger.debug(f"Creating Porcupine with custom keyword paths: {self.keyword_paths}")
                        return pvporcupine.create(
                            access_key=self.access_key,
                            keyword_paths=self.keyword_paths,
                            sensitivities=self.sensitivities
                        )
                    else:
                        self.logger.debug(f"Creating Porcupine with built-in keywords: {self.keywords}")
                        return pvporcupine.create(
                            access_key=self.access_key,
                            keywords=self.keywords,
                            sensitivities=self.sensitivities
                        )
                
                # Use ThreadPoolExecutor to run blocking call with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = loop.run_in_executor(executor, create_porcupine)
                    try:
                        # Wait up to 30 seconds for initialization
                        self.logger.debug("Waiting for Porcupine creation...")
                        self.porcupine = await asyncio.wait_for(future, timeout=30.0)
                        self.logger.debug("Porcupine creation completed")
                    except asyncio.TimeoutError:
                        self.logger.error("Porcupine initialization timed out after 30 seconds")
                        self.logger.error("This may indicate:")
                        self.logger.error("  - Invalid access key")
                        self.logger.error("  - Network connectivity issues")
                        self.logger.error("  - Firewall blocking Picovoice servers")
                        raise TimeoutError("Porcupine initialization timed out")
            else:
                self.logger.warning("Porcupine already initialized - skipping creation")
                self.logger.debug("Skipping Porcupine creation - already exists")
            
            # Log successful initialization
            self.logger.info("Porcupine initialized successfully!")
            self.logger.info(f"Sample rate: {self.porcupine.sample_rate}Hz")
            self.logger.info(f"Frame length: {self.porcupine.frame_length} samples")
            self.logger.info(f"Version: {self.porcupine.version}")
            
            # Model verification logging
            self.logger.debug("Model verification:")
            self.logger.debug(f"  Config model name: {self.config.model}")
            if self.keyword_paths:
                self.logger.debug(f"  Custom keyword paths: {self.keyword_paths}")
            else:
                self.logger.debug(f"  Mapped keywords: {self.keywords}")
            self.logger.debug(f"  Sensitivities: {self.sensitivities}")
            self.logger.debug(f"  Audio gain: {self.audio_gain}")
            
            # Start detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.is_running = True
            self.logger.info("Porcupine wake word detection thread started")
            
        except Exception as e:
            self.logger.error(f"Failed to start wake word detection: {e}")
            if "AccessKey" in str(e):
                self.logger.error("Invalid Picovoice access key!")
                self.logger.error("Please check your PICOVOICE_ACCESS_KEY environment variable")
                self.logger.error("Get your free access key at: https://console.picovoice.ai/")
            elif isinstance(e, TimeoutError):
                self.logger.error("Check your internet connection and firewall settings")
            raise
    
    async def stop(self) -> None:
        """Stop wake word detection"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop detection thread
        self.stop_event.set()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        # Clear audio buffer
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Clean up Porcupine
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        self.logger.info("Porcupine wake word detection stopped")
    
    def add_detection_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Add callback for wake word detection
        
        Args:
            callback: Function to call when wake word detected (model_name, confidence)
        """
        self.detection_callbacks.append(callback)
    
    def remove_detection_callback(self, callback: Callable[[str, float], None]) -> None:
        """Remove detection callback"""
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
    
    def process_audio(self, audio_data: bytes, input_sample_rate: int = 24000) -> None:
        """
        Process audio data for wake word detection
        
        Args:
            audio_data: Audio data (PCM16 format)
            input_sample_rate: Sample rate of input audio (default 24000 for OpenAI)
        """
        if not self.is_running:
            return
        
        # Debug logging
        if not hasattr(self, '_process_counter'):
            self._process_counter = 0
            print(f"DEBUG: Porcupine process_audio first call - input_rate: {input_sample_rate}Hz, data: {len(audio_data)} bytes", flush=True)
        
        self._process_counter += 1
        
        try:
            # Convert bytes to numpy array (PCM16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Log audio characteristics every 50th call
            if self._process_counter % 50 == 0:
                audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                print(f"DEBUG: Porcupine process #{self._process_counter} - samples: {len(audio_array)}, max level: {audio_level}, rate: {input_sample_rate}Hz", flush=True)
            
            # Apply gain if configured
            if self.audio_gain != 1.0:
                # Convert to float for multiplication to avoid overflow
                audio_float = audio_array.astype(np.float32)
                
                # Apply soft limiting to prevent harsh clipping
                # Normalize to -1 to 1 range
                audio_normalized = audio_float / 32768.0
                
                # Apply gain
                audio_gained = audio_normalized * self.audio_gain
                
                # Soft limiting using tanh compression for values > 0.9
                threshold = 0.9
                mask = np.abs(audio_gained) > threshold
                if np.any(mask):
                    # Apply soft limiting to values above threshold
                    audio_gained[mask] = np.sign(audio_gained[mask]) * (threshold + (1 - threshold) * np.tanh((np.abs(audio_gained[mask]) - threshold) / (1 - threshold)))
                
                # Convert back to int16 range
                audio_float = audio_gained * 32767.0
                
                # Track clipping statistics
                if not hasattr(self, '_clip_stats'):
                    self._clip_stats = {'total_frames': 0, 'clipped_frames': 0, 'soft_limited_frames': 0}
                
                self._clip_stats['total_frames'] += 1
                max_val = np.max(np.abs(audio_gained)) if len(audio_gained) > 0 else 0
                
                if np.any(mask):
                    self._clip_stats['soft_limited_frames'] += 1
                    if self._clip_stats['soft_limited_frames'] % 10 == 0:
                        limit_percentage = (self._clip_stats['soft_limited_frames'] / self._clip_stats['total_frames']) * 100
                        self.logger.info(f"Soft limiting active - {self._clip_stats['soft_limited_frames']} frames limited ({limit_percentage:.1f}%) with gain {self.audio_gain}")
                
                # Log stats periodically
                if self._process_counter % 50 == 0:
                    if self._clip_stats['soft_limited_frames'] > 0:
                        limit_percentage = (self._clip_stats['soft_limited_frames'] / self._clip_stats['total_frames']) * 100
                        self.logger.debug(f"Audio stats - Max: {max_val:.3f}, Soft limited: {limit_percentage:.1f}% ({self._clip_stats['soft_limited_frames']}/{self._clip_stats['total_frames']} frames)")
                    else:
                        self.logger.debug(f"Audio stats - Max: {max_val:.3f}, No limiting needed (gain={self.audio_gain})")
                
                # Final safety clip (should rarely be needed with soft limiting)
                audio_float = np.clip(audio_float, -32768, 32767)
                audio_array = audio_float.astype(np.int16)
            
            # Apply high-pass filter if enabled
            if self.highpass_filter_enabled and SCIPY_AVAILABLE:
                # Log filter status on first call
                if self._process_counter == 1:
                    print(f"DEBUG: High-pass filter enabled at {self.highpass_filter_cutoff}Hz", flush=True)
                
                if not hasattr(self, '_highpass_sos'):
                    # Design a 4th order Butterworth high-pass filter
                    nyquist = input_sample_rate / 2
                    cutoff = self.highpass_filter_cutoff / nyquist
                    self._highpass_sos = signal.butter(4, cutoff, btype='high', output='sos')
                
                # Apply filter
                audio_filtered = signal.sosfilt(self._highpass_sos, audio_array)
                audio_array = audio_filtered.astype(np.int16)
            elif self._process_counter == 1:
                print(f"DEBUG: High-pass filter disabled for better wake word detection", flush=True)
            
            # Resample to 16kHz if needed
            if input_sample_rate != self.sample_rate:
                # Log resampling and audio levels
                if self._process_counter % 50 == 0:
                    pre_resample_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                    print(f"DEBUG: Resampling from {input_sample_rate}Hz to {self.sample_rate}Hz, level after gain: {pre_resample_level}", flush=True)
                
                # Calculate resampling parameters
                resample_ratio = self.sample_rate / input_sample_rate
                new_length = int(len(audio_array) * resample_ratio)
                
                # Log the conversion
                if self._process_counter % 50 == 0:
                    print(f"DEBUG: Resampling {len(audio_array)} samples to {new_length} samples (ratio: {resample_ratio})", flush=True)
                
                # Use scipy for high-quality resampling if available
                if SCIPY_AVAILABLE:
                    # Use scipy's resample for better quality
                    audio_array = signal.resample(audio_array, new_length).astype(np.int16)
                    if self._process_counter == 1:
                        print("DEBUG: Using scipy.signal.resample for high-quality resampling", flush=True)
                else:
                    # Fallback to linear interpolation
                    old_indices = np.arange(len(audio_array))
                    new_indices = np.linspace(0, len(audio_array) - 1, new_length)
                    audio_array = np.interp(new_indices, old_indices, audio_array).astype(np.int16)
                    if self._process_counter == 1:
                        print("DEBUG: Using numpy.interp for resampling (install scipy for better quality)", flush=True)
            
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            
            # Process complete frames
            frames_queued = 0
            while len(self.audio_buffer) >= self.frame_length:
                # Extract frame
                frame = self.audio_buffer[:self.frame_length]
                self.audio_buffer = self.audio_buffer[self.frame_length:]
                
                # Validate frame before queuing
                if len(frame) != self.frame_length:
                    self.logger.error(f"Extracted frame size mismatch! Expected {self.frame_length}, got {len(frame)}")
                    continue
                
                # Queue for processing
                self.audio_queue.put(frame, block=False)
                frames_queued += 1
            
            # Log queuing activity
            if frames_queued > 0 and self._process_counter % 10 == 0:
                print(f"DEBUG: Queued {frames_queued} frames, buffer remaining: {len(self.audio_buffer)}, queue size: {self.audio_queue.qsize()}", flush=True)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _detection_loop(self) -> None:
        """Background thread for wake word detection"""
        self.logger.debug("Porcupine detection thread started")
        if self.keyword_paths:
            self.logger.debug(f"Porcupine detection loop started - listening for custom wake words from: {self.keyword_paths}")
        else:
            self.logger.debug(f"Porcupine detection loop started - listening for: {self.keywords}")
        
        frames_processed = 0
        
        while not self.stop_event.is_set():
            try:
                # Get audio frame from queue
                try:
                    audio_frame = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                frames_processed += 1
                
                # Log processing activity
                if frames_processed == 1:
                    self.logger.debug(f"First frame received in detection loop, shape: {audio_frame.shape}, dtype: {audio_frame.dtype}")
                elif frames_processed % 100 == 0:
                    audio_level = np.max(np.abs(audio_frame)) if len(audio_frame) > 0 else 0
                    self.logger.debug(f"Detection loop processed {frames_processed} frames, current level: {audio_level}, queue: {self.audio_queue.qsize()}")
                
                # Process with Porcupine
                # Debug: Check frame type
                if frames_processed == 1:
                    self.logger.debug(f"Frame type: {type(audio_frame)}, is numpy: {type(audio_frame).__module__ == 'numpy'}")
                    if hasattr(audio_frame, 'dtype'):
                        self.logger.debug(f"Frame dtype: {audio_frame.dtype}")
                
                # Ensure frame is a list of integers (Porcupine requirement)
                if isinstance(audio_frame, np.ndarray):
                    audio_frame = audio_frame.tolist()
                    if frames_processed == 1:
                        self.logger.debug(f"Converted numpy array to list, length: {len(audio_frame)}")
                
                # Validate frame size
                if len(audio_frame) != self.frame_length:
                    self.logger.error(f"Frame size mismatch! Expected {self.frame_length}, got {len(audio_frame)}")
                    continue
                
                # Log frame details periodically for debugging
                if frames_processed % 100 == 0 and frames_processed > 0:
                    # Check audio data integrity
                    frame_array = np.array(audio_frame, dtype=np.int16)
                    frame_max = np.max(np.abs(frame_array)) if len(frame_array) > 0 else 0
                    self.logger.debug(f"Frame #{frames_processed} - size: {len(audio_frame)}, max amplitude: {frame_max}")
                
                # Process with Porcupine
                try:
                    keyword_index = self.porcupine.process(audio_frame)
                except Exception as e:
                    self.logger.error(f"Porcupine process error: {e}")
                    continue
                
                # Log detection result periodically
                if frames_processed % 50 == 0:
                    self.logger.debug(f"Porcupine process result: {keyword_index} (>= 0 means detection)")
                
                # Check for detection
                if keyword_index >= 0:
                    # Get the detected keyword name
                    if self.keyword_paths:
                        # For custom keywords, use the filename without .ppn extension
                        keyword_path = self.keyword_paths[keyword_index] if keyword_index < len(self.keyword_paths) else 'unknown'
                        keyword = os.path.basename(keyword_path).replace('.ppn', '')
                    else:
                        keyword = self.keywords[keyword_index] if keyword_index < len(self.keywords) else 'unknown'
                    
                    # Log wake word detection
                    self.logger.info(f"Porcupine wake word detected: {keyword} (index: {keyword_index})")
                    
                    current_time = time.time()
                    time_since_last = current_time - self.last_detection_time
                    
                    # Check cooldown
                    if time_since_last >= self.detection_cooldown:
                        sensitivity = self.sensitivities[keyword_index]
                        
                        self.logger.info(f"Wake word active: {keyword}")
                        self.last_detection_time = current_time
                        
                        # Call detection callbacks
                        # Porcupine doesn't provide confidence scores, so we use sensitivity as a proxy
                        self.logger.debug(f"Triggering {len(self.detection_callbacks)} detection callbacks")
                        for callback in self.detection_callbacks:
                            try:
                                callback(keyword, sensitivity)
                                self.logger.debug("Callback executed successfully")
                            except Exception as e:
                                self.logger.error(f"Error in detection callback: {e}")
                    else:
                        self.logger.debug(f"Wake word detected but in cooldown period ({time_since_last:.1f}s < {self.detection_cooldown}s)")
                
                # Log progress periodically
                if frames_processed % 100 == 0:
                    self.logger.debug(f"Processed {frames_processed} frames, queue size: {self.audio_queue.qsize()}")
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
        
        self.logger.debug("Porcupine detection thread stopped")
    
    def get_available_models(self) -> list:
        """
        Get list of available wake word models
        
        Returns:
            List of available model names
        """
        return list(self.KEYWORD_MAPPING.keys())
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'engine': 'porcupine',
            'sensitivities': self.sensitivities,
            'sample_rate': self.sample_rate,
            'frame_length': self.frame_length,
            'audio_buffer_size': len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 0,
            'access_key_set': bool(self.access_key)
        }
        
        if self.keyword_paths:
            info['keyword_paths'] = self.keyword_paths
            info['keywords'] = [os.path.basename(p).replace('.ppn', '') for p in self.keyword_paths]
            info['type'] = 'custom'
        else:
            info['keywords'] = self.keywords
            info['type'] = 'built-in'
            
        return info
    
    def reset_audio_buffers(self) -> None:
        """
        Reset audio buffers
        
        Porcupine handles its own internal buffers, so we just clear our accumulation buffer
        """
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("Audio buffers reset")
    
    @staticmethod
    def test_installation() -> bool:
        """
        Test if Porcupine is properly installed
        
        Returns:
            True if installation is working, False otherwise
        """
        logger = get_logger("PorcupineTest")
        
        try:
            import pvporcupine
            
            logger.info("Testing Porcupine installation...")
            
            # Check if we have an access key
            access_key = os.getenv('PICOVOICE_ACCESS_KEY')
            if not access_key:
                logger.error("PICOVOICE_ACCESS_KEY environment variable not set")
                logger.error("Get your free access key at: https://console.picovoice.ai/")
                return False
            
            # Try to create a simple instance
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=['picovoice']
            )
            
            logger.info(f"Porcupine version: {porcupine.version}")
            logger.info(f"Sample rate: {porcupine.sample_rate}Hz")
            logger.info(f"Frame length: {porcupine.frame_length}")
            
            # Clean up
            porcupine.delete()
            
            logger.info("Porcupine installation test passed")
            return True
            
        except ImportError as e:
            logger.error(f"Porcupine not installed: {e}")
            logger.error("Install with: pip install pvporcupine")
            return False
        except Exception as e:
            logger.error(f"Porcupine installation test failed: {e}")
            return False