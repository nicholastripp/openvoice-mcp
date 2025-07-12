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
    # NOTE: 'jarvis' is NOT a built-in Porcupine keyword, removed to prevent errors
    KEYWORD_MAPPING = {
        'alexa': 'alexa',
        'hey_google': 'hey google',
        'ok_google': 'ok google',
        'hey_siri': 'hey siri',
        'picovoice': 'picovoice',
        'bumblebee': 'bumblebee',
        'grasshopper': 'grasshopper',
        'americano': 'americano',
        'blueberry': 'blueberry',
        'grapefruit': 'grapefruit',
        'porcupine': 'porcupine',
        'terminator': 'terminator',
        # Aliases for convenience
        'hey_picovoice': 'picovoice',
        'ok_picovoice': 'picovoice'
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
        
        # Detection parameters
        self.keywords = self._get_keywords()
        self.sensitivities = self._get_sensitivities()
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_cooldown = config.cooldown
        
        # Audio gain configuration from config
        self.audio_gain = config.audio_gain if hasattr(config, 'audio_gain') else 1.0
    
    def _get_keywords(self) -> List[str]:
        """Get list of keywords based on config"""
        model_name = self.config.model
        
        # Map config name to Porcupine keyword
        if model_name in self.KEYWORD_MAPPING:
            return [self.KEYWORD_MAPPING[model_name]]
        elif model_name in self.KEYWORD_MAPPING.values():
            return [model_name]
        else:
            self.logger.warning(f"Unknown wake word model: {model_name}, defaulting to 'picovoice'")
            return ['picovoice']
    
    def _get_sensitivities(self) -> List[float]:
        """Get sensitivity values for each keyword"""
        # Porcupine sensitivity is 0.0-1.0, same as our config
        # Validate and clamp sensitivity to valid range
        sensitivity = self.config.sensitivity
        if sensitivity < 0.0 or sensitivity > 1.0:
            self.logger.warning(f"Sensitivity {sensitivity} is outside valid range [0.0, 1.0], clamping")
            sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Log the actual sensitivity being used
        print(f"DEBUG: Using sensitivity value: {sensitivity} (original: {self.config.sensitivity})", flush=True)
        
        # Use the same sensitivity for all keywords
        return [sensitivity] * len(self.keywords)
    
    async def start(self) -> None:
        """Start wake word detection"""
        print("DEBUG: PorcupineDetector.start() called", flush=True)
        if self.is_running:
            self.logger.warning("Wake word detector already running")
            return
        
        try:
            # Check access key before initialization
            if not self.access_key:
                raise ValueError("Picovoice access key not found. Set PICOVOICE_ACCESS_KEY environment variable")
            
            # Log initialization status
            self.logger.info("Starting Porcupine wake word detector...")
            self.logger.info(f"Access key configured: {'Yes' if self.access_key else 'No'}")
            self.logger.info(f"Access key length: {len(self.access_key) if self.access_key else 0}")
            self.logger.info(f"Keywords to detect: {self.keywords}")
            self.logger.info(f"Sensitivities: {self.sensitivities}")
            
            # Initialize Porcupine with timeout
            self.logger.info("Creating Porcupine instance (this may take a moment)...")
            
            # Create Porcupine in a separate thread to allow timeout
            loop = asyncio.get_event_loop()
            
            def create_porcupine():
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
                    self.porcupine = await asyncio.wait_for(future, timeout=30.0)
                except asyncio.TimeoutError:
                    self.logger.error("Porcupine initialization timed out after 30 seconds")
                    self.logger.error("This may indicate:")
                    self.logger.error("  - Invalid access key")
                    self.logger.error("  - Network connectivity issues")
                    self.logger.error("  - Firewall blocking Picovoice servers")
                    raise TimeoutError("Porcupine initialization timed out")
            
            # Log successful initialization
            self.logger.info("Porcupine initialized successfully!")
            self.logger.info(f"Sample rate: {self.porcupine.sample_rate}Hz")
            self.logger.info(f"Frame length: {self.porcupine.frame_length} samples")
            self.logger.info(f"Version: {self.porcupine.version}")
            
            # Model verification logging
            print("DEBUG: Model verification:", flush=True)
            print(f"  Config model name: {self.config.model}", flush=True)
            print(f"  Mapped keywords: {self.keywords}", flush=True)
            print(f"  Sensitivities: {self.sensitivities}", flush=True)
            print(f"  Audio gain: {self.audio_gain}", flush=True)
            
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
                        print(f"INFO: Soft limiting active - {self._clip_stats['soft_limited_frames']} frames limited ({limit_percentage:.1f}%) with gain {self.audio_gain}", flush=True)
                
                # Log stats periodically
                if self._process_counter % 50 == 0:
                    if self._clip_stats['soft_limited_frames'] > 0:
                        limit_percentage = (self._clip_stats['soft_limited_frames'] / self._clip_stats['total_frames']) * 100
                        print(f"DEBUG: Audio stats - Max: {max_val:.3f}, Soft limited: {limit_percentage:.1f}% ({self._clip_stats['soft_limited_frames']}/{self._clip_stats['total_frames']} frames)", flush=True)
                    else:
                        print(f"DEBUG: Audio stats - Max: {max_val:.3f}, No limiting needed (gain={self.audio_gain})", flush=True)
                
                # Final safety clip (should rarely be needed with soft limiting)
                audio_float = np.clip(audio_float, -32768, 32767)
                audio_array = audio_float.astype(np.int16)
            
            # Apply high-pass filter to remove low-frequency noise (< 80Hz)
            # This helps with wake word detection by removing rumble and DC offset
            if not hasattr(self, '_highpass_sos'):
                from scipy import signal
                # Design a 4th order Butterworth high-pass filter at 80Hz
                nyquist = input_sample_rate / 2
                cutoff = 80 / nyquist
                self._highpass_sos = signal.butter(4, cutoff, btype='high', output='sos')
            
            # Apply filter
            audio_filtered = signal.sosfilt(self._highpass_sos, audio_array)
            audio_array = audio_filtered.astype(np.int16)
            
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
        print(f"DEBUG: Porcupine detection loop started - listening for: {self.keywords}", flush=True)
        
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
                    print(f"DEBUG: First frame received in detection loop, shape: {audio_frame.shape}, dtype: {audio_frame.dtype}", flush=True)
                elif frames_processed % 100 == 0:
                    audio_level = np.max(np.abs(audio_frame)) if len(audio_frame) > 0 else 0
                    print(f"DEBUG: Detection loop processed {frames_processed} frames, current level: {audio_level}, queue: {self.audio_queue.qsize()}", flush=True)
                
                # Process with Porcupine
                # Debug: Check frame type
                if frames_processed == 1:
                    print(f"DEBUG: Frame type: {type(audio_frame)}, is numpy: {type(audio_frame).__module__ == 'numpy'}", flush=True)
                    if hasattr(audio_frame, 'dtype'):
                        print(f"DEBUG: Frame dtype: {audio_frame.dtype}", flush=True)
                
                # Ensure frame is a list of integers (Porcupine requirement)
                if isinstance(audio_frame, np.ndarray):
                    audio_frame = audio_frame.tolist()
                    if frames_processed == 1:
                        print(f"DEBUG: Converted numpy array to list, length: {len(audio_frame)}", flush=True)
                
                keyword_index = self.porcupine.process(audio_frame)
                
                # Log detection result periodically
                if frames_processed % 50 == 0:
                    print(f"DEBUG: Porcupine process result: {keyword_index} (>= 0 means detection)", flush=True)
                
                # Check for detection
                if keyword_index >= 0:
                    keyword = self.keywords[keyword_index] if keyword_index < len(self.keywords) else 'unknown'
                    
                    # PROMINENT DETECTION LOGGING
                    print("\n" + "="*70, flush=True)
                    print("🎯 PORCUPINE WAKE WORD DETECTED! 🎯".center(70), flush=True)
                    print("="*70, flush=True)
                    print(f"Keyword: {keyword}".center(70), flush=True)
                    print(f"Index: {keyword_index}".center(70), flush=True)
                    print(f"Frame: {frames_processed}".center(70), flush=True)
                    print("="*70 + "\n", flush=True)
                    
                    current_time = time.time()
                    time_since_last = current_time - self.last_detection_time
                    
                    # Check cooldown
                    if time_since_last >= self.detection_cooldown:
                        keyword = self.keywords[keyword_index]
                        sensitivity = self.sensitivities[keyword_index]
                        
                        self.logger.info(f"Wake word detected: {keyword} (index: {keyword_index})")
                        print(f"✅ WAKE WORD ACTIVE: '{keyword}' (sensitivity: {sensitivity}, cooldown OK: {time_since_last:.2f}s)", flush=True)
                        self.last_detection_time = current_time
                        
                        # Call detection callbacks
                        # Porcupine doesn't provide confidence scores, so we use sensitivity as a proxy
                        print(f"📢 Triggering {len(self.detection_callbacks)} detection callbacks...", flush=True)
                        for callback in self.detection_callbacks:
                            try:
                                callback(keyword, sensitivity)
                                print(f"✅ Callback executed successfully", flush=True)
                            except Exception as e:
                                self.logger.error(f"Error in detection callback: {e}")
                                print(f"❌ Callback error: {e}", flush=True)
                    else:
                        print(f"⏸️ COOLDOWN ACTIVE: Wake word detected but waiting ({time_since_last:.1f}s < {self.detection_cooldown}s)", flush=True)
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
        return {
            'engine': 'porcupine',
            'keywords': self.keywords,
            'sensitivities': self.sensitivities,
            'sample_rate': self.sample_rate,
            'frame_length': self.frame_length,
            'audio_buffer_size': len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 0,
            'access_key_set': bool(self.access_key)
        }
    
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