"""
OpenWakeWord wake word detection implementation
"""
import asyncio
import numpy as np
import threading
from typing import Callable, Optional, Any
from queue import Queue, Empty
import time

from config import WakeWordConfig
from utils.logger import get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


class WakeWordDetector:
    """
    Wake word detection using OpenWakeWord
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.logger = get_logger("WakeWordDetector")
        
        if not OPENWAKEWORD_AVAILABLE:
            raise ImportError("OpenWakeWord not available. Install with: pip install openwakeword>=0.6.0")
        
        # Audio parameters (OpenWakeWord requirements)
        self.sample_rate = 16000  # OpenWakeWord requires 16kHz
        self.chunk_size = 1280    # 80ms at 16kHz (recommended)
        
        # State
        self.is_running = False
        self.model: Optional[WakeWordModel] = None
        self.audio_queue = Queue()
        self.detection_callbacks = []
        
        # Audio chunk buffer for accumulating to 80ms (1280 samples at 16kHz)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.target_chunk_size = self.chunk_size  # 1280 samples (80ms at 16kHz)
        
        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Detection parameters
        self.model_name = config.model
        self.sensitivity = config.sensitivity
        self.vad_enabled = config.vad_enabled
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_cooldown = config.cooldown
        
        # Remove startup time tracking (replaced with VAD and buffer initialization)
        # self.startup_time = 0  # No longer needed
    
    async def start(self) -> None:
        """Start wake word detection"""
        if self.is_running:
            self.logger.warning("Wake word detector already running")
            return
        
        try:
            # Initialize OpenWakeWord model
            self.logger.info(f"Loading wake word model: {self.model_name}")
            self._load_model()
            
            # Start detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.is_running = True
            self.logger.info(f"Wake word detection started with model: {self.model_name}")
            self.logger.info(f"Audio parameters: {self.sample_rate}Hz, chunk_size={self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")
            self.logger.info("Ready for immediate wake word detection (no startup delay)")
            
        except Exception as e:
            self.logger.error(f"Failed to start wake word detection: {e}")
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
        
        # Clear queue and buffer
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        # Clear audio buffer
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer = np.array([], dtype=np.float32)
        
        self.logger.info("Wake word detection stopped")
    
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
            self.logger.debug("process_audio called but detector not running")
            return
        
        # Debug: Log that we're receiving audio
        self.logger.debug(f"process_audio called: {len(audio_data)} bytes at {input_sample_rate}Hz")
        
        try:
            # Convert bytes to numpy array (assuming PCM16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Calculate audio level for debugging
            audio_level = np.max(np.abs(audio_float))
            
            # Resample to 16kHz if needed (OpenWakeWord requirement)
            if input_sample_rate != self.sample_rate:
                from scipy import signal
                # Calculate new length for resampling
                new_length = int(len(audio_float) * self.sample_rate / input_sample_rate)
                audio_float = signal.resample(audio_float, new_length)
                
                self.logger.debug(f"Resampled audio from {input_sample_rate}Hz to {self.sample_rate}Hz: {len(audio_array)} -> {len(audio_float)} samples (level: {audio_level:.3f})")
            
            # Immediate debug feedback (not just debug logs)
            print(f"   DETECTOR: audio_level={audio_level:.3f}, samples={len(audio_float)}")
            
            # Lower threshold to allow speech audio (was 0.005)
            if audio_level > 0.001:  # Much lower threshold for legitimate audio activity
                # Add to buffer instead of directly queuing
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
                
                # Check if we have enough samples for OpenWakeWord (80ms = 1280 samples at 16kHz)
                while len(self.audio_buffer) >= self.target_chunk_size:
                    # Extract exactly 1280 samples
                    chunk_for_oww = self.audio_buffer[:self.target_chunk_size]
                    self.audio_buffer = self.audio_buffer[self.target_chunk_size:]
                    
                    # Queue the properly-sized chunk for OpenWakeWord
                    self.audio_queue.put(chunk_for_oww, block=False)
                    print(f"   DETECTOR: QUEUED 80ms chunk for OpenWakeWord (samples={len(chunk_for_oww)}, buffer_remaining={len(self.audio_buffer)}, queue_size={self.audio_queue.qsize()})")
                    self.logger.debug(f"Queued 80ms chunk for processing: samples={len(chunk_for_oww)}, buffer_remaining={len(self.audio_buffer)}, queue_size={self.audio_queue.qsize()}")
                
                # Debug: Log audio activity
                if audio_level > 0.01:  # Only log when there's significant audio
                    self.logger.debug(f"Audio activity detected: level={audio_level:.3f}, samples={len(audio_float)}, buffer_size={len(self.audio_buffer)}")
            else:
                # Skip silence/noise that causes false positives
                print(f"   DETECTOR: FILTERED OUT low audio (level={audio_level:.3f})")
                self.logger.debug(f"Skipping low-level audio: level={audio_level:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _check_model_availability(self) -> bool:
        """Check if the requested model is available"""
        try:
            import openwakeword
            from pathlib import Path
            
            # Map common model names to actual file names
            model_mapping = {
                'hey_jarvis': 'hey_jarvis_v0.1.tflite',
                'alexa': 'alexa_v0.1.tflite', 
                'hey_mycroft': 'hey_mycroft_v0.1.tflite',
                'hey_rhasspy': 'hey_rhasspy_v0.1.tflite',
                'ok_nabu': 'ok_nabu_v0.1.tflite'
            }
            
            model_filename = model_mapping.get(self.model_name, f"{self.model_name}.tflite")
            models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
            model_path = models_dir / model_filename
            
            self.logger.debug(f"Checking for model file: {model_path}")
            
            if model_path.exists():
                self.logger.debug(f"Model file found: {model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                
                # List available models
                if models_dir.exists():
                    available_files = list(models_dir.glob("*.tflite"))
                    if available_files:
                        self.logger.info("Available model files:")
                        for file in available_files:
                            self.logger.info(f"  - {file.name}")
                    else:
                        self.logger.warning("No model files found in models directory")
                else:
                    self.logger.warning(f"Models directory does not exist: {models_dir}")
                
                return False
                
        except Exception as e:
            self.logger.warning(f"Error checking model availability: {e}")
            return False
    
    def _download_models(self) -> bool:
        """Download OpenWakeWord models if not available"""
        # Check if auto-download is enabled
        if not getattr(self.config, 'auto_download', True):
            self.logger.info("Automatic model download is disabled")
            return False
            
        try:
            import openwakeword
            from openwakeword import utils
            import signal
            
            self.logger.info("Downloading OpenWakeWord models...")
            self.logger.info("This may take a few minutes on first run...")
            
            # Set up timeout for download
            download_timeout = getattr(self.config, 'download_timeout', 300)
            retry_attempts = getattr(self.config, 'retry_downloads', 3)
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Model download timed out after {download_timeout} seconds")
            
            # Try download with retries
            for attempt in range(retry_attempts):
                try:
                    if attempt > 0:
                        self.logger.info(f"Retry attempt {attempt + 1}/{retry_attempts}")
                    
                    # Set timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(download_timeout)
                    
                    # Download models
                    utils.download_models()
                    
                    # Clear timeout
                    signal.alarm(0)
                    
                    self.logger.info("[OK] Model download completed successfully")
                    return True
                    
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)  # Clear timeout
                    if attempt < retry_attempts - 1:
                        self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                        continue
                    else:
                        raise e
            
            return False
            
        except Exception as e:
            self.logger.error(f"[ERROR] Model download failed: {e}")
            
            # Provide helpful troubleshooting information
            self.logger.error("")
            self.logger.error("[HELP] Manual download options:")
            self.logger.error("1. Check internet connectivity")
            self.logger.error("2. Try running: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
            self.logger.error("3. Download manually from: https://github.com/dscripka/openWakeWord/releases")
            self.logger.error("4. Set 'auto_download: false' in config to disable automatic downloads")
            self.logger.error("")
            
            return False

    def _load_model(self) -> None:
        """Load OpenWakeWord model"""
        try:
            # Check if model is available first
            if not self._check_model_availability():
                self.logger.info(f"Model '{self.model_name}' not found, attempting download...")
                
                # Try to download models
                if self._download_models():
                    # Check again after download
                    if not self._check_model_availability():
                        raise FileNotFoundError(f"Model '{self.model_name}' is still not available after download")
                    else:
                        self.logger.info(f"Successfully downloaded model '{self.model_name}'")
                else:
                    raise FileNotFoundError(f"Model '{self.model_name}' is not available and download failed")
            
            # Map common model names to OpenWakeWord models
            model_mapping = {
                'hey_jarvis': 'hey_jarvis_v0.1',
                'alexa': 'alexa_v0.1', 
                'hey_mycroft': 'hey_mycroft_v0.1',
                'hey_rhasspy': 'hey_rhasspy_v0.1',
                'ok_nabu': 'ok_nabu_v0.1'
            }
            
            actual_model_name = model_mapping.get(self.model_name, self.model_name)
            
            # Initialize model with VAD and noise suppression
            model_kwargs = {}
            
            # TEMPORARILY DISABLE VAD threshold for debugging (likely cause of 0.0 predictions)
            if self.vad_enabled:
                # model_kwargs['vad_threshold'] = 0.5  # VAD threshold for filtering non-speech
                self.logger.info("[DEBUG] VAD threshold DISABLED for debugging - should see non-zero predictions")
                self.logger.info("[DEBUG] If this fixes the issue, VAD threshold 0.5 was too aggressive")
            
            # TEMPORARILY DISABLE Speex noise suppression for debugging
            speex_enabled = False  # Force disable for minimal config testing
            self.logger.info("[DEBUG] Speex noise suppression DISABLED for debugging")
            self.logger.info("[DEBUG] Testing with minimal OpenWakeWord configuration")
            
            self.model = WakeWordModel(
                wakeword_models=[actual_model_name],
                **model_kwargs
            )
            
            self.logger.info(f"[OK] Successfully loaded model: {actual_model_name}")
            self.logger.info(f"Available models: {list(self.model.models.keys())}")
            self.logger.info(f"Model expects: {self.sample_rate}Hz audio in {self.chunk_size} sample chunks (80ms)")
            self.logger.info(f"Model configuration: VAD={self.vad_enabled}, Speex={speex_enabled}, sensitivity={self.sensitivity}")
            
            # REMOVED: Buffer initialization with silence chunks
            # The 1.5s silence buffer initialization was causing the model to get stuck
            # in a state where it only returns constant prediction values.
            # 
            # Original issue: Model was returning identical predictions (1.1165829e-06)
            # for every audio chunk regardless of content.
            #
            # Solution: Let the model start with a clean state and process real audio immediately.
            
            self.logger.info("[OK] Model loaded without buffer initialization")
            self.logger.info("[INFO] Model will start processing audio immediately with clean state")
            
            # Test one minimal chunk to verify model is functional
            try:
                test_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                test_predictions = self.model.predict(test_chunk)
                self.logger.info(f"[TEST] Initial test prediction: {test_predictions}")
                
                # Verify we get a dictionary with model predictions
                if not isinstance(test_predictions, dict) or not test_predictions:
                    self.logger.warning("[WARNING] Model test prediction is empty or invalid")
                else:
                    self.logger.info("[OK] Model test prediction successful")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Initial model test failed: {e}")
                raise
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load wake word model '{self.model_name}': {e}")
            
            # Check if this is a missing model file error
            if "Could not open" in str(e) and ".tflite" in str(e):
                self.logger.error("")
                self.logger.error("[HELP] Troubleshooting steps:")
                self.logger.error("1. Check internet connectivity for model download")
                self.logger.error("2. Try running: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
                self.logger.error("3. Download manually from: https://github.com/dscripka/openWakeWord/releases")
                self.logger.error("4. Check if your model name is correct (alexa, hey_jarvis, hey_mycroft, etc.)")
                self.logger.error("")
                
                # Try to list available models
                try:
                    self._suggest_available_models()
                except:
                    pass
            
            # Try to suggest available models
            try:
                dummy_model = WakeWordModel()
                available = list(dummy_model.models.keys())
                if available:
                    self.logger.info(f"Available models after download attempt: {available}")
                else:
                    self.logger.warning("No models available - download may have failed")
            except:
                pass
            raise
    
    def _suggest_available_models(self) -> None:
        """Try to suggest available wake word models"""
        try:
            # Try to create a model with no specific models to see what's available
            dummy_model = WakeWordModel()
            available = list(dummy_model.models.keys())
            if available:
                self.logger.info(f"Available pre-installed models: {available}")
                self.logger.info("You can use one of these models instead.")
            else:
                self.logger.info("No pre-installed models found.")
                self.logger.info("You may need to download wake word models manually.")
        except Exception as e:
            self.logger.debug(f"Could not determine available models: {e}")
    
    def _detection_loop(self) -> None:
        """Background thread for wake word detection"""
        self.logger.debug("Wake word detection thread started")
        
        chunks_processed = 0
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                chunks_processed += 1
                
                # Immediate debug feedback for OpenWakeWord processing
                if chunks_processed % 50 == 0:  # Every 50 chunks
                    chunk_length_ms = len(audio_chunk) / self.sample_rate * 1000
                    print(f"   DETECTOR: OpenWakeWord processing chunk #{chunks_processed}, samples={len(audio_chunk)}, duration={chunk_length_ms:.1f}ms, queue_size={self.audio_queue.qsize()}")
                
                # Debug: Log queue processing periodically
                if chunks_processed % 100 == 0:
                    self.logger.debug(f"Detection loop processed {chunks_processed} chunks, queue size: {self.audio_queue.qsize()}")
                
                # Process with OpenWakeWord - ADD COMPREHENSIVE DEBUGGING
                try:
                    print(f"   DETECTOR: Calling model.predict() with chunk: samples={len(audio_chunk)}, dtype={audio_chunk.dtype}")
                    predictions = self.model.predict(audio_chunk)
                    print(f"   DETECTOR: model.predict() returned: {predictions}")
                except Exception as e:
                    print(f"   DETECTOR: ERROR in model.predict(): {e}")
                    self.logger.error(f"OpenWakeWord prediction error: {e}")
                    continue
                
                # Debug: Log all predictions to understand what's happening
                if predictions:
                    max_confidence = max(predictions.values()) if predictions else 0.0
                    print(f"   DETECTOR: OpenWakeWord predictions: {predictions} (max: {max_confidence:.3f})")
                    
                    # Show prediction scores periodically and for any significant activity
                    if chunks_processed % 100 == 0 or max_confidence > 0.05:
                        formatted_predictions = {k: f"{v:.3f}" for k, v in predictions.items()}
                        print(f"   DETECTOR: OpenWakeWord predictions: {formatted_predictions} (max: {max_confidence:.3f})")
                    
                    if max_confidence > 0.1:  # Log any significant predictions
                        self.logger.debug(f"OpenWakeWord predictions: {predictions}")
                else:
                    print(f"   DETECTOR: OpenWakeWord returned EMPTY predictions!")
                    if chunks_processed % 50 == 0:  # Log empty predictions periodically
                        self.logger.warning(f"OpenWakeWord predictions are empty at chunk {chunks_processed}")
                    continue
                
                # Check for wake word detection
                for model_name, confidence in predictions.items():
                    # Log any confidence above threshold for debugging
                    if confidence > self.sensitivity * 0.3:  # Log at 30% of sensitivity
                        print(f"   DETECTOR: {model_name} confidence {confidence:.3f} (threshold: {self.sensitivity:.3f}, above 30%)")
                    
                    if confidence >= self.sensitivity:
                        current_time = time.time()
                        
                        print(f"   DETECTOR: DETECTION CANDIDATE: {model_name} confidence {confidence:.3f} >= {self.sensitivity:.3f}")
                        
                        # Check cooldown to prevent rapid re-triggers (removed artificial startup delay)
                        time_since_last = current_time - self.last_detection_time
                        if time_since_last >= self.detection_cooldown:
                            print(f"   DETECTOR: WAKE WORD DETECTED! {model_name} (confidence: {confidence:.3f})")
                            self.logger.info(f"Wake word detected: {model_name} (confidence: {confidence:.3f})")
                            self.logger.debug(f"Cooldown passed: {time_since_last:.1f}s >= {self.detection_cooldown}s")
                            self.last_detection_time = current_time
                            
                            # Call detection callbacks
                            for callback in self.detection_callbacks:
                                try:
                                    callback(model_name, confidence)
                                except Exception as e:
                                    self.logger.error(f"Error in detection callback: {e}")
                        else:
                            print(f"   DETECTOR: Detection in cooldown: {model_name} (confidence: {confidence:.3f}), {time_since_last:.1f}s < {self.detection_cooldown}s")
                            self.logger.debug(f"Wake word detected but in cooldown: {model_name} (confidence: {confidence:.3f}), {time_since_last:.1f}s < {self.detection_cooldown}s")
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
        
        self.logger.debug("Wake word detection thread stopped")
    
    def get_available_models(self) -> list:
        """
        Get list of available wake word models
        
        Returns:
            List of available model names
        """
        try:
            dummy_model = WakeWordModel()
            return list(dummy_model.models.keys())
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {}
        
        return {
            'model_name': self.model_name,
            'loaded_models': list(self.model.models.keys()) if self.model else [],
            'sensitivity': self.sensitivity,
            'vad_enabled': self.vad_enabled,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'chunk_duration_ms': self.chunk_size / self.sample_rate * 1000,
            'audio_buffer_size': len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 0
        }
    
    def reset_audio_buffers(self) -> None:
        """
        Reset OpenWakeWord model buffers and clear internal audio buffer
        
        Use this method if you need to clear persistent wake word detections
        or reset the model state during operation.
        """
        if self.model:
            try:
                self.model.reset()
                self.logger.info("OpenWakeWord model buffers reset")
            except Exception as e:
                self.logger.warning(f"Failed to reset model buffers: {e}")
        
        # Clear internal audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.logger.debug("Internal audio buffer cleared")
    
    @staticmethod
    def test_installation() -> bool:
        """
        Test if OpenWakeWord is properly installed
        
        Returns:
            True if installation is working, False otherwise
        """
        logger = get_logger("WakeWordTest")
        
        try:
            # Try to import and create a simple model
            import openwakeword
            from openwakeword import Model as WakeWordModel
            from openwakeword import utils
            
            logger.info("Testing OpenWakeWord installation...")
            
            # Try to download models if they don't exist
            try:
                logger.info("Ensuring models are available...")
                utils.download_models()
                logger.info("[OK] Models download completed")
            except Exception as e:
                logger.warning(f"[WARNING] Model download failed: {e}")
                logger.info("Attempting to use existing models...")
            
            # Try to load a model
            test_model = WakeWordModel(wakeword_models=['hey_jarvis_v0.1'])
            
            # Test with dummy audio
            dummy_audio = np.zeros(1280, dtype=np.float32)  # 80ms of silence
            predictions = test_model.predict(dummy_audio)
            
            logger.info("[OK] OpenWakeWord installation test passed")
            logger.info(f"Available models: {list(test_model.models.keys())}")
            
            return True
            
        except ImportError as e:
            logger.error(f"[ERROR] OpenWakeWord not installed: {e}")
            logger.error("Install with: pip install openwakeword>=0.6.0")
            return False
        except Exception as e:
            logger.error(f"[ERROR] OpenWakeWord installation test failed: {e}")
            
            # Provide helpful troubleshooting info
            logger.error("")
            logger.error("[HELP] Troubleshooting steps:")
            logger.error("1. Check internet connectivity for model download")
            logger.error("2. Try: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
            logger.error("3. Check if you have write permissions in the package directory")
            logger.error("")
            
            return False