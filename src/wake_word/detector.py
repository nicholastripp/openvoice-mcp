"""
OpenWakeWord wake word detection implementation
"""
import asyncio
import numpy as np
import threading
from typing import Callable, Optional, Any, Dict
from queue import Queue, Empty
import time
import signal

from config import WakeWordConfig
from utils.logger import get_logger
from utils.audio_diagnostics import AudioDiagnostics

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
        self.model_lock = threading.Lock()  # Protect model access
        
        # Model warm-up state
        self.model_ready = False
        self.warmup_chunks_required = 10  # Based on test results showing predictions start after 5-6 chunks
        self.warmup_chunks_processed = 0
        self._model_ready_logged = False  # Track when we log model ready status
        
        # Detection parameters
        self.model_name = config.model
        self.sensitivity = config.sensitivity
        self.vad_enabled = config.vad_enabled
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_cooldown = config.cooldown
        
        # Detection stability - require consecutive chunks above threshold
        self.detection_stability_count = 1  # Require N consecutive chunks above threshold (reduced for testing)
        self.recent_confidences = []  # Track recent confidence levels
        
        # Model state management to prevent stuck predictions
        self.predictions_history = []  # Track recent predictions to detect stuck state
        self.stuck_detection_threshold = 3  # Number of identical predictions to trigger reset (reduced for faster detection)
        self.last_model_reset_time = 0
        self.model_reset_interval = 300.0  # Reset model every 5 minutes (reduced for better responsiveness)
        self.min_reset_cooldown = 2.0  # Minimum time between resets (reduced to 2s for faster recovery)
        self.chunks_since_reset = 0
        self.reset_on_stuck = True  # Enable automatic reset when stuck state detected
        self.stuck_confidence_threshold = 0.0  # Check for stuck state at any confidence level
        
        # Thread safety for model access
        self.model_lock = threading.Lock()  # Protect model access across threads
        self.known_stuck_values = [0.0]  # Known stuck values to detect (removed false positive)
        self.stuck_value_tolerance = 1e-8  # Tolerance for floating point comparison
        
        # Confidence monitoring for reliability analysis
        self.confidence_history = []
        self.confidence_window_size = 50  # Track last 50 predictions
        self.avg_confidence = 0.0
        self.peak_confidence = 0.0
        
        # Prediction failure tracking (no more ThreadPoolExecutor)
        self.failed_predictions = 0
        self.max_failed_predictions = 3  # Reset model after 3 consecutive failures
        self.last_successful_prediction = time.time()
        self.prediction_timeout = 2.0  # Direct timeout check (no threading)
        
        # Performance optimization for Raspberry Pi
        self.skip_predictions = False  # Skip predictions if queue is backing up
        self.max_queue_size = 5  # Max queue size before skipping
    
    async def start(self) -> None:
        """Start wake word detection"""
        if self.is_running:
            self.logger.warning("Wake word detector already running")
            return
        
        try:
            # Run audio system diagnostics
            self.logger.info("Running audio system diagnostics...")
            diagnostics = AudioDiagnostics()
            diagnostic_results = diagnostics.validate_system_audio_config()
            
            # Log critical audio configuration issues
            if diagnostic_results['recommendations']:
                self.logger.warning("Audio configuration issues detected:")
                for rec in diagnostic_results['recommendations']:
                    self.logger.warning(f"  - {rec}")
            else:
                self.logger.info("Audio configuration validation passed")
            
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
        
        # Clear audio buffer and reset state tracking
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer = np.array([], dtype=np.float32)
        
        # Reset state tracking
        self.predictions_history = []
        self.chunks_since_reset = 0
        self.last_model_reset_time = 0
        self.recent_confidences = []  # Reset detection stability tracking
        
        # No longer need to shutdown executor (removed ThreadPoolExecutor)
        
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
            
            # Convert to float32 and normalize to [-1.0, 1.0] range (OpenWakeWord requirement)
            # FIX: Use symmetric normalization to avoid bias
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Remove DC bias for better signal quality
            audio_float = audio_float - np.mean(audio_float)
            
            # Validate audio format
            if not self._validate_audio_chunk(audio_float):
                self.logger.debug("Invalid audio chunk skipped")
                return
            
            # Calculate audio level for debugging
            audio_level = np.max(np.abs(audio_float))
            
            # Resample to 16kHz if needed (OpenWakeWord requirement)
            if input_sample_rate != self.sample_rate:
                from scipy import signal
                # Calculate new length for resampling
                original_length = len(audio_float)
                new_length = int(len(audio_float) * self.sample_rate / input_sample_rate)
                
                # CRITICAL: Log resampling details
                self.logger.info(f"RESAMPLING: {input_sample_rate}Hz -> {self.sample_rate}Hz, samples: {original_length} -> {new_length}")
                print(f"   DETECTOR: RESAMPLING {input_sample_rate}Hz -> {self.sample_rate}Hz, samples: {original_length} -> {new_length}")
                
                # FIX: Use high-quality polyphase resampling with proper anti-aliasing
                # This preserves signal quality much better than basic FFT resampling
                up_factor = self.sample_rate
                down_factor = input_sample_rate
                
                # Find GCD for efficient resampling
                import math
                gcd = math.gcd(up_factor, down_factor)
                up_factor //= gcd
                down_factor //= gcd
                
                # Apply high-quality polyphase resampling
                audio_float = signal.resample_poly(audio_float, up_factor, down_factor, window=('kaiser', 8.0))
                
                # Verify resampling worked
                if abs(len(audio_float) - new_length) > 1:  # Allow 1 sample tolerance
                    self.logger.error(f"RESAMPLING ERROR: Expected ~{new_length} samples, got {len(audio_float)}")
                
                # Check if resampling destroyed the signal
                post_resample_level = np.max(np.abs(audio_float))
                post_resample_rms = np.sqrt(np.mean(audio_float ** 2))
                self.logger.info(f"Post-resample: level={post_resample_level:.6f}, RMS={post_resample_rms:.6f} (was {audio_level:.6f})")
                
                self.logger.debug(f"High-quality resampled audio from {input_sample_rate}Hz to {self.sample_rate}Hz: {original_length} -> {len(audio_float)} samples (level: {audio_level:.3f} -> {post_resample_level:.3f})")
            
            # DISABLED NORMALIZATION FOR DEBUGGING
            # The aggressive normalization might be causing the model to get stuck
            # Let's test with raw audio first
            pre_amp_level = np.max(np.abs(audio_float))
            pre_amp_rms = np.sqrt(np.mean(audio_float ** 2))
            
            # FIX: Implement proper RMS-based gain control  
            # Balance between wake word detection and OpenAI VAD needs
            target_rms = 0.3  # Reduced for better performance while maintaining detection
            
            if pre_amp_rms > 0.001:  # Avoid division by zero
                gain = min(10.0, target_rms / pre_amp_rms)  # Reasonable gain limit
                # Apply gain with soft clipping to avoid distortion
                audio_float = np.tanh(audio_float * gain).astype(np.float32)
                
                # Performance optimization: Removed noise gate (too CPU intensive)
                # The gain control and pre-emphasis provide sufficient enhancement
                
                self.logger.debug(f"Applied gain of {gain:.2f}x (RMS: {pre_amp_rms:.6f} -> target: {target_rms:.2f})")
            else:
                gain = 1.0  # No gain for silence
            
            # Pre-emphasis filter removed for performance optimization
            # The gain control provides sufficient enhancement for wake word detection
            
            post_amp_level = np.max(np.abs(audio_float))
            post_amp_rms = np.sqrt(np.mean(audio_float ** 2))
            
            # Debug logging for amplification with enhanced metrics
            if hasattr(self, '_amp_debug_counter'):
                self._amp_debug_counter += 1
            else:
                self._amp_debug_counter = 1
                
            if self._amp_debug_counter % 500 == 0:  # Every 500 chunks (reduced for production)
                self.logger.debug(f"Audio stats: RMS {pre_amp_rms:.6f} -> {post_amp_rms:.6f} (gain: {gain:.2f}x)")
                print(f"   DETECTOR: AUDIO STATS: RMS {pre_amp_rms:.6f} -> {post_amp_rms:.6f} (gain: {gain:.2f}x)")
            
            # Immediate debug feedback (not just debug logs)
            print(f"   DETECTOR: audio_level={audio_level:.3f}, samples={len(audio_float)}", flush=True)
            
            # Lower threshold to allow speech audio (was 0.005)
            if audio_level > 0.001:  # Much lower threshold for legitimate audio activity
                # Add to buffer instead of directly queuing
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
                
                # Check if we have enough samples for OpenWakeWord (80ms = 1280 samples at 16kHz)
                while len(self.audio_buffer) >= self.target_chunk_size:
                    # Extract exactly 1280 samples
                    chunk_for_oww = self.audio_buffer[:self.target_chunk_size]
                    self.audio_buffer = self.audio_buffer[self.target_chunk_size:]
                    
                    # Validate chunk before queuing
                    if len(chunk_for_oww) == self.target_chunk_size and chunk_for_oww.dtype == np.float32:
                        # Check for valid audio range
                        if np.all(np.isfinite(chunk_for_oww)) and np.max(np.abs(chunk_for_oww)) <= 1.0:
                            # Queue the properly-sized chunk for OpenWakeWord
                            self.audio_queue.put(chunk_for_oww, block=False)
                            print(f"   DETECTOR: QUEUED 80ms chunk for OpenWakeWord (samples={len(chunk_for_oww)}, buffer_remaining={len(self.audio_buffer)}, queue_size={self.audio_queue.qsize()})", flush=True)
                            self.logger.debug(f"Queued 80ms chunk for processing: samples={len(chunk_for_oww)}, buffer_remaining={len(self.audio_buffer)}, queue_size={self.audio_queue.qsize()}")
                        else:
                            self.logger.warning(f"Skipping invalid audio chunk: infinite/out-of-range values detected")
                    else:
                        self.logger.warning(f"Skipping invalid chunk: size={len(chunk_for_oww)}, expected={self.target_chunk_size}")
                
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
                # Check file size to ensure it's not corrupted
                file_size = model_path.stat().st_size
                self.logger.info(f"Model file found: {model_path} (size: {file_size} bytes)")
                
                if file_size < 10000:  # TFLite models should be at least 10KB
                    self.logger.error(f"Model file appears corrupted - too small: {file_size} bytes")
                    print(f"*** ERROR: WAKE WORD MODEL FILE TOO SMALL: {file_size} bytes ***")
                    return False
                    
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
            
            # Initialize model with noise suppression for better performance
            model_kwargs = {
                'enable_speex_noise_suppression': True  # Enable noise suppression for better detection
            }
            
            # Enable noise suppression to improve detection rates
            self.logger.info("[INFO] Initializing OpenWakeWord with Speex noise suppression enabled")
            self.logger.info("[INFO] Noise suppression should improve both false-reject and false-accept rates")
            
            # Force reinitialize the model to clear any cached state
            if hasattr(self, 'model') and self.model:
                try:
                    del self.model
                    self.logger.info("[INFO] Cleared existing model instance")
                except:
                    pass
            
            with self.model_lock:
                self.model = WakeWordModel(
                    wakeword_models=[actual_model_name],
                    **model_kwargs
                )
            
            self.logger.info(f"[OK] Successfully loaded model: {actual_model_name}")
            self.logger.info(f"Available models: {list(self.model.models.keys())}")
            self.logger.info(f"Model expects: {self.sample_rate}Hz audio in {self.chunk_size} sample chunks (80ms)")
            self.logger.info(f"Model configuration: VAD={self.vad_enabled}, sensitivity={self.sensitivity}")
            
            # NEW: Model warm-up phase
            # Based on test results, OpenWakeWord needs 5-6 predictions before producing non-zero values
            # We'll warm up with 10 varied audio chunks to ensure proper initialization
            self.logger.info("[INFO] Starting model warm-up phase...")
            self._warmup_model()
            
            # Test prediction after warm-up
            try:
                test_chunk = np.random.normal(0, 0.05, self.chunk_size).astype(np.float32)
                with self.model_lock:
                    test_predictions = self.model.predict(test_chunk)
                max_conf = max(test_predictions.values()) if test_predictions else 0.0
                self.logger.info(f"[TEST] Post-warmup test prediction: {test_predictions} (max: {max_conf:.8f})")
                
                # CRITICAL: Check if model is stuck at 5.0768717e-06 immediately
                if any(abs(conf - 5.0768717e-06) < 1e-10 for conf in test_predictions.values()):
                    self.logger.error(f"[CRITICAL] Model '{actual_model_name}' is returning stuck value 5.0768717e-06 immediately after load!")
                    self.logger.error("[CRITICAL] This indicates the model file may be corrupted. Try using 'alexa' instead of 'hey_jarvis'")
                    print("*** CRITICAL: WAKE WORD MODEL IS CORRUPTED - STUCK AT 5.0768717e-06 ***")
                    print("*** PLEASE CHANGE config.yaml TO USE 'alexa' INSTEAD OF 'hey_jarvis' ***")
                
                if max_conf > 0.0:
                    self.logger.info("[OK] Model warm-up successful - producing non-zero predictions")
                else:
                    self.logger.warning("[WARNING] Model still returning zero after warm-up")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Post-warmup test failed: {e}")
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
    
    def _warmup_model(self) -> None:
        """
        Warm up the OpenWakeWord model with varied audio chunks
        
        Based on test results, OpenWakeWord needs several predictions before
        producing non-zero values. This method feeds varied audio to the model
        to ensure it's properly initialized.
        """
        if not self.model:
            return
            
        self.logger.info(f"[WARMUP] Processing {self.warmup_chunks_required} warm-up chunks...")
        
        warmup_predictions = []
        stuck_value_count = 0
        last_prediction = None
        
        for i in range(self.warmup_chunks_required):
            # Generate varied audio for warm-up
            if i < 3:
                # Start with silence
                audio_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                chunk_type = "silence"
            elif i < 6:
                # Low noise
                audio_chunk = np.random.normal(0, 0.01, self.chunk_size).astype(np.float32)
                chunk_type = "low noise"
            elif i < 8:
                # Medium noise
                audio_chunk = np.random.normal(0, 0.05, self.chunk_size).astype(np.float32)
                chunk_type = "medium noise"
            else:
                # High noise with some variation
                base_noise = np.random.normal(0, 0.1, self.chunk_size)
                # Add some sine wave variation to ensure different predictions
                t = np.linspace(0, 1, self.chunk_size)
                sine_component = 0.05 * np.sin(2 * np.pi * (i - 8) * t)
                audio_chunk = (base_noise + sine_component).astype(np.float32)
                chunk_type = "high noise + variation"
            
            try:
                with self.model_lock:
                    predictions = self.model.predict(audio_chunk)
                max_conf = max(predictions.values()) if predictions else 0.0
                warmup_predictions.append(max_conf)
                
                # Check for stuck value during warmup
                if last_prediction is not None and abs(max_conf - last_prediction) < 1e-10:
                    stuck_value_count += 1
                    if stuck_value_count >= 3:
                        self.logger.warning(f"[WARMUP] Model appears stuck at {max_conf:.8e} during warmup")
                        # Try to unstick by feeding very different audio
                        unstick_audio = np.random.uniform(-0.5, 0.5, self.chunk_size).astype(np.float32)
                        with self.model_lock:
                            _ = self.model.predict(unstick_audio)
                        stuck_value_count = 0
                else:
                    stuck_value_count = 0
                
                last_prediction = max_conf
                
                if i % 3 == 0 or max_conf > 0.0:
                    self.logger.debug(f"[WARMUP] Chunk {i+1}/{self.warmup_chunks_required} ({chunk_type}): max={max_conf:.8f}")
                
                # Check if we're getting non-zero predictions
                if max_conf > 0.0 and max_conf != 5.0768717e-06:  # Exclude known stuck value
                    self.logger.info(f"[WARMUP] Model started producing valid predictions at chunk {i+1}")
                    
            except Exception as e:
                self.logger.warning(f"[WARMUP] Failed to process chunk {i+1}: {e}")
        
        # Check warm-up results
        non_zero_count = sum(1 for p in warmup_predictions if p > 0.0 and p != 5.0768717e-06)
        max_prediction = max(warmup_predictions) if warmup_predictions else 0.0
        unique_predictions = len(set(warmup_predictions))
        
        self.logger.info(f"[WARMUP] Complete: {non_zero_count}/{self.warmup_chunks_required} valid predictions, max={max_prediction:.8f}, unique values={unique_predictions}")
        
        # Warn if model seems stuck
        if unique_predictions <= 2:
            self.logger.warning(f"[WARMUP] Model may be stuck - only {unique_predictions} unique prediction values")
        
        # Mark model as ready
        self.model_ready = True
        self.warmup_chunks_processed = self.warmup_chunks_required
        self._model_ready_logged = False  # Reset ready logging flag
    
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
                
                # CRITICAL: Verify chunk from queue
                if len(audio_chunk) != 1280:
                    self.logger.error(f"CRITICAL: Got wrong chunk size from queue: {len(audio_chunk)} samples (expected 1280)")
                    print(f"   DETECTOR: CRITICAL ERROR - Wrong chunk size from queue: {len(audio_chunk)}")
                    continue
                
                chunks_processed += 1
                
                # Immediate debug feedback for OpenWakeWord processing
                if chunks_processed % 50 == 0:  # Every 50 chunks
                    chunk_length_ms = len(audio_chunk) / self.sample_rate * 1000
                    print(f"   DETECTOR: OpenWakeWord processing chunk #{chunks_processed}, samples={len(audio_chunk)}, duration={chunk_length_ms:.1f}ms, queue_size={self.audio_queue.qsize()}")
                
                # Debug: Log queue processing periodically
                if chunks_processed % 100 == 0:
                    self.logger.debug(f"Detection loop processed {chunks_processed} chunks, queue size: {self.audio_queue.qsize()}")
                
                # Process with OpenWakeWord with comprehensive state management
                try:
                    # Check if model needs reset (preventive or stuck state)
                    if self._should_reset_model():
                        # Determine reset reason
                        current_time = time.time()
                        time_since_last_reset = current_time - self.last_model_reset_time
                        if self._is_stuck_state():
                            self.logger.warning(f"Triggering reset due to stuck state")
                            self._reset_model_state("stuck_state")
                        elif time_since_last_reset > self.model_reset_interval:
                            self._reset_model_state("scheduled_interval")
                        else:
                            self._reset_model_state("preventive")
                    else:
                        # Log why reset was not triggered (every 50 chunks)
                        if chunks_processed % 50 == 0 and self._is_stuck_state():
                            current_time = time.time()
                            time_since_last_reset = current_time - self.last_model_reset_time
                            self.logger.warning(f"Model stuck but reset blocked - time since last: {time_since_last_reset:.1f}s, cooldown: {self.min_reset_cooldown}s")
                    
                    # Check if we should skip predictions due to queue backup
                    current_queue_size = self.audio_queue.qsize()
                    if current_queue_size > self.max_queue_size:
                        if chunks_processed % 2 == 1:  # Skip every other chunk when backed up
                            print(f"   DETECTOR: Skipping prediction due to queue backup ({current_queue_size} items)")
                            continue
                    
                    # Enhanced logging for debugging stuck model
                    chunk_stats = {
                        'min': float(np.min(audio_chunk)),
                        'max': float(np.max(audio_chunk)),
                        'mean': float(np.mean(audio_chunk)),
                        'std': float(np.std(audio_chunk)),
                        'rms': float(np.sqrt(np.mean(audio_chunk ** 2)))
                    }
                    
                    # CRITICAL: Validate audio format for OpenWakeWord
                    expected_samples = 1280  # 80ms at 16kHz
                    if len(audio_chunk) != expected_samples:
                        self.logger.error(f"INVALID CHUNK SIZE: {len(audio_chunk)} samples, expected {expected_samples}")
                        print(f"   DETECTOR: ERROR - INVALID CHUNK SIZE: {len(audio_chunk)} != {expected_samples}")
                    
                    if audio_chunk.dtype != np.float32:
                        self.logger.error(f"INVALID DTYPE: {audio_chunk.dtype}, expected float32")
                        print(f"   DETECTOR: ERROR - INVALID DTYPE: {audio_chunk.dtype}")
                    
                    # Check if audio is in correct range [-1.0, 1.0]
                    if chunk_stats['max'] > 1.0 or chunk_stats['min'] < -1.0:
                        self.logger.warning(f"Audio out of range: [{chunk_stats['min']:.3f}, {chunk_stats['max']:.3f}]")
                        print(f"   DETECTOR: WARNING - Audio exceeds [-1, 1] range")
                    
                    # Log exact format being sent to model (every 100th chunk)
                    if chunks_processed % 100 == 0:
                        self.logger.info(f"AUDIO FORMAT CHECK: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}, range=[{chunk_stats['min']:.6f}, {chunk_stats['max']:.6f}], RMS={chunk_stats['rms']:.6f}")
                        print(f"   DETECTOR: AUDIO FORMAT: {audio_chunk.shape} {audio_chunk.dtype} [{chunk_stats['min']:.6f}, {chunk_stats['max']:.6f}] RMS={chunk_stats['rms']:.6f}")
                    
                    print(f"   DETECTOR: Calling model.predict() with chunk: samples={len(audio_chunk)}, dtype={audio_chunk.dtype}")
                    print(f"   DETECTOR: Chunk stats: min={chunk_stats['min']:.6f}, max={chunk_stats['max']:.6f}, mean={chunk_stats['mean']:.6f}, std={chunk_stats['std']:.6f}, rms={chunk_stats['rms']:.6f}")
                    
                    # Direct prediction with simple timeout check
                    start_time = time.time()
                    predictions = None
                    
                    try:
                        with self.model_lock:
                            predictions = self.model.predict(audio_chunk)
                            self.last_successful_prediction = time.time()
                            self.failed_predictions = 0  # Reset on success
                    except Exception as e:
                        self.logger.error(f"Model prediction failed: {e}")
                        self.failed_predictions += 1
                        predictions = None
                    
                    # Check if prediction took too long
                    prediction_time = time.time() - start_time
                    if prediction_time > self.prediction_timeout:
                        self.logger.warning(f"Model prediction slow: {prediction_time:.2f}s")
                    
                    if predictions is None:
                        print(f"   DETECTOR: model.predict() FAILED")
                        
                        # Reset model if too many consecutive failures
                        if self.failed_predictions >= self.max_failed_predictions:
                            print(f"   DETECTOR: Model failed {self.failed_predictions} times, resetting")
                            self.logger.error(f"Model failed {self.failed_predictions} consecutive times, resetting")
                            self._reset_model_state("prediction_failure")
                            self.failed_predictions = 0
                        continue
                    
                    print(f"   DETECTOR: model.predict() returned: {predictions}")
                    
                    # Log if we're getting the stuck value
                    if predictions and any(abs(conf - 5.0768717e-06) < 1e-10 for conf in predictions.values()):
                        print(f"   DETECTOR: WARNING: Model returned known stuck value 5.0768717e-06")
                        self.logger.warning(f"Model returned stuck value with chunk stats: {chunk_stats}")
                    
                    # Track predictions for stuck state detection
                    self._track_prediction(predictions)
                    self.chunks_since_reset += 1
                    
                    # Track confidence for monitoring
                    self._track_confidence(predictions)
                    
                    # Immediate check for known stuck value with tolerance
                    if predictions:
                        first_model = list(predictions.keys())[0]
                        confidence = predictions[first_model]
                        # Check for the specific stuck value 5.0768717e-06
                        if abs(confidence - 5.0768717e-06) < 1e-10:
                            self.logger.error(f"CRITICAL: Model stuck at known bad value {confidence:.8e}")
                            print(f"   DETECTOR: CRITICAL STUCK VALUE 5.0768717e-06 DETECTED! Forcing immediate reset NOW", flush=True)
                            # Force immediate reset - bypass ALL checks
                            self.last_model_reset_time = 0  # Force bypass cooldown
                            self._reset_model_state("critical_stuck_value")
                            continue
                        # Use tolerance-based comparison for other floating point stuck values
                        for stuck_value in self.known_stuck_values:
                            if abs(confidence - stuck_value) < self.stuck_value_tolerance:
                                self.logger.warning(f"Detected known stuck value {confidence:.8e} (matches {stuck_value:.8e}) at chunk {chunks_processed}, chunks_since_reset={self.chunks_since_reset}")
                                print(f"   DETECTOR: KNOWN STUCK VALUE DETECTED! {confidence:.8e} ~= {stuck_value:.8e}, forcing immediate reset", flush=True)
                                # Force immediate reset for known stuck value - no delay
                                self.last_model_reset_time = 0  # Force bypass cooldown
                                self._reset_model_state("known_stuck_value")
                                continue
                    
                except Exception as e:
                    print(f"   DETECTOR: ERROR in model.predict(): {e}")
                    self.logger.error(f"OpenWakeWord prediction error: {e}")
                    # Reset model on error
                    self._reset_model_state("prediction_error")
                    continue
                
                # Enhanced prediction logging with stuck state detection
                if predictions:
                    max_confidence = max(predictions.values()) if predictions else 0.0
                    
                    # Check if we're in stuck state and log it
                    stuck_indicator = " [STUCK]" if self._is_stuck_state() else ""
                    reset_indicator = f" [RESET #{self.chunks_since_reset}]" if self.chunks_since_reset <= 3 else ""
                    
                    print(f"   DETECTOR: OpenWakeWord predictions: {predictions} (max: {max_confidence:.3f}){stuck_indicator}{reset_indicator}")
                    
                    # Show prediction scores periodically and for any significant activity
                    if chunks_processed % 100 == 0 or max_confidence > 0.05:
                        formatted_predictions = {k: f"{v:.8f}" for k, v in predictions.items()}  # More precision
                        print(f"   DETECTOR: Detailed predictions: {formatted_predictions} (max: {max_confidence:.8f}){stuck_indicator}")
                    
                    # Log significant predictions or stuck state warnings
                    if max_confidence > 0.1:  # Significant predictions
                        self.logger.debug(f"OpenWakeWord significant prediction: {predictions}")
                    elif self._is_stuck_state() and chunks_processed % 25 == 0:  # Stuck state warnings
                        self.logger.warning(f"Model stuck state detected at chunk {chunks_processed}: {predictions}")
                        
                else:
                    print(f"   DETECTOR: OpenWakeWord returned EMPTY predictions!")
                    if chunks_processed % 50 == 0:  # Log empty predictions periodically
                        self.logger.warning(f"OpenWakeWord predictions are empty at chunk {chunks_processed}")
                    continue
                
                # Check for wake word detection (only if model is ready)
                if not self.model_ready:
                    if chunks_processed % 50 == 0:
                        print(f"   DETECTOR: Model still warming up, skipping detection check")
                    continue
                
                # Debug: Log when model becomes ready for detection
                if not getattr(self, '_model_ready_logged', True):
                    print(f"   DETECTOR: MODEL NOW READY FOR DETECTION after {chunks_processed} chunks")
                    self.logger.info(f"Model ready for detection after {chunks_processed} processed chunks")
                    self._model_ready_logged = True
                
                for model_name, confidence in predictions.items():
                    # Track confidence for stability detection
                    self.recent_confidences.append(confidence)
                    
                    # Keep only recent confidences for stability check
                    max_history = self.detection_stability_count * 2
                    if len(self.recent_confidences) > max_history:
                        self.recent_confidences.pop(0)
                    
                    # DEBUG: Log all confidence levels with enhanced threshold comparison
                    threshold_ratio = confidence / self.sensitivity if self.sensitivity > 0 else 0
                    print(f"   DETECTOR: {model_name} confidence {confidence:.6f} (threshold: {self.sensitivity:.6f}, ratio: {threshold_ratio:.1f}x)")
                    
                    # Track confidence for monitoring (helps tune threshold)
                    if chunks_processed % 100 == 0 and self.avg_confidence > 0:
                        self.logger.info(f"Confidence monitoring - Avg: {self.avg_confidence:.6f}, Peak: {self.peak_confidence:.6f}, Threshold: {self.sensitivity:.6f}")
                    
                    # Enhanced threshold check with detailed logging
                    if confidence >= self.sensitivity:
                        current_time = time.time()
                        
                        print(f"   DETECTOR: DETECTION CANDIDATE: {model_name} confidence {confidence:.6f} >= {self.sensitivity:.6f}")
                        self.logger.info(f"Detection candidate: {model_name} confidence {confidence:.6f} >= threshold {self.sensitivity:.6f}")
                        
                        # Additional validation: require minimum confidence level for reliability
                        min_reliable_confidence = self.sensitivity * 1.2  # 1.2x threshold for reliability
                        if confidence < min_reliable_confidence:
                            print(f"   DETECTOR: CONFIDENCE TOO LOW: {confidence:.6f} < {min_reliable_confidence:.6f} (need {min_reliable_confidence/confidence:.1f}x more for reliability)")
                            self.logger.debug(f"Wake word confidence too low for reliability: {confidence:.6f} < {min_reliable_confidence:.6f}")
                            continue
                        
                        print(f"   DETECTOR: CONFIDENCE RELIABLE: {confidence:.6f} >= {min_reliable_confidence:.6f}")
                        
                        # Check detection stability - require consecutive chunks above threshold
                        recent_above_threshold = sum(1 for c in self.recent_confidences[-self.detection_stability_count:] if c >= self.sensitivity)
                        stable_detection = recent_above_threshold >= self.detection_stability_count
                        
                        print(f"   DETECTOR: Stability check: {recent_above_threshold}/{self.detection_stability_count} consecutive chunks above threshold")
                        print(f"   DETECTOR: Recent confidences: {[f'{c:.6f}' for c in self.recent_confidences[-5:]]}")  # Show last 5
                        
                        if not stable_detection:
                            print(f"   DETECTOR: Detection not stable: {recent_above_threshold}/{self.detection_stability_count} consecutive chunks above threshold")
                            self.logger.debug(f"Detection not stable: {recent_above_threshold}/{self.detection_stability_count} chunks above threshold")
                            continue
                        
                        print(f"   DETECTOR: STABLE DETECTION: {recent_above_threshold}/{self.detection_stability_count} consecutive chunks above threshold")
                        
                        # Check cooldown to prevent rapid re-triggers
                        time_since_last = current_time - self.last_detection_time
                        print(f"   DETECTOR: Cooldown check: {time_since_last:.1f}s since last detection (cooldown: {self.detection_cooldown}s)")
                        
                        if time_since_last >= self.detection_cooldown:
                            print(f"   DETECTOR: WAKE WORD DETECTED! {model_name} (confidence: {confidence:.6f})")
                            self.logger.info(f"Wake word detected: {model_name} (confidence: {confidence:.6f})")
                            self.logger.debug(f"Cooldown passed: {time_since_last:.1f}s >= {self.detection_cooldown}s")
                            self.last_detection_time = current_time
                            
                            # Clear confidence history after successful detection
                            self.recent_confidences = []
                            
                            # Call detection callbacks with enhanced logging
                            print(f"   DETECTOR: Calling {len(self.detection_callbacks)} detection callbacks...")
                            for i, callback in enumerate(self.detection_callbacks):
                                try:
                                    print(f"   DETECTOR: Calling callback {i+1}/{len(self.detection_callbacks)}: {callback}")
                                    callback(model_name, confidence)
                                    print(f"   DETECTOR: Callback {i+1} completed successfully")
                                except Exception as e:
                                    print(f"   DETECTOR: ERROR in callback {i+1}: {e}")
                                    self.logger.error(f"Error in detection callback {i+1}: {e}")
                        else:
                            print(f"   DETECTOR: WAKE WORD BLOCKED BY COOLDOWN: {model_name} (confidence: {confidence:.6f}), {time_since_last:.1f}s < {self.detection_cooldown}s")
                            self.logger.warning(f"Wake word detected but BLOCKED by cooldown: {model_name} (confidence: {confidence:.6f}), {time_since_last:.1f}s < {self.detection_cooldown}s")
                            # This is likely why subsequent wake words aren't working!
                    else:
                        # DEBUG: Log why detection didn't trigger
                        if confidence > self.sensitivity * 0.1:  # Log if close to threshold
                            print(f"   DETECTOR: Below threshold: {model_name} confidence {confidence:.6f} < {self.sensitivity:.6f} (need {self.sensitivity/confidence:.1f}x more)")
                        elif chunks_processed % 50 == 0:  # Periodic logging for very low confidence
                            print(f"   DETECTOR: Low confidence: {model_name} {confidence:.6f} (threshold: {self.sensitivity:.6f})")
                
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
        self._reset_model_state("manual_reset")
    
    def _validate_audio_chunk(self, audio_float: np.ndarray) -> bool:
        """
        Validate audio chunk format and quality
        
        Args:
            audio_float: Audio data as float32 array
            
        Returns:
            True if audio chunk is valid, False otherwise
        """
        # Check data type
        if audio_float.dtype != np.float32:
            self.logger.warning(f"Invalid audio dtype: {audio_float.dtype}, expected float32")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(audio_float)) or np.any(np.isinf(audio_float)):
            self.logger.warning("Audio contains NaN or infinite values")
            return False
        
        # Check range (should be [-1.0, 1.0] for OpenWakeWord)
        if np.any(np.abs(audio_float) > 1.0):
            self.logger.warning(f"Audio values out of range: max={np.max(np.abs(audio_float)):.3f}, expected <= 1.0")
            # Don't reject, just warn and clip
            np.clip(audio_float, -1.0, 1.0, out=audio_float)
        
        return True
    
    def _should_reset_model(self) -> bool:
        """
        Check if model should be reset based on time or stuck state
        
        Returns:
            True if model should be reset, False otherwise
        """
        current_time = time.time()
        time_since_last_reset = current_time - self.last_model_reset_time
        
        # Check if stuck state detected FIRST (bypass cooldown for stuck states)
        if self.reset_on_stuck and self._is_stuck_state():
            self.logger.info(f"Stuck state detected, bypassing cooldown (time since last reset: {time_since_last_reset:.1f}s)")
            return True
        
        # Enforce minimum cooldown between resets for non-stuck states
        if time_since_last_reset < self.min_reset_cooldown:
            self.logger.debug(f"Reset blocked by cooldown: {time_since_last_reset:.1f}s < {self.min_reset_cooldown}s")
            return False
        
        # Preventive reset based on time interval (extended from 5 to 10 minutes)
        if time_since_last_reset > self.model_reset_interval:
            self.logger.debug(f"Model reset due to time interval ({self.model_reset_interval}s)")
            return True
        
        return False
    
    def _is_stuck_state(self) -> bool:
        """
        Check if model is in stuck state (identical predictions)
        
        Returns:
            True if model appears to be stuck, False otherwise
        """
        if len(self.predictions_history) < self.stuck_detection_threshold:
            return False
        
        # Check if last N predictions are identical
        recent_predictions = self.predictions_history[-self.stuck_detection_threshold:]
        
        # Extract the first model's prediction value for comparison
        if not recent_predictions[0]:
            return False
        
        first_model_name = list(recent_predictions[0].keys())[0]
        first_value = recent_predictions[0][first_model_name]
        
        # Special handling for the critical stuck value 5.0768717e-06
        if abs(first_value - 5.0768717e-06) < 1e-10:
            # For this specific stuck value, trigger reset after just 2 occurrences
            stuck_threshold = 2
            recent_predictions = self.predictions_history[-stuck_threshold:] if len(self.predictions_history) >= stuck_threshold else self.predictions_history
            
            stuck_count = sum(1 for pred in recent_predictions 
                            if pred and first_model_name in pred 
                            and abs(pred[first_model_name] - 5.0768717e-06) < 1e-10)
            
            if stuck_count >= stuck_threshold:
                self.logger.error(f"CRITICAL: Model stuck at 5.0768717e-06 for {stuck_count} predictions")
                return True
        
        # Check for other stuck values with tolerance
        for stuck_value in self.known_stuck_values:
            if abs(first_value - stuck_value) < self.stuck_value_tolerance:
                # If we see a known stuck value, only need fewer consecutive to trigger reset
                stuck_threshold = min(3, self.stuck_detection_threshold)
                recent_predictions = self.predictions_history[-stuck_threshold:]
                
                # Count how many are the same stuck value (with tolerance)
                stuck_count = sum(1 for pred in recent_predictions 
                                if pred and first_model_name in pred 
                                and abs(pred[first_model_name] - stuck_value) < self.stuck_value_tolerance)
                
                if stuck_count >= stuck_threshold:
                    # Only log once when we first detect stuck state
                    if not hasattr(self, '_last_stuck_log_value') or self._last_stuck_log_value is None or abs(self._last_stuck_log_value - first_value) > self.stuck_value_tolerance:
                        self.logger.warning(f"Model stuck: {stuck_count} identical predictions of {first_value:.8e} (~= {stuck_value:.8e})")
                        self._last_stuck_log_value = first_value
                    return True
        
        # Check variance in recent predictions to detect stuck state
        values = [pred[first_model_name] for pred in recent_predictions if pred and first_model_name in pred]
        if len(values) < self.stuck_detection_threshold:
            return False
            
        # Calculate variance
        variance = np.var(values)
        
        # If variance is extremely low, model is stuck
        if variance < 1e-12:  # Essentially zero variance
            is_stuck = True
            if not hasattr(self, '_last_stuck_log_value') or self._last_stuck_log_value is None or abs(self._last_stuck_log_value - first_value) > 1e-10:
                self.logger.warning(f"Model stuck: Zero variance in {len(values)} predictions (all {first_value:.8e})")
                self._last_stuck_log_value = first_value
        else:
            is_stuck = False
            # Clear the last logged value if no longer stuck
            if hasattr(self, '_last_stuck_log_value'):
                self._last_stuck_log_value = None
                
        return is_stuck
    
    def _track_prediction(self, predictions: dict) -> None:
        """
        Track prediction for stuck state detection
        
        Args:
            predictions: Dictionary of model predictions
        """
        # Keep only recent predictions in history
        max_history = self.stuck_detection_threshold * 2
        self.predictions_history.append(predictions.copy() if predictions else {})
        
        if len(self.predictions_history) > max_history:
            self.predictions_history.pop(0)
    
    def _track_confidence(self, predictions: Dict[str, float]) -> None:
        """Track confidence levels for monitoring and analysis"""
        if not predictions:
            return
        
        # Get the maximum confidence value
        max_confidence = max(predictions.values())
        
        # Add to confidence history
        self.confidence_history.append(max_confidence)
        
        # Keep only recent predictions
        if len(self.confidence_history) > self.confidence_window_size:
            self.confidence_history.pop(0)
        
        # Update statistics
        if self.confidence_history:
            self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            self.peak_confidence = max(self.confidence_history)
        
        # Log confidence statistics periodically
        if len(self.confidence_history) >= self.confidence_window_size and len(self.confidence_history) % 25 == 0:
            recent_high_confidence = sum(1 for c in self.confidence_history[-10:] if c > 0.01)
            self.logger.debug(f"Confidence stats: avg={self.avg_confidence:.6f}, peak={self.peak_confidence:.6f}, recent_high={recent_high_confidence}/10")
    
    def _analyze_audio_quality(self, audio_data: np.ndarray, stage: str) -> Dict[str, float]:
        """
        Analyze audio quality metrics for debugging
        
        Args:
            audio_data: Audio samples as numpy array
            stage: Description of processing stage (e.g., "pre-amp", "post-amp")
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Basic metrics
            rms = np.sqrt(np.mean(audio_data ** 2))
            peak = np.max(np.abs(audio_data))
            mean_abs = np.mean(np.abs(audio_data))
            
            # Signal-to-noise ratio estimation
            # Use a simple noise floor estimation based on quietest 10% of samples
            sorted_abs = np.sort(np.abs(audio_data))
            noise_floor = np.mean(sorted_abs[:len(sorted_abs)//10]) if len(sorted_abs) > 10 else 0.0
            snr = 20 * np.log10(rms / max(noise_floor, 1e-10))  # Avoid division by zero
            
            # Dynamic range
            dynamic_range = 20 * np.log10(peak / max(rms, 1e-10))
            
            # Zero crossing rate (measure of signal complexity)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            zcr = zero_crossings / len(audio_data)
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio_data) >= 0.99) / len(audio_data)
            
            return {
                'rms': rms,
                'peak': peak,
                'mean_abs': mean_abs,
                'snr': snr,
                'dynamic_range': dynamic_range,
                'zero_crossing_rate': zcr,
                'clipping_ratio': clipping_ratio,
                'noise_floor': noise_floor
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio quality at {stage}: {e}")
            return {
                'rms': 0.0, 'peak': 0.0, 'mean_abs': 0.0, 'snr': 0.0,
                'dynamic_range': 0.0, 'zero_crossing_rate': 0.0,
                'clipping_ratio': 0.0, 'noise_floor': 0.0
            }
    
    def _reset_model_state(self, reset_reason: str = "unknown") -> None:
        """
        Reset model state and clear tracking data
        
        Args:
            reset_reason: Reason for the reset (for logging/diagnostics)
        """
        current_time = time.time()
        time_since_last_reset = current_time - self.last_model_reset_time
        
        # Enhanced logging for reset diagnostics
        self.logger.info(f"[RESET] Resetting wake word model - Reason: {reset_reason}")
        self.logger.info(f"[RESET] Performance metrics:")
        self.logger.info(f"[RESET]   - Chunks processed since last reset: {self.chunks_since_reset}")
        self.logger.info(f"[RESET]   - Time since last reset: {time_since_last_reset:.1f}s")
        self.logger.info(f"[RESET]   - Predictions in history: {len(self.predictions_history)}")
        self.logger.info(f"[RESET]   - Recent confidences tracked: {len(self.recent_confidences)}")
        self.logger.info(f"[RESET]   - Failed predictions count: {self.failed_predictions}")
        
        # Log recent prediction pattern if stuck state
        if reset_reason == "stuck_state" and self.predictions_history:
            recent_predictions = self.predictions_history[-min(5, len(self.predictions_history)):]
            self.logger.info(f"[RESET] Recent prediction pattern (last {len(recent_predictions)} predictions):")
            for i, pred in enumerate(recent_predictions):
                if pred:
                    model_name = list(pred.keys())[0]
                    confidence = pred[model_name]
                    self.logger.info(f"[RESET]   {i+1}. {model_name}: {confidence:.8f}")
        
        # For stuck state, known stuck value, or failure, always try complete model reload
        if reset_reason in ["stuck_state", "known_stuck_value", "prediction_failure"] and self.model:
            try:
                self.logger.info(f"[RESET] {reset_reason} - performing complete model reload")
                # Save current model name
                current_model = self.model_name
                # Delete the model completely with lock
                with self.model_lock:
                    del self.model
                    self.model = None
                # Force garbage collection
                import gc
                gc.collect()
                # Small delay to ensure cleanup
                time.sleep(0.1)
                # Reload the model
                self._load_model()
                self.logger.info("[RESET] Complete model reload successful")
            except Exception as e:
                self.logger.error(f"[RESET] Failed to reload model: {e}")
                # Try normal reset as fallback
                if self.model:
                    try:
                        self.model.reset()
                        self.logger.info(f"[RESET] Fallback to model.reset() completed")
                    except Exception as e2:
                        self.logger.warning(f"[RESET] Fallback reset also failed: {e2}")
        elif self.model:
            try:
                with self.model_lock:
                    self.model.reset()
                self.logger.info(f"[RESET] OpenWakeWord model.reset() completed successfully")
            except Exception as e:
                self.logger.warning(f"[RESET] Failed to reset model: {e}")
        
        # Clear internal audio buffer and ensure clean state
        buffer_size = len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.logger.debug(f"[RESET] Cleared audio buffer ({buffer_size} samples)")
        
        # Clear audio queue as well
        queue_size = self.audio_queue.qsize()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        self.logger.debug(f"[RESET] Cleared audio queue ({queue_size} items)")
        
        # Reset tracking data
        self.predictions_history = []
        self.chunks_since_reset = 0
        self.last_model_reset_time = current_time
        self.recent_confidences = []  # Reset detection stability tracking
        self.failed_predictions = 0  # Reset failed predictions counter
        # Clear stuck state logging tracker
        if hasattr(self, '_last_stuck_log_value'):
            self._last_stuck_log_value = None
        
        # Reset warm-up state - model needs re-warming after reset
        self.model_ready = False
        self.warmup_chunks_processed = 0
        self._model_ready_logged = False  # Reset ready logging flag
        
        # Re-warm the model only if it exists
        if self.model:
            self.logger.info("[RESET] Re-warming model after reset...")
            self._warmup_model()
        
        self.logger.info("[RESET] Model state reset completed - ready for new detection cycle")
    
    def _check_model_health(self) -> bool:
        """
        Check if model is healthy based on recent predictions
        
        Returns:
            True if model appears healthy, False if it needs reset
        """
        # Check if we've had too many failures
        if self.failed_predictions >= self.max_failed_predictions:
            return False
            
        # Check if it's been too long since last successful prediction
        time_since_success = time.time() - self.last_successful_prediction
        if time_since_success > 30.0:  # 30 seconds without success
            self.logger.warning(f"No successful predictions for {time_since_success:.1f}s")
            return False
            
        return True
    
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