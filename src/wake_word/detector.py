"""
OpenWakeWord wake word detection implementation
"""
import asyncio
import numpy as np
import threading
from typing import Callable, Optional, Any
from queue import Queue, Empty
import time

from ..config import WakeWordConfig
from ..utils.logger import get_logger

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
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
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
    
    def process_audio(self, audio_data: bytes) -> None:
        """
        Process audio data for wake word detection
        
        Args:
            audio_data: Audio data (will be resampled to 16kHz if needed)
        """
        if not self.is_running:
            return
        
        try:
            # Convert bytes to numpy array (assuming PCM16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Queue for processing
            self.audio_queue.put(audio_float, block=False)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _load_model(self) -> None:
        """Load OpenWakeWord model"""
        try:
            # Map common model names to OpenWakeWord models
            model_mapping = {
                'hey_jarvis': 'hey_jarvis_v0.1',
                'alexa': 'alexa_v0.1', 
                'hey_mycroft': 'hey_mycroft_v0.1',
                'hey_rhasspy': 'hey_rhasspy_v0.1',
                'ok_nabu': 'ok_nabu_v0.1'
            }
            
            actual_model_name = model_mapping.get(self.model_name, self.model_name)
            
            # Initialize model with VAD if enabled
            model_kwargs = {}
            if self.vad_enabled:
                model_kwargs['enable_speex_noise_suppression'] = True
            
            self.model = WakeWordModel(
                wakeword_models=[actual_model_name],
                **model_kwargs
            )
            
            self.logger.info(f"Loaded model: {actual_model_name}")
            self.logger.info(f"Available models: {list(self.model.models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to load wake word model '{self.model_name}': {e}")
            # Try to suggest available models
            try:
                dummy_model = WakeWordModel()
                available = list(dummy_model.models.keys())
                self.logger.info(f"Available models: {available}")
            except:
                pass
            raise
    
    def _detection_loop(self) -> None:
        """Background thread for wake word detection"""
        self.logger.debug("Wake word detection thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process with OpenWakeWord
                predictions = self.model.predict(audio_chunk)
                
                # Check for wake word detection
                for model_name, confidence in predictions.items():
                    if confidence >= self.sensitivity:
                        current_time = time.time()
                        
                        # Check cooldown to prevent rapid re-triggers
                        if current_time - self.last_detection_time >= self.detection_cooldown:
                            self.logger.info(f"Wake word detected: {model_name} (confidence: {confidence:.3f})")
                            self.last_detection_time = current_time
                            
                            # Call detection callbacks
                            for callback in self.detection_callbacks:
                                try:
                                    callback(model_name, confidence)
                                except Exception as e:
                                    self.logger.error(f"Error in detection callback: {e}")
                        else:
                            self.logger.debug(f"Wake word detected but in cooldown: {model_name} (confidence: {confidence:.3f})")
                
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
            'chunk_size': self.chunk_size
        }
    
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
            
            # Try to load a model (this will download if needed)
            test_model = WakeWordModel(wakeword_models=['hey_jarvis_v0.1'])
            
            # Test with dummy audio
            dummy_audio = np.zeros(1280, dtype=np.float32)  # 80ms of silence
            predictions = test_model.predict(dummy_audio)
            
            logger.info("✅ OpenWakeWord installation test passed")
            logger.info(f"Available models: {list(test_model.models.keys())}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ OpenWakeWord not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ OpenWakeWord installation test failed: {e}")
            return False