#!/usr/bin/env python3
"""
Test version of WakeWordDetector without buffer initialization

This removes the 1.5s silence buffer initialization that might be
causing the model to get stuck in a constant prediction state.
"""
import sys
import asyncio
import numpy as np
import threading
from typing import Callable, Optional
from queue import Queue, Empty
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import WakeWordConfig
from utils.logger import setup_logging, get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


class TestWakeWordDetector:
    """
    Test version of WakeWordDetector without buffer initialization
    """
    
    def __init__(self, model_name="alexa", sensitivity=0.5):
        self.logger = get_logger("TestWakeWordDetector")
        
        if not OPENWAKEWORD_AVAILABLE:
            raise ImportError("OpenWakeWord not available")
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1280
        
        # State
        self.is_running = False
        self.model: Optional[WakeWordModel] = None
        self.audio_queue = Queue()
        self.detection_callbacks = []
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Detection parameters
        self.model_name = model_name
        self.sensitivity = sensitivity
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_cooldown = 2.0
    
    async def start(self) -> None:
        """Start wake word detection"""
        if self.is_running:
            return
        
        try:
            # Load model WITHOUT buffer initialization
            self.logger.info(f"Loading model {self.model_name} WITHOUT buffer initialization...")
            self._load_model_no_buffer_init()
            
            # Start detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.is_running = True
            self.logger.info(f"Wake word detection started (NO buffer init)")
            
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop wake word detection"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        self.audio_buffer = np.array([], dtype=np.float32)
        self.logger.info("Wake word detection stopped")
    
    def add_detection_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add detection callback"""
        self.detection_callbacks.append(callback)
    
    def process_audio(self, audio_data: bytes, input_sample_rate: int = 48000) -> None:
        """Process audio data"""
        if not self.is_running:
            return
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Calculate audio level
            audio_level = np.max(np.abs(audio_float))
            
            # Resample to 16kHz if needed
            if input_sample_rate != self.sample_rate:
                from scipy import signal
                new_length = int(len(audio_float) * self.sample_rate / input_sample_rate)
                audio_float = signal.resample(audio_float, new_length)
            
            # Add to buffer
            if audio_level > 0.001:  # Low threshold for audio activity
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
                
                # Process in 80ms chunks
                while len(self.audio_buffer) >= self.chunk_size:
                    chunk = self.audio_buffer[:self.chunk_size]
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                    
                    # Queue for processing
                    self.audio_queue.put(chunk, block=False)
                    print(f"   TEST_DETECTOR: Queued chunk (level={audio_level:.3f}, queue_size={self.audio_queue.qsize()})")
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _load_model_no_buffer_init(self) -> None:
        """Load model WITHOUT buffer initialization"""
        try:
            # Map model names
            model_mapping = {
                'alexa': 'alexa_v0.1',
                'hey_jarvis': 'hey_jarvis_v0.1',
                'hey_mycroft': 'hey_mycroft_v0.1',
                'hey_rhasspy': 'hey_rhasspy_v0.1',
                'ok_nabu': 'ok_nabu_v0.1'
            }
            
            actual_model_name = model_mapping.get(self.model_name, self.model_name)
            
            # Create model with minimal configuration
            self.model = WakeWordModel(wakeword_models=[actual_model_name])
            
            self.logger.info(f"Model loaded: {actual_model_name}")
            self.logger.info(f"Available models: {list(self.model.models.keys())}")
            
            # TEST: Make a single prediction to verify model is working
            test_chunk = np.zeros(self.chunk_size, dtype=np.float32)
            test_predictions = self.model.predict(test_chunk)
            self.logger.info(f"Initial test prediction: {test_predictions}")
            
            # NO BUFFER INITIALIZATION - this is the key change!
            self.logger.info("SKIPPING buffer initialization - testing raw model state")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _detection_loop(self) -> None:
        """Detection loop"""
        self.logger.debug("Detection thread started")
        
        chunks_processed = 0
        while not self.stop_event.is_set():
            try:
                # Get audio chunk
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                chunks_processed += 1
                
                # Process with OpenWakeWord
                try:
                    predictions = self.model.predict(audio_chunk)
                    
                    # Log all predictions for debugging
                    if predictions:
                        for model_name, confidence in predictions.items():
                            print(f"   TEST_DETECTOR: {model_name} = {confidence:.8f} (chunk #{chunks_processed})")
                            
                            # Check for detection
                            if confidence >= self.sensitivity:
                                current_time = time.time()
                                if current_time - self.last_detection_time >= self.detection_cooldown:
                                    print(f"   TEST_DETECTOR: *** DETECTION! *** {model_name} = {confidence:.8f}")
                                    self.logger.info(f"Wake word detected: {model_name} (confidence: {confidence:.8f})")
                                    self.last_detection_time = current_time
                                    
                                    # Call callbacks
                                    for callback in self.detection_callbacks:
                                        try:
                                            callback(model_name, confidence)
                                        except Exception as e:
                                            self.logger.error(f"Error in callback: {e}")
                    else:
                        print(f"   TEST_DETECTOR: Empty predictions! (chunk #{chunks_processed})")
                
                except Exception as e:
                    print(f"   TEST_DETECTOR: Prediction error: {e}")
                    self.logger.error(f"Prediction error: {e}")
            
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
        
        self.logger.debug("Detection thread stopped")


async def test_no_buffer_init():
    """Test the detector without buffer initialization"""
    setup_logging("INFO", console=True)
    logger = get_logger("TestNoBufInit")
    
    logger.info("Testing wake word detector WITHOUT buffer initialization...")
    
    # Create test detector
    detector = TestWakeWordDetector(model_name="alexa", sensitivity=0.1)  # Low sensitivity
    
    # Track detections
    detections = []
    
    def on_detection(model_name, confidence):
        detections.append((model_name, confidence))
        print(f"\n*** CALLBACK: Wake word detected: {model_name} = {confidence:.8f} ***\n")
    
    detector.add_detection_callback(on_detection)
    
    # Start detector
    await detector.start()
    
    # Simulate audio input with various levels
    logger.info("Simulating audio input...")
    
    # Test with different noise levels
    test_levels = [0.001, 0.01, 0.05, 0.1, 0.15]
    
    for level in test_levels:
        logger.info(f"Testing with audio level: {level:.3f}")
        
        # Generate 5 chunks at this level
        for i in range(5):
            # Create random audio
            audio_samples = np.random.normal(0, level, 1280).astype(np.int16)
            audio_bytes = audio_samples.tobytes()
            
            # Process audio
            detector.process_audio(audio_bytes, input_sample_rate=48000)
            
            # Small delay
            await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(2.0)
    
    # Stop detector
    await detector.stop()
    
    # Report results
    logger.info(f"Test completed. Detections: {len(detections)}")
    for model_name, confidence in detections:
        logger.info(f"  - {model_name}: {confidence:.8f}")
    
    return len(detections)


if __name__ == "__main__":
    asyncio.run(test_no_buffer_init())