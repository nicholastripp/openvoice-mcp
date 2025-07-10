#!/usr/bin/env python3
"""
Isolated wake word detection test script
Tests OpenWakeWord functionality without the full application
"""
import asyncio
import numpy as np
import sounddevice as sd
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wake_word.detector import WakeWordDetector
from config import WakeWordConfig
from utils.logger import get_logger

# Test configuration
TEST_CONFIG = WakeWordConfig(
    enabled=True,
    model='hey_jarvis',
    sensitivity=0.0001,  # Very sensitive for testing
    timeout=5.0,
    vad_enabled=False,  # Disabled to avoid stuck states
    cooldown=2.0,
    audio_gain=5.0
)

# Global flag for detection
detection_occurred = False
detection_count = 0

def detection_callback(model_name: str, confidence: float):
    """Callback when wake word is detected"""
    global detection_occurred, detection_count
    detection_occurred = True
    detection_count += 1
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[{timestamp}] WAKE WORD DETECTED!")
    print(f"  Model: {model_name}")
    print(f"  Confidence: {confidence:.6f}")
    print(f"  Detection #{detection_count}")
    print("-" * 50)

async def audio_capture_loop(detector: WakeWordDetector):
    """Continuous audio capture and processing"""
    logger = get_logger("AudioCapture")
    
    # Audio parameters
    sample_rate = 24000  # OpenAI format
    chunk_size = 1200    # 50ms at 24kHz
    
    logger.info(f"Starting audio capture: {sample_rate}Hz, {chunk_size} samples/chunk")
    
    # Track audio statistics
    chunks_processed = 0
    last_stats_time = time.time()
    
    def audio_callback(indata, frames, time_info, status):
        """Callback for audio stream"""
        nonlocal chunks_processed
        
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert to bytes (PCM16 format)
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        
        # Process with detector
        detector.process_audio(audio_bytes, sample_rate)
        
        chunks_processed += 1
        
        # Print statistics periodically
        current_time = time.time()
        if current_time - last_stats_time > 5.0:  # Every 5 seconds
            print(f"\n[STATS] Processed {chunks_processed} chunks, Queue size: {detector.audio_queue.qsize()}")
            print(f"[STATS] Audio buffer: {len(detector.audio_buffer)} samples")
            print(f"[STATS] Model ready: {detector.model_ready}, Detections: {detection_count}")
            chunks_processed = 0
            last_stats_time = current_time
    
    # Start audio stream
    try:
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            dtype=np.float32,
            callback=audio_callback
        ):
            logger.info("Audio stream started - listening for wake word...")
            print("\n" + "="*60)
            print("WAKE WORD TEST STARTED")
            print("="*60)
            print(f"Model: {TEST_CONFIG.model}")
            print(f"Sensitivity: {TEST_CONFIG.sensitivity}")
            print(f"Say 'Hey Jarvis' to test detection")
            print("Press Ctrl+C to stop")
            print("="*60 + "\n")
            
            # Keep running
            while True:
                await asyncio.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("Audio capture stopped by user")
    except Exception as e:
        logger.error(f"Audio capture error: {e}")
        raise

async def monitor_model_state(detector: WakeWordDetector):
    """Monitor and log model state periodically"""
    logger = get_logger("ModelMonitor")
    
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        
        if detector.model and hasattr(detector, 'predictions_history'):
            # Check for stuck state
            if len(detector.predictions_history) >= 5:
                recent_preds = detector.predictions_history[-5:]
                if recent_preds:
                    values = []
                    for pred in recent_preds:
                        if pred:
                            model_name = list(pred.keys())[0]
                            values.append(pred[model_name])
                    
                    if values and len(set(values)) == 1:
                        logger.warning(f"Possible stuck state detected: {values[0]:.8e} repeated {len(values)} times")
                        print(f"\n[WARNING] Model may be stuck - all predictions are {values[0]:.8e}")
            
            # Log confidence statistics
            if hasattr(detector, 'confidence_history') and detector.confidence_history:
                avg_conf = sum(detector.confidence_history) / len(detector.confidence_history)
                max_conf = max(detector.confidence_history)
                print(f"\n[MODEL STATE] Avg confidence: {avg_conf:.6f}, Peak: {max_conf:.6f}")

async def test_wake_word_detection():
    """Main test function"""
    logger = get_logger("WakeWordTest")
    
    try:
        # Create detector
        detector = WakeWordDetector(TEST_CONFIG)
        detector.add_detection_callback(detection_callback)
        
        # Start detector
        logger.info("Starting wake word detector...")
        await detector.start()
        
        # Run audio capture and monitoring concurrently
        await asyncio.gather(
            audio_capture_loop(detector),
            monitor_model_state(detector)
        )
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up
        if 'detector' in locals():
            await detector.stop()
            logger.info("Detector stopped")
        
        # Print final statistics
        print("\n" + "="*60)
        print("TEST COMPLETED")
        print("="*60)
        print(f"Total detections: {detection_count}")
        print("="*60)

if __name__ == "__main__":
    # Run the test
    try:
        asyncio.run(test_wake_word_detection())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")