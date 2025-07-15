#!/usr/bin/env python3
"""
Enhanced test script for wake word detection with stuck state fixes

This script tests the comprehensive fixes for OpenWakeWord stuck prediction issues:
- Model reset mechanisms
- Stuck state detection
- Enhanced audio validation
- Prediction monitoring
"""
import sys
import asyncio
import argparse
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config, WakeWordConfig
from wake_word.detector import WakeWordDetector
from utils.logger import setup_logging, get_logger
from audio.capture import AudioCapture


async def test_stuck_state_detection():
    """Test the stuck state detection mechanism"""
    logger = get_logger("StuckStateTest")
    
    logger.info("Testing stuck state detection mechanism...")
    
    # Create test configuration
    config = WakeWordConfig(
        model="alexa",
        sensitivity=0.1,
        enabled=True,
        vad_enabled=False,
        cooldown=2.0
    )
    
    detector = WakeWordDetector(config)
    
    # Test stuck state detection logic
    logger.info("Testing stuck state detection with artificial identical predictions...")
    
    # Simulate identical predictions
    identical_prediction = {'alexa_v0.1': 1.1165829e-06}
    
    for i in range(10):
        detector._track_prediction(identical_prediction)
        is_stuck = detector._is_stuck_state()
        logger.info(f"Prediction {i+1}: stuck={is_stuck}, history_len={len(detector.predictions_history)}")
        
        if is_stuck:
            logger.info(f"Stuck state detected after {i+1} identical predictions")
            break
    
    # Test with varying predictions
    logger.info("Testing with varying predictions...")
    detector.predictions_history = []  # Clear history
    
    varying_predictions = [
        {'alexa_v0.1': 1.0e-06},
        {'alexa_v0.1': 2.0e-06},
        {'alexa_v0.1': 1.5e-06},
        {'alexa_v0.1': 3.0e-06},
    ]
    
    for i, pred in enumerate(varying_predictions):
        detector._track_prediction(pred)
        is_stuck = detector._is_stuck_state()
        logger.info(f"Varying prediction {i+1}: stuck={is_stuck}, value={pred['alexa_v0.1']:.8f}")
    
    logger.info("Stuck state detection test completed")


async def test_model_reset_timing():
    """Test model reset timing mechanisms"""
    logger = get_logger("ModelResetTest")
    
    logger.info("Testing model reset timing mechanisms...")
    
    config = WakeWordConfig(
        model="alexa",
        sensitivity=0.1,
        enabled=True,
        vad_enabled=False,
        cooldown=2.0
    )
    
    detector = WakeWordDetector(config)
    
    # Set short reset interval for testing
    detector.model_reset_interval = 2.0  # 2 seconds
    
    await detector.start()
    
    # Test time-based reset
    logger.info("Testing time-based model reset...")
    
    for i in range(5):
        should_reset = detector._should_reset_model()
        logger.info(f"Check {i+1}: should_reset={should_reset}, time_since_reset={time.time() - detector.last_model_reset_time:.1f}s")
        
        if should_reset:
            detector._reset_model_state()
            logger.info(f"Model reset triggered at check {i+1}")
        
        await asyncio.sleep(1.0)
    
    await detector.stop()
    logger.info("Model reset timing test completed")


async def test_audio_validation():
    """Test audio validation mechanisms"""
    logger = get_logger("AudioValidationTest")
    
    logger.info("Testing audio validation mechanisms...")
    
    config = WakeWordConfig(
        model="alexa",
        sensitivity=0.1,
        enabled=True,
        vad_enabled=False,
        cooldown=2.0
    )
    
    detector = WakeWordDetector(config)
    
    # Test various audio formats
    test_cases = [
        ("Normal audio", np.random.normal(0, 0.1, 1280).astype(np.float32)),
        ("Out of range audio", np.random.normal(0, 2.0, 1280).astype(np.float32)),  # Will be clipped
        ("NaN audio", np.full(1280, np.nan, dtype=np.float32)),  # Invalid
        ("Infinite audio", np.full(1280, np.inf, dtype=np.float32)),  # Invalid
        ("Zero audio", np.zeros(1280, dtype=np.float32)),  # Valid but silent
        ("Wrong dtype", np.random.normal(0, 0.1, 1280).astype(np.int16)),  # Invalid dtype
    ]
    
    for name, audio_data in test_cases:
        try:
            is_valid = detector._validate_audio_chunk(audio_data)
            logger.info(f"{name}: valid={is_valid}, dtype={audio_data.dtype}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        except Exception as e:
            logger.error(f"{name}: validation failed with error: {e}")
    
    logger.info("Audio validation test completed")


async def test_comprehensive_detection(config_path, duration=30):
    """Test comprehensive wake word detection with all fixes"""
    logger = get_logger("ComprehensiveTest")
    
    logger.info("Testing comprehensive wake word detection with enhanced fixes...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Override settings for testing
    config.wake_word.sensitivity = 0.05  # Very sensitive for testing
    config.wake_word.enabled = True
    config.wake_word.vad_enabled = False
    
    detector = WakeWordDetector(config.wake_word)
    
    # Track detections and resets
    detections = []
    resets = []
    
    def on_detection(model_name, confidence):
        detections.append((time.time(), model_name, confidence))
        logger.info(f"[DETECTION] {model_name}: {confidence:.8f}")
    
    # Override reset method to track resets
    original_reset = detector._reset_model_state
    def track_reset():
        resets.append(time.time())
        logger.info(f"[MODEL RESET] Total resets: {len(resets)}")
        original_reset()
    
    detector._reset_model_state = track_reset
    detector.add_detection_callback(on_detection)
    
    # Start audio capture
    audio_capture = AudioCapture(config.audio)
    await audio_capture.start()
    
    # Connect audio to detector
    def audio_callback(audio_data):
        detector.process_audio(audio_data, input_sample_rate=config.audio.sample_rate)
    
    audio_capture.add_callback(audio_callback)
    
    # Start detector
    await detector.start()
    
    logger.info(f"Comprehensive test running for {duration} seconds...")
    logger.info("Features enabled:")
    logger.info(f"  - Stuck state detection: {detector.reset_on_stuck}")
    logger.info(f"  - Model reset interval: {detector.model_reset_interval}s")
    logger.info(f"  - Stuck detection threshold: {detector.stuck_detection_threshold}")
    logger.info(f"  - Sensitivity: {config.wake_word.sensitivity}")
    logger.info("")
    logger.info("Say 'Alexa' to test detection!")
    logger.info("Watch for [STUCK] and [RESET] indicators in output")
    
    # Wait for test duration
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            await asyncio.sleep(1.0)
            
            # Periodic status
            if int(time.time() - start_time) % 10 == 0:
                logger.info(f"Test progress: {int(time.time() - start_time)}s/{duration}s, detections: {len(detections)}, resets: {len(resets)}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    # Stop components
    await detector.stop()
    await audio_capture.stop()
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Detections: {len(detections)}")
    logger.info(f"Model resets: {len(resets)}")
    
    if detections:
        logger.info("\nDetections:")
        for timestamp, model_name, confidence in detections:
            logger.info(f"  {timestamp - start_time:.1f}s: {model_name} = {confidence:.8f}")
    
    if resets:
        logger.info("\nModel resets:")
        for timestamp in resets:
            logger.info(f"  {timestamp - start_time:.1f}s: Model reset")
    
    # Success criteria
    logger.info("\nEvaluation:")
    if len(detections) > 0:
        logger.info("[PASS] Wake word detection is working")
    else:
        logger.info("[FAIL] No wake word detections")
    
    if len(resets) > 0:
        logger.info("[PASS] Model reset mechanism is working")
    else:
        logger.info("[INFO] No model resets were triggered")
    
    logger.info("="*60)


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Enhanced wake word detection test")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--stuck-test", action="store_true", help="Test stuck state detection")
    parser.add_argument("--reset-test", action="store_true", help="Test model reset timing")
    parser.add_argument("--audio-test", action="store_true", help="Test audio validation")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, console=True)
    logger = get_logger("EnhancedWakeWordTest")
    
    logger.info("Starting enhanced wake word detection tests...")
    
    try:
        if args.stuck_test:
            await test_stuck_state_detection()
        elif args.reset_test:
            await test_model_reset_timing()
        elif args.audio_test:
            await test_audio_validation()
        elif args.comprehensive:
            await test_comprehensive_detection(args.config, args.duration)
        else:
            # Run all tests
            logger.info("Running all enhanced tests...")
            await test_stuck_state_detection()
            await asyncio.sleep(1)
            await test_model_reset_timing()
            await asyncio.sleep(1)
            await test_audio_validation()
            await asyncio.sleep(1)
            logger.info("All unit tests completed. Run with --comprehensive for full test.")
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    logger.info("Enhanced wake word detection tests completed")


if __name__ == "__main__":
    asyncio.run(main())