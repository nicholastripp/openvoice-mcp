#!/usr/bin/env python3
"""
Test script for wake word detection

Usage:
    ./venv/bin/python examples/test_wake_word.py --interactive
    
Note: Must be run from the project root using the virtual environment.
Requires audio input device (microphone) for interactive testing.
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config, WakeWordConfig
from wake_word.detector import WakeWordDetector
from utils.logger import setup_logging, get_logger


async def test_wake_word_installation():
    """Test OpenWakeWord installation"""
    logger = get_logger("WakeWordTest")
    
    logger.info("Testing OpenWakeWord installation...")
    success = WakeWordDetector.test_installation()
    
    if success:
        logger.info("[OK] OpenWakeWord installation test passed")
    else:
        logger.error("[ERROR] OpenWakeWord installation test failed")
    
    return success


async def test_wake_word_models():
    """Test available wake word models"""
    logger = get_logger("WakeWordTest")
    
    try:
        # Create dummy config to get available models
        config = WakeWordConfig()
        detector = WakeWordDetector(config)
        
        models = detector.get_available_models()
        logger.info(f"Available wake word models: {models}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return False


async def test_wake_word_detection(config_path, duration=30):
    """Test wake word detection with real audio"""
    logger = get_logger("WakeWordTest")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        if not config.wake_word.enabled:
            logger.warning("Wake word detection is disabled in configuration")
            return False
        
        # Create detector
        detector = WakeWordDetector(config.wake_word)
        
        # Setup detection callback
        detected_words = []
        
        def on_detection(model_name, confidence):
            detected_words.append((model_name, confidence))
            logger.info(f"[DETECTED] WAKE WORD: {model_name} (confidence: {confidence:.3f})")
        
        detector.add_detection_callback(on_detection)
        
        # Start detection
        await detector.start()
        
        # Display model info
        model_info = detector.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        logger.info(f"Listening for wake word '{config.wake_word.model}' for {duration} seconds...")
        logger.info("Say the wake word to test detection!")
        
        # Wait for detections
        await asyncio.sleep(duration)
        
        # Stop detection
        await detector.stop()
        
        # Report results
        if detected_words:
            logger.info(f"[OK] Detected {len(detected_words)} wake word(s):")
            for model_name, confidence in detected_words:
                logger.info(f"  - {model_name}: {confidence:.3f}")
        else:
            logger.warning("[WARNING] No wake words detected during test period")
        
        return len(detected_words) > 0
        
    except Exception as e:
        logger.error(f"Wake word detection test failed: {e}")
        return False


async def test_model_switching():
    """Test switching between different wake word models"""
    logger = get_logger("WakeWordTest")
    
    models_to_test = ["alexa", "hey_mycroft", "ok_nabu"]
    
    for model_name in models_to_test:
        logger.info(f"Testing model: {model_name}")
        
        try:
            config = WakeWordConfig(model=model_name)
            detector = WakeWordDetector(config)
            
            await detector.start()
            model_info = detector.get_model_info()
            logger.info(f"  [OK] {model_name}: {model_info}")
            await detector.stop()
            
        except Exception as e:
            logger.error(f"  [ERROR] {model_name}: {e}")


async def interactive_test(config_path, sensitivity=None, debug_predictions=False):
    """Interactive wake word testing"""
    logger = get_logger("WakeWordTest")
    
    config = load_config(config_path)
    
    # Override sensitivity if specified
    if sensitivity is not None:
        config.wake_word.sensitivity = sensitivity
        print(f"Overriding sensitivity to {sensitivity}")
    
    # Enable debug predictions if requested
    if debug_predictions:
        print("Debug predictions enabled - will show all OpenWakeWord predictions")
    
    detector = WakeWordDetector(config.wake_word)
    
    # Test audio device first
    from audio.capture import AudioCapture
    logger.info("Testing audio device...")
    devices = AudioCapture.list_devices()
    if devices:
        logger.info(f"Available audio devices: {len(devices)}")
        for device in devices[:3]:  # Show first 3 devices
            logger.info(f"  - {device['name']} ({device['index']})")
    else:
        logger.warning("No audio devices found!")
        return
    
    # Import audio capture for microphone input
    from audio.capture import AudioCapture
    
    # Track detections
    import time
    detection_count = 0
    last_detection_time = time.time()
    
    def on_detection(model_name, confidence):
        nonlocal detection_count, last_detection_time
        detection_count += 1
        import time
        current_time = time.time()
        
        # Debug: Show that callback is being called
        logger.debug(f"Detection callback called: count={detection_count}, confidence={confidence:.3f}")
        
        print(f"\n[DETECTED] WAKE WORD #{detection_count}: {model_name} (confidence: {confidence:.3f})")
        print(f"   Detection time: {current_time - last_detection_time:.1f}s since last")
        print(f"   Cooldown: {config.wake_word.cooldown}s - next detection possible at {current_time + config.wake_word.cooldown:.1f}")
        print("   Listening for next detection...")
        
        last_detection_time = current_time
    
    detector.add_detection_callback(on_detection)
    
    # Start audio capture and test microphone
    audio_capture = AudioCapture(config.audio)
    await audio_capture.start()
    
    # Test microphone for 2 seconds
    print("Testing microphone for 2 seconds...")
    test_audio_levels = []
    test_start_time = time.time()
    
    def mic_test_callback(audio_data):
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0
        audio_level = np.max(np.abs(audio_float))
        test_audio_levels.append(audio_level)
    
    audio_capture.add_callback(mic_test_callback)
    
    # Wait for test
    while time.time() - test_start_time < 2.0:
        await asyncio.sleep(0.1)
    
    # Remove test callback
    audio_capture.remove_callback(mic_test_callback)
    
    # Show results
    if test_audio_levels:
        max_level = max(test_audio_levels)
        avg_level = sum(test_audio_levels) / len(test_audio_levels)
        print(f"Microphone test: max={max_level:.3f}, avg={avg_level:.3f}")
        if max_level < 0.001:
            print("WARNING: Very low audio levels detected - check microphone")
    else:
        print("WARNING: No audio data received - check microphone connection")
    
    # Connect audio to wake word detector with debugging
    audio_chunks_processed = 0
    
    def audio_callback(audio_data):
        nonlocal audio_chunks_processed
        audio_chunks_processed += 1
        
        # Calculate audio level for debugging
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0
        audio_level = np.max(np.abs(audio_float))
        
        # Log audio activity periodically
        if audio_chunks_processed % 100 == 0:  # Every 100 chunks (~5 seconds)
            print(f"   Audio processing: {audio_chunks_processed} chunks, current level: {audio_level:.3f}")
        
        detector.process_audio(audio_data, input_sample_rate=config.audio.sample_rate)
    
    audio_capture.add_callback(audio_callback)
    
    await detector.start()
    
    # Add startup delay to prevent false positives
    print("Starting wake word detector...")
    await asyncio.sleep(2.0)  # 2 second startup delay
    
    print(f"\n[MIC] Wake word detector started with audio input!")
    print(f"Model: {config.wake_word.model}")
    print(f"Sensitivity: {config.wake_word.sensitivity}")
    print(f"Audio device: {config.audio.input_device}")
    print(f"Sample rate: {config.audio.sample_rate}Hz")
    print(f"Say '{config.wake_word.model}' to test detection")
    print("Press Ctrl+C to stop")
    if config.wake_word.sensitivity > 0.4:
        print(f"TIP: If no detections, try lower sensitivity: --sensitivity 0.3")
    print("\nListening...")
    
    try:
        import time
        last_status_time = time.time()
        while True:
            await asyncio.sleep(1.0)
            # Show periodic status
            current_time = time.time()
            if current_time - last_status_time > 5:  # Every 5 seconds
                print(f"   Still listening... ({detection_count} detections so far, last: {current_time - last_detection_time:.1f}s ago)")
                last_status_time = current_time
    except KeyboardInterrupt:
        print("\nStopping...")
        await detector.stop()
        await audio_capture.stop()


def main():
    parser = argparse.ArgumentParser(description="Test wake word detection")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--installation", action="store_true", help="Test installation only")
    parser.add_argument("--models", action="store_true", help="Test available models")
    parser.add_argument("--detection", type=int, metavar="DURATION", help="Test detection for N seconds")
    parser.add_argument("--switch", action="store_true", help="Test model switching")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    parser.add_argument("--sensitivity", type=float, help="Override wake word sensitivity (0.0-1.0)")
    parser.add_argument("--debug-predictions", action="store_true", help="Show all OpenWakeWord predictions")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", console=True)
    
    async def run_tests():
        if args.installation:
            await test_wake_word_installation()
        elif args.models:
            await test_wake_word_models()
        elif args.detection:
            await test_wake_word_detection(args.config, args.detection)
        elif args.switch:
            await test_model_switching()
        elif args.interactive:
            await interactive_test(args.config, args.sensitivity, args.debug_predictions)
        else:
            # Run basic tests
            logger = get_logger("WakeWordTest")
            logger.info("Running wake word tests...")
            
            success1 = await test_wake_word_installation()
            if success1:
                await test_wake_word_models()
                await test_model_switching()
    
    # Run tests
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")


if __name__ == "__main__":
    main()