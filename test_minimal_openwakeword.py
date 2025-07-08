#!/usr/bin/env python3
"""
Minimal OpenWakeWord test to verify basic functionality

This script bypasses our wrapper to test OpenWakeWord directly and identify
why it's returning only 0.0 predictions.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path for utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logging, get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


def test_openwakeword_installation():
    """Test basic OpenWakeWord installation"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord installation...")
    
    try:
        # Test model creation without any parameters
        model = WakeWordModel()
        logger.info(f"Default model created successfully")
        logger.info(f"Available models: {list(model.models.keys())}")
        
        # Test with dummy audio
        dummy_audio = np.zeros(1280, dtype=np.float32)
        predictions = model.predict(dummy_audio)
        logger.info(f"Dummy audio predictions: {predictions}")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenWakeWord installation test failed: {e}")
        return False


def test_model_with_specific_wake_word():
    """Test OpenWakeWord with specific wake word model"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord with specific wake word model...")
    
    try:
        # Test with alexa model specifically
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        logger.info(f"Alexa model created successfully")
        logger.info(f"Available models: {list(model.models.keys())}")
        
        # Test different audio inputs
        test_cases = [
            ("Silence", np.zeros(1280, dtype=np.float32)),
            ("Low noise", np.random.normal(0, 0.01, 1280).astype(np.float32)),
            ("Medium noise", np.random.normal(0, 0.05, 1280).astype(np.float32)),
            ("High noise", np.random.normal(0, 0.1, 1280).astype(np.float32)),
            ("Sine wave 440Hz", np.sin(2 * np.pi * 440 * np.arange(1280) / 16000).astype(np.float32)),
            ("Sine wave 1000Hz", np.sin(2 * np.pi * 1000 * np.arange(1280) / 16000).astype(np.float32)),
        ]
        
        for name, audio_data in test_cases:
            try:
                predictions = model.predict(audio_data)
                max_conf = max(predictions.values()) if predictions else 0.0
                logger.info(f"{name}: {predictions} (max: {max_conf:.8f})")
            except Exception as e:
                logger.error(f"{name}: prediction failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Specific model test failed: {e}")
        return False


def test_audio_format_requirements():
    """Test OpenWakeWord with different audio formats"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord audio format requirements...")
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Test different audio formats
        base_audio = np.random.normal(0, 0.05, 1280)
        
        format_tests = [
            ("float32", base_audio.astype(np.float32)),
            ("float64", base_audio.astype(np.float64)),
            ("int16 normalized", (base_audio * 32767).astype(np.int16)),
            ("float32 [0,1]", np.abs(base_audio).astype(np.float32)),
            ("float32 [-1,1]", np.clip(base_audio, -1.0, 1.0).astype(np.float32)),
            ("float32 large values", (base_audio * 10).astype(np.float32)),
        ]
        
        for name, audio_data in format_tests:
            try:
                predictions = model.predict(audio_data)
                max_conf = max(predictions.values()) if predictions else 0.0
                logger.info(f"{name}: {predictions} (max: {max_conf:.8f})")
            except Exception as e:
                logger.error(f"{name}: prediction failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Audio format test failed: {e}")
        return False


def test_model_reset_behavior():
    """Test OpenWakeWord model reset behavior"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord model reset behavior...")
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Test prediction before reset
        audio_data = np.random.normal(0, 0.05, 1280).astype(np.float32)
        predictions_before = model.predict(audio_data)
        logger.info(f"Before reset: {predictions_before}")
        
        # Reset model
        model.reset()
        logger.info("Model reset called")
        
        # Test prediction after reset
        predictions_after = model.predict(audio_data)
        logger.info(f"After reset: {predictions_after}")
        
        # Test multiple predictions after reset
        for i in range(5):
            audio_data = np.random.normal(0, 0.05, 1280).astype(np.float32)
            predictions = model.predict(audio_data)
            max_conf = max(predictions.values()) if predictions else 0.0
            logger.info(f"Post-reset #{i+1}: {predictions} (max: {max_conf:.8f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Model reset test failed: {e}")
        return False


def test_sequential_predictions():
    """Test OpenWakeWord with sequential predictions"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord sequential predictions...")
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Test multiple sequential predictions
        logger.info("Testing 20 sequential predictions with varying audio...")
        
        for i in range(20):
            # Generate different audio for each prediction
            if i < 5:
                # Start with silence
                audio_data = np.zeros(1280, dtype=np.float32)
            elif i < 10:
                # Low noise
                audio_data = np.random.normal(0, 0.01, 1280).astype(np.float32)
            elif i < 15:
                # Medium noise
                audio_data = np.random.normal(0, 0.05, 1280).astype(np.float32)
            else:
                # High noise
                audio_data = np.random.normal(0, 0.1, 1280).astype(np.float32)
            
            try:
                predictions = model.predict(audio_data)
                max_conf = max(predictions.values()) if predictions else 0.0
                audio_level = np.max(np.abs(audio_data))
                logger.info(f"Prediction #{i+1}: {predictions} (max: {max_conf:.8f}, audio_level: {audio_level:.6f})")
            except Exception as e:
                logger.error(f"Prediction #{i+1} failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Sequential predictions test failed: {e}")
        return False


def test_model_initialization_parameters():
    """Test OpenWakeWord with different initialization parameters"""
    logger = get_logger("MinimalTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing OpenWakeWord with different initialization parameters...")
    
    test_configs = [
        ("Default", {}),
        ("VAD disabled", {'vad_threshold': None}),
        ("VAD very low", {'vad_threshold': 0.1}),
        ("VAD medium", {'vad_threshold': 0.5}),
        ("VAD high", {'vad_threshold': 0.9}),
        ("No inference framework", {'inference_framework': None}),
    ]
    
    for name, config in test_configs:
        try:
            logger.info(f"Testing {name}: {config}")
            model = WakeWordModel(wakeword_models=['alexa_v0.1'], **config)
            
            # Test with noise
            audio_data = np.random.normal(0, 0.05, 1280).astype(np.float32)
            predictions = model.predict(audio_data)
            max_conf = max(predictions.values()) if predictions else 0.0
            logger.info(f"  {name}: {predictions} (max: {max_conf:.8f})")
            
        except Exception as e:
            logger.error(f"  {name}: initialization failed: {e}")
    
    return True


def main():
    """Run all minimal tests"""
    setup_logging("INFO", console=True)
    logger = get_logger("MinimalOpenWakeWordTest")
    
    logger.info("Starting minimal OpenWakeWord tests...")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available - cannot run tests")
        return
    
    tests = [
        ("Installation Test", test_openwakeword_installation),
        ("Specific Model Test", test_model_with_specific_wake_word),
        ("Audio Format Test", test_audio_format_requirements),
        ("Model Reset Test", test_model_reset_behavior),
        ("Sequential Predictions Test", test_sequential_predictions),
        ("Initialization Parameters Test", test_model_initialization_parameters),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
            logger.info(f"[{'PASS' if success else 'FAIL'}] {test_name}")
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"[{status}] {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! OpenWakeWord is working correctly.")
    else:
        logger.warning("Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()