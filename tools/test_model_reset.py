#!/usr/bin/env python3
"""
Test script to investigate OpenWakeWord model state issues

This script tests whether the model buffer initialization is causing
the constant prediction values problem.
"""
import sys
import asyncio
import numpy as np
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


async def test_model_without_buffer_init():
    """Test OpenWakeWord model without buffer initialization"""
    logger = get_logger("ModelResetTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    try:
        logger.info("Testing OpenWakeWord model WITHOUT buffer initialization...")
        
        # Create model without any buffer initialization
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        logger.info(f"Model loaded: {list(model.models.keys())}")
        
        # Test with varying audio chunks
        test_chunks = [
            np.zeros(1280, dtype=np.float32),  # Silence
            np.random.normal(0, 0.01, 1280).astype(np.float32),  # Low noise
            np.random.normal(0, 0.05, 1280).astype(np.float32),  # Medium noise
            np.random.normal(0, 0.1, 1280).astype(np.float32),   # High noise
        ]
        
        logger.info("Testing with different audio chunks:")
        for i, chunk in enumerate(test_chunks):
            predictions = model.predict(chunk)
            logger.info(f"Chunk {i+1}: {predictions}")
            
            # Check if predictions are varying
            if predictions:
                for model_name, confidence in predictions.items():
                    print(f"  {model_name}: {confidence:.8f}")
        
        # Test with multiple sequential chunks to see if values change
        logger.info("\nTesting sequential chunks:")
        for i in range(10):
            chunk = np.random.normal(0, 0.02, 1280).astype(np.float32)
            predictions = model.predict(chunk)
            if predictions:
                max_confidence = max(predictions.values())
                logger.info(f"Sequential chunk {i+1}: max confidence = {max_confidence:.8f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_model_with_reset():
    """Test OpenWakeWord model with explicit reset"""
    logger = get_logger("ModelResetTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    try:
        logger.info("Testing OpenWakeWord model with explicit reset...")
        
        # Create model
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Reset the model buffers
        try:
            model.reset()
            logger.info("Model buffers reset successfully")
        except Exception as e:
            logger.warning(f"Model reset failed: {e}")
        
        # Test with varying audio chunks
        test_chunks = [
            np.zeros(1280, dtype=np.float32),  # Silence
            np.random.normal(0, 0.01, 1280).astype(np.float32),  # Low noise
            np.random.normal(0, 0.05, 1280).astype(np.float32),  # Medium noise
        ]
        
        logger.info("Testing after reset:")
        for i, chunk in enumerate(test_chunks):
            predictions = model.predict(chunk)
            logger.info(f"Post-reset chunk {i+1}: {predictions}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_different_models():
    """Test different wake word models to see if issue is model-specific"""
    logger = get_logger("ModelResetTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    models_to_test = ['alexa_v0.1', 'hey_jarvis_v0.1', 'hey_mycroft_v0.1']
    
    for model_name in models_to_test:
        try:
            logger.info(f"Testing model: {model_name}")
            
            model = WakeWordModel(wakeword_models=[model_name])
            
            # Test with a few chunks
            for i in range(3):
                chunk = np.random.normal(0, 0.02, 1280).astype(np.float32)
                predictions = model.predict(chunk)
                if predictions:
                    max_confidence = max(predictions.values())
                    logger.info(f"  {model_name} chunk {i+1}: {max_confidence:.8f}")
            
        except Exception as e:
            logger.warning(f"Failed to test {model_name}: {e}")
    
    return True


async def test_very_low_sensitivity():
    """Test wake word detection with very low sensitivity"""
    logger = get_logger("ModelResetTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    try:
        logger.info("Testing with very low sensitivity (0.000001)...")
        
        # Create model
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Test with different audio levels
        test_levels = [0.001, 0.01, 0.05, 0.1]
        
        for level in test_levels:
            chunk = np.random.normal(0, level, 1280).astype(np.float32)
            predictions = model.predict(chunk)
            
            if predictions:
                for model_name, confidence in predictions.items():
                    logger.info(f"Audio level {level:.3f}: {model_name} = {confidence:.8f}")
                    
                    # Check if confidence exceeds very low threshold
                    if confidence > 0.000001:
                        logger.info(f"  DETECTION at very low threshold: {confidence:.8f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def main():
    """Run all tests"""
    setup_logging("INFO", console=True)
    logger = get_logger("ModelResetTest")
    
    logger.info("Starting OpenWakeWord model state investigation...")
    
    # Test 1: Model without buffer initialization
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Model without buffer initialization")
    logger.info("="*60)
    await test_model_without_buffer_init()
    
    # Test 2: Model with explicit reset
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Model with explicit reset")
    logger.info("="*60)
    await test_model_with_reset()
    
    # Test 3: Different models
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Different wake word models")
    logger.info("="*60)
    await test_different_models()
    
    # Test 4: Very low sensitivity
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Very low sensitivity testing")
    logger.info("="*60)
    await test_very_low_sensitivity()
    
    logger.info("\n" + "="*60)
    logger.info("All tests completed")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())