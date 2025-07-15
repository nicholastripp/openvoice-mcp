#!/usr/bin/env python3
"""
Audio processing comparison test

This script compares our current audio processing approach with different
methods to identify why OpenWakeWord returns only 0.0 predictions.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logging, get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


def simulate_current_audio_processing(audio_data_pcm16, input_sample_rate=48000):
    """Simulate our current audio processing pipeline"""
    logger = get_logger("AudioProcessingTest")
    
    logger.info("Simulating current audio processing pipeline...")
    
    # Step 1: Convert PCM16 to float32 (our current approach)
    audio_array = np.frombuffer(audio_data_pcm16, dtype=np.int16)
    audio_float = audio_array.astype(np.float32) / 32767.0
    
    logger.info(f"After PCM16 to float32: shape={audio_float.shape}, dtype={audio_float.dtype}")
    logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
    logger.info(f"  Level: {np.max(np.abs(audio_float)):.6f}")
    
    # Step 2: Resample to 16kHz if needed
    if input_sample_rate != 16000:
        from scipy import signal
        new_length = int(len(audio_float) * 16000 / input_sample_rate)
        audio_float = signal.resample(audio_float, new_length)
        
        logger.info(f"After resampling to 16kHz: shape={audio_float.shape}")
        logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
        logger.info(f"  Level: {np.max(np.abs(audio_float)):.6f}")
    
    # Step 3: Ensure exactly 1280 samples (our approach)
    if len(audio_float) > 1280:
        audio_float = audio_float[:1280]
    elif len(audio_float) < 1280:
        # Pad with zeros
        audio_float = np.pad(audio_float, (0, 1280 - len(audio_float)), mode='constant')
    
    logger.info(f"After ensuring 1280 samples: shape={audio_float.shape}")
    logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
    logger.info(f"  Level: {np.max(np.abs(audio_float)):.6f}")
    
    return audio_float


def test_different_audio_processing_methods():
    """Test different audio processing methods with OpenWakeWord"""
    logger = get_logger("AudioProcessingTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing different audio processing methods...")
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Generate test audio at different sample rates
        test_cases = []
        
        # Case 1: Direct 16kHz audio (no resampling needed)
        duration_s = 1280 / 16000  # 80ms
        t = np.arange(0, duration_s, 1/16000)[:1280]
        audio_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
        test_cases.append(("Direct 16kHz", audio_16k, 16000))
        
        # Case 2: 48kHz audio (needs resampling)
        duration_s = 1280 / 16000  # Same duration, but at 48kHz
        t_48k = np.arange(0, duration_s, 1/48000)[:int(1280 * 3)]  # 3840 samples at 48kHz
        audio_48k_float = np.sin(2 * np.pi * 440 * t_48k).astype(np.float32) * 0.1
        audio_48k_pcm16 = (audio_48k_float * 32767).astype(np.int16).tobytes()
        test_cases.append(("48kHz -> 16kHz", audio_48k_pcm16, 48000))
        
        # Case 3: Random noise at different levels
        for level in [0.01, 0.05, 0.1]:
            noise_16k = np.random.normal(0, level, 1280).astype(np.float32)
            test_cases.append((f"Noise level {level}", noise_16k, 16000))
            
            # Also test as PCM16
            noise_pcm16 = (noise_16k * 32767).astype(np.int16).tobytes()
            test_cases.append((f"Noise level {level} (PCM16)", noise_pcm16, 16000))
        
        for name, audio_data, sample_rate in test_cases:
            logger.info(f"\n--- Testing: {name} ---")
            
            # Method 1: Direct (if already float32)
            if isinstance(audio_data, np.ndarray):
                logger.info("Method 1: Direct float32 audio")
                try:
                    predictions = model.predict(audio_data)
                    max_conf = max(predictions.values()) if predictions else 0.0
                    logger.info(f"  Direct: {predictions} (max: {max_conf:.8f})")
                except Exception as e:
                    logger.error(f"  Direct: failed: {e}")
            
            # Method 2: Our current processing (if PCM16 bytes)
            if isinstance(audio_data, bytes):
                logger.info("Method 2: Our current processing pipeline")
                try:
                    processed_audio = simulate_current_audio_processing(audio_data, sample_rate)
                    predictions = model.predict(processed_audio)
                    max_conf = max(predictions.values()) if predictions else 0.0
                    logger.info(f"  Current: {predictions} (max: {max_conf:.8f})")
                except Exception as e:
                    logger.error(f"  Current: failed: {e}")
            
            # Method 3: Alternative processing (ensure proper normalization)
            if isinstance(audio_data, bytes):
                logger.info("Method 3: Alternative processing (strict normalization)")
                try:
                    # Convert PCM16 to float32 with strict normalization
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0  # Slightly different normalization
                    
                    # Resample if needed
                    if sample_rate != 16000:
                        from scipy import signal
                        new_length = int(len(audio_float) * 16000 / sample_rate)
                        audio_float = signal.resample(audio_float, new_length)
                    
                    # Ensure exactly 1280 samples
                    if len(audio_float) != 1280:
                        if len(audio_float) > 1280:
                            audio_float = audio_float[:1280]
                        else:
                            audio_float = np.pad(audio_float, (0, 1280 - len(audio_float)), mode='constant')
                    
                    # Ensure proper range
                    audio_float = np.clip(audio_float, -1.0, 1.0)
                    
                    predictions = model.predict(audio_float)
                    max_conf = max(predictions.values()) if predictions else 0.0
                    logger.info(f"  Alternative: {predictions} (max: {max_conf:.8f})")
                except Exception as e:
                    logger.error(f"  Alternative: failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Audio processing test failed: {e}")
        return False


def test_our_exact_audio_pipeline():
    """Test our exact audio pipeline with debug information"""
    logger = get_logger("AudioProcessingTest")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available")
        return False
    
    logger.info("Testing our exact audio pipeline...")
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Simulate exactly what our detector does
        logger.info("Simulating exact detector audio processing...")
        
        # Generate test PCM16 audio (what we get from sounddevice)
        samples_48k = int(48000 * 0.08)  # 80ms at 48kHz
        test_audio_48k = np.sin(2 * np.pi * 440 * np.arange(samples_48k) / 48000) * 0.1
        pcm16_bytes = (test_audio_48k * 32767).astype(np.int16).tobytes()
        
        logger.info(f"Generated test PCM16 audio: {len(pcm16_bytes)} bytes")
        
        # Step 1: Convert bytes to numpy array (exactly as in detector)
        audio_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
        logger.info(f"PCM16 array: shape={audio_array.shape}, dtype={audio_array.dtype}")
        logger.info(f"  Range: [{np.min(audio_array)}, {np.max(audio_array)}]")
        
        # Step 2: Convert to float32 and normalize (exactly as in detector)
        audio_float = audio_array.astype(np.float32) / 32767.0
        logger.info(f"Float32 audio: shape={audio_float.shape}, dtype={audio_float.dtype}")
        logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
        
        # Step 3: Calculate audio level (exactly as in detector)
        audio_level = np.max(np.abs(audio_float))
        logger.info(f"Audio level: {audio_level:.6f}")
        
        # Step 4: Resample to 16kHz (exactly as in detector)
        input_sample_rate = 48000
        if input_sample_rate != 16000:
            from scipy import signal
            new_length = int(len(audio_float) * 16000 / input_sample_rate)
            audio_float = signal.resample(audio_float, new_length)
            logger.info(f"Resampled to 16kHz: shape={audio_float.shape}")
            logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
        
        # Step 5: Ensure exactly 1280 samples
        if len(audio_float) > 1280:
            audio_float = audio_float[:1280]
        elif len(audio_float) < 1280:
            audio_float = np.pad(audio_float, (0, 1280 - len(audio_float)), mode='constant')
        
        logger.info(f"Final audio for OpenWakeWord: shape={audio_float.shape}")
        logger.info(f"  Range: [{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]")
        logger.info(f"  Level: {np.max(np.abs(audio_float)):.6f}")
        logger.info(f"  Dtype: {audio_float.dtype}")
        
        # Step 6: Feed to OpenWakeWord
        predictions = model.predict(audio_float)
        max_conf = max(predictions.values()) if predictions else 0.0
        logger.info(f"OpenWakeWord predictions: {predictions} (max: {max_conf:.8f})")
        
        # Test with different audio levels
        logger.info("\nTesting with different audio levels...")
        for level in [0.001, 0.01, 0.05, 0.1, 0.5]:
            test_audio = np.sin(2 * np.pi * 440 * np.arange(1280) / 16000) * level
            test_audio = test_audio.astype(np.float32)
            
            predictions = model.predict(test_audio)
            max_conf = max(predictions.values()) if predictions else 0.0
            logger.info(f"  Level {level}: {predictions} (max: {max_conf:.8f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Exact pipeline test failed: {e}")
        return False


def main():
    """Run all audio processing tests"""
    setup_logging("INFO", console=True)
    logger = get_logger("AudioProcessingComparisonTest")
    
    logger.info("Starting audio processing comparison tests...")
    
    if not OPENWAKEWORD_AVAILABLE:
        logger.error("OpenWakeWord not available - cannot run tests")
        return
    
    tests = [
        ("Different Processing Methods", test_different_audio_processing_methods),
        ("Exact Pipeline Test", test_our_exact_audio_pipeline),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            logger.info(f"[{'PASS' if success else 'FAIL'}] {test_name}")
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {e}")


if __name__ == "__main__":
    main()