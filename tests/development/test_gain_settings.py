#!/usr/bin/env python3
"""
Test script to help tune wake word gain settings
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wake_word.detector import WakeWordDetector
from config import WakeWordConfig, load_config
import numpy as np
import time


def test_gain_calculations():
    """Test different gain calculations without full model initialization"""
    
    # Test configurations
    test_configs = [
        {"name": "Fixed 2.0x (old)", "fixed_gain": 2.0, "use_fixed": True},
        {"name": "Fixed 3.5x (new)", "fixed_gain": 3.5, "use_fixed": True},
        {"name": "Fixed 4.0x", "fixed_gain": 4.0, "use_fixed": True},
        {"name": "Dynamic 2-5x", "use_fixed": False, "min": 2.0, "max": 5.0, "target_rms": 0.04},
        {"name": "Dynamic 3-6x", "use_fixed": False, "min": 3.0, "max": 6.0, "target_rms": 0.04},
    ]
    
    print("Wake Word Gain Calculation Testing")
    print("=" * 60)
    print("Testing gain calculations for different input RMS levels")
    print("Note: This tests gain calculations only, not actual model predictions")
    print("=" * 60)
    
    # Test with different audio RMS levels
    test_rms_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    
    for test_config in test_configs:
        print(f"\n{test_config['name']}:")
        print("-" * 40)
        
        for rms_level in test_rms_levels:
            if test_config["use_fixed"]:
                applied_gain = test_config["fixed_gain"]
                gain_type = "fixed"
            else:
                # Calculate bounded dynamic gain
                target_rms = test_config["target_rms"]
                raw_gain = target_rms / rms_level
                applied_gain = np.clip(raw_gain, test_config["min"], test_config["max"])
                gain_type = "dynamic"
            
            # Calculate final RMS after gain
            final_rms = rms_level * applied_gain
            
            print(f"  RMS={rms_level:.3f} -> Gain={applied_gain:.1f}x ({gain_type}) -> Final RMS={final_rms:.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Fixed gain provides consistent amplification regardless of input level")
    print("- Dynamic gain adapts to input level but is bounded to prevent extremes")
    print("- Higher final RMS generally leads to better wake word confidence")
    print("- Target final RMS should be in 0.1-0.2 range for best results")


def test_with_actual_model():
    """Test with actual model predictions (simpler version)"""
    
    try:
        # Load actual configuration
        print("\n" + "=" * 60)
        print("Testing with Actual Wake Word Model")
        print("=" * 60)
        
        # Try to load config from file, fall back to defaults
        try:
            config = load_config()
            wake_config = config.wake_word
            print(f"Loaded config from file: model={wake_config.model}, sensitivity={wake_config.sensitivity}")
        except Exception as e:
            print(f"Could not load config file ({e}), using defaults")
            wake_config = WakeWordConfig(
                enabled=True,
                model="hey_jarvis",
                sensitivity=0.005,
                timeout=5.0,
                vad_enabled=False,
                cooldown=2.0
            )
        
        # Create detector
        detector = WakeWordDetector(wake_config)
        
        # Load model only (don't start full detection)
        try:
            detector._load_model()
            print(f"Successfully loaded model: {wake_config.model}")
            
            # Test current configuration
            print(f"\nCurrent gain configuration:")
            print(f"- Use fixed gain: {detector.use_fixed_gain}")
            if detector.use_fixed_gain:
                print(f"- Fixed gain: {detector.fixed_gain}x")
            else:
                print(f"- Dynamic gain range: {detector.dynamic_gain_min}x - {detector.dynamic_gain_max}x")
                print(f"- Target RMS: {detector.dynamic_gain_target_rms}")
            
            # Test prediction with sample audio
            print("\nTesting with sample audio:")
            duration_samples = 1280  # 80ms at 16kHz
            
            # Generate speech-like test audio
            t = np.linspace(0, duration_samples/16000, duration_samples)
            audio = (
                np.sin(2 * np.pi * 300 * t) * 0.7 +  # Fundamental
                np.sin(2 * np.pi * 900 * t) * 0.3 +  # Harmonic
                np.random.normal(0, 0.05, duration_samples)  # Noise
            )
            
            # Test different gain levels
            test_gains = [2.0, 3.0, 3.5, 4.0, 5.0]
            
            for gain in test_gains:
                # Apply gain and clip
                test_audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
                
                # Get prediction
                try:
                    with detector.model_lock:
                        prediction = detector.model.predict(test_audio)
                    confidence = max(prediction.values()) if prediction else 0.0
                    rms = np.sqrt(np.mean(test_audio ** 2))
                    
                    threshold_ratio = confidence / wake_config.sensitivity if wake_config.sensitivity > 0 else 0
                    status = "DETECT" if confidence >= wake_config.sensitivity else "below"
                    
                    print(f"  Gain {gain:.1f}x: RMS={rms:.3f}, Confidence={confidence:.6f} ({threshold_ratio:.1f}x threshold) [{status}]")
                    
                except Exception as e:
                    print(f"  Gain {gain:.1f}x: Prediction failed - {e}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Make sure you're running this on the Raspberry Pi with OpenWakeWord installed")
            return
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # First test gain calculations (works everywhere)
    test_gain_calculations()
    
    # Then test with actual model (only works on Pi with proper setup)
    test_with_actual_model()