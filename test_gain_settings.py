#!/usr/bin/env python3
"""
Test script to help tune wake word gain settings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wake_word.detector import WakeWordDetector
from config import WakeWordConfig
import numpy as np
import time


def test_gain_settings():
    """Test different gain settings and report confidence levels"""
    
    # Test configurations
    test_configs = [
        {"name": "Fixed 2.0x (old)", "fixed_gain": 2.0, "use_fixed": True},
        {"name": "Fixed 3.5x (new)", "fixed_gain": 3.5, "use_fixed": True},
        {"name": "Fixed 4.0x", "fixed_gain": 4.0, "use_fixed": True},
        {"name": "Dynamic 2-5x", "use_fixed": False, "min": 2.0, "max": 5.0},
        {"name": "Dynamic 3-6x", "use_fixed": False, "min": 3.0, "max": 6.0},
    ]
    
    # Load config
    config = WakeWordConfig(
        enabled=True,
        model="hey_jarvis",
        sensitivity=0.005,  # Use a middle sensitivity for testing
        timeout=5.0,
        vad_enabled=False,
        cooldown=2.0
    )
    
    print("Wake Word Gain Testing")
    print("=" * 50)
    print(f"Model: {config.model}")
    print(f"Sensitivity threshold: {config.sensitivity}")
    print("=" * 50)
    
    for test_config in test_configs:
        print(f"\nTesting: {test_config['name']}")
        print("-" * 30)
        
        # Create detector
        detector = WakeWordDetector(config)
        
        # Apply test configuration
        detector.use_fixed_gain = test_config.get("use_fixed", True)
        if detector.use_fixed_gain:
            detector.fixed_gain = test_config["fixed_gain"]
        else:
            detector.dynamic_gain_min = test_config["min"]
            detector.dynamic_gain_max = test_config["max"]
        
        # Initialize model
        detector._initialize_model()
        
        # Test with different audio levels
        test_rms_levels = [0.01, 0.02, 0.03, 0.05, 0.08]
        
        for rms_level in test_rms_levels:
            # Generate test audio at specific RMS level
            # Using noise that somewhat resembles speech patterns
            duration_samples = 1280  # 80ms at 16kHz
            
            # Create audio with speech-like characteristics
            # Mix of low and high frequency components
            t = np.linspace(0, duration_samples/16000, duration_samples)
            low_freq = np.sin(2 * np.pi * 200 * t)  # 200 Hz component
            mid_freq = np.sin(2 * np.pi * 800 * t) * 0.5  # 800 Hz component
            high_freq = np.sin(2 * np.pi * 2000 * t) * 0.3  # 2000 Hz component
            noise = np.random.normal(0, 0.1, duration_samples)
            
            audio = low_freq + mid_freq + high_freq + noise
            
            # Normalize to target RMS
            current_rms = np.sqrt(np.mean(audio ** 2))
            audio = audio * (rms_level / current_rms)
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # Get prediction
            prediction = detector.model.predict(audio)
            confidence = max(prediction.values()) if prediction else 0.0
            
            # Calculate what the gain would be
            if detector.use_fixed_gain:
                applied_gain = detector.fixed_gain
            else:
                applied_gain = np.clip(detector.dynamic_gain_target_rms / rms_level,
                                     detector.dynamic_gain_min, detector.dynamic_gain_max)
            
            print(f"  RMS={rms_level:.3f}, Gain={applied_gain:.1f}x, Confidence={confidence:.6f}")
        
        # Cleanup
        detector.stop()
        time.sleep(0.5)
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nRecommendations:")
    print("- If confidence values are still too low, try Fixed 4.0x")
    print("- If getting false positives, reduce gain or increase sensitivity threshold")
    print("- Dynamic gain can help with varying input levels but may be less predictable")


if __name__ == "__main__":
    test_gain_settings()