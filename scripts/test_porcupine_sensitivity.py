#!/usr/bin/env python3
"""
Test different sensitivity values for Porcupine wake word detection
"""
import asyncio
import numpy as np
import time
import sys
sys.path.insert(0, '/home/ansible/ha-voice-assistant/src')

from config import WakeWordConfig
from wake_word.porcupine_detector import PorcupineDetector


async def test_sensitivity_values():
    """Test different sensitivity values"""
    print("Testing Porcupine sensitivity values")
    print("=" * 50)
    
    # Test values including the extremely low one from config
    test_values = [0.000001, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    for sensitivity in test_values:
        print(f"\nTesting sensitivity: {sensitivity}")
        print("-" * 30)
        
        # Create config with test sensitivity
        config = WakeWordConfig()
        config.engine = "porcupine"
        config.model = "picovoice"
        config.sensitivity = sensitivity
        config.cooldown = 1.0
        
        try:
            # Create detector
            detector = PorcupineDetector(config)
            
            # Add detection callback
            detections = []
            def on_detection(keyword, conf):
                detections.append((keyword, conf, time.time()))
                print(f"  [DETECTED] '{keyword}' with confidence {conf}")
            
            detector.add_detection_callback(on_detection)
            
            # Start detector
            await detector.start()
            
            print(f"  Detector started successfully")
            print(f"  Actual sensitivities used: {detector.sensitivities}")
            
            # Run for 10 seconds
            print(f"  Say 'picovoice' to test detection...")
            await asyncio.sleep(10.0)
            
            # Stop detector
            await detector.stop()
            
            print(f"  Total detections: {len(detections)}")
            
        except Exception as e:
            print(f"  [ERROR] Failed with sensitivity {sensitivity}: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete")


if __name__ == "__main__":
    asyncio.run(test_sensitivity_values())