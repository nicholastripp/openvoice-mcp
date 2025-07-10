#!/usr/bin/env python3
"""
Test script to verify wake word model stuck state fixes
"""
import asyncio
import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from config import WakeWordConfig
from wake_word.detector import WakeWordDetector


async def test_stuck_model_detection():
    """Test that the model properly detects and recovers from stuck states"""
    print("Testing wake word model stuck state detection and recovery...")
    
    # Create config
    config = WakeWordConfig(
        enabled=True,
        model="hey_jarvis",
        sensitivity=0.0001,
        timeout=5.0,
        vad_enabled=False,
        cooldown=2.0
    )
    
    # Create detector
    detector = WakeWordDetector(config)
    
    # Track detections
    detections = []
    def on_detection(model_name, confidence):
        detections.append((model_name, confidence))
        print(f"WAKE WORD DETECTED: {model_name} (confidence: {confidence:.6f})")
    
    detector.add_detection_callback(on_detection)
    
    try:
        # Start detector
        await detector.start()
        print("Wake word detector started successfully")
        
        # Test 1: Feed normal audio and check for stuck values
        print("\nTest 1: Monitoring for stuck predictions...")
        for i in range(20):
            # Generate varied test audio
            if i % 3 == 0:
                # Silence
                audio = np.zeros(1200, dtype=np.int16)
            elif i % 3 == 1:
                # Low noise
                audio = (np.random.normal(0, 0.01, 1200) * 32767).astype(np.int16)
            else:
                # Medium noise
                audio = (np.random.normal(0, 0.05, 1200) * 32767).astype(np.int16)
            
            # Convert to bytes and process
            audio_bytes = audio.tobytes()
            detector.process_audio(audio_bytes, input_sample_rate=24000)
            
            # Small delay between chunks
            await asyncio.sleep(0.05)
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Check if model got stuck
        print(f"\nModel statistics:")
        print(f"- Chunks since reset: {detector.chunks_since_reset}")
        print(f"- Predictions in history: {len(detector.predictions_history)}")
        print(f"- Average confidence: {detector.avg_confidence:.6f}")
        print(f"- Peak confidence: {detector.peak_confidence:.6f}")
        
        # Test 2: Simulate audio that might trigger detection
        print("\nTest 2: Testing with speech-like audio...")
        for i in range(10):
            # Generate speech-like audio (higher amplitude with variation)
            t = np.linspace(0, 0.05, 1200)
            # Mix of frequencies similar to speech
            audio = np.sin(2 * np.pi * 200 * t) * 0.1  # Low frequency
            audio += np.sin(2 * np.pi * 800 * t) * 0.05  # Mid frequency
            audio += np.random.normal(0, 0.02, 1200)  # Noise
            audio = (audio * 32767).astype(np.int16)
            
            audio_bytes = audio.tobytes()
            detector.process_audio(audio_bytes, input_sample_rate=24000)
            
            await asyncio.sleep(0.05)
        
        # Final wait
        await asyncio.sleep(2.0)
        
        # Report results
        print(f"\nTest complete:")
        print(f"- Total detections: {len(detections)}")
        print(f"- Model resets performed: Check logs for [RESET] entries")
        print(f"- Final chunks since reset: {detector.chunks_since_reset}")
        
        if detections:
            print("\nDetections:")
            for model, conf in detections:
                print(f"  - {model}: {conf:.6f}")
        
    finally:
        # Stop detector
        await detector.stop()
        print("\nWake word detector stopped")


if __name__ == "__main__":
    asyncio.run(test_stuck_model_detection())