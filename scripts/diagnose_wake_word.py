#!/usr/bin/env python3
"""
Diagnose wake word detection issues
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, '/home/ansible/ha-voice-assistant/src')

from config import load_config
from wake_word import create_wake_word_detector


async def diagnose_wake_word():
    """Diagnose wake word detection configuration and behavior"""
    print("Wake Word Detection Diagnostics")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading configuration...")
    config_path = "/home/ansible/ha-voice-assistant/config/config.yaml"
    
    # Read raw config to see exact values
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    wake_config = raw_config.get('wake_word', {})
    print(f"   Engine: {wake_config.get('engine', 'not set')}")
    print(f"   Model: {wake_config.get('model', 'not set')}")
    print(f"   Sensitivity (raw): {wake_config.get('sensitivity', 'not set')}")
    print(f"   Sensitivity type: {type(wake_config.get('sensitivity', 0))}")
    
    # Load config properly
    config = load_config(config_path)
    print(f"\n2. Loaded config sensitivity: {config.wake_word.sensitivity}")
    print(f"   Type: {type(config.wake_word.sensitivity)}")
    
    # Check access key
    access_key = os.getenv('PICOVOICE_ACCESS_KEY', getattr(config.wake_word, 'porcupine_access_key', None))
    print(f"\n3. Access key present: {'Yes' if access_key else 'No'}")
    if access_key:
        print(f"   Key length: {len(access_key)}")
    
    # Create detector
    print("\n4. Creating wake word detector...")
    try:
        detector = create_wake_word_detector(config.wake_word)
        print(f"   Detector type: {type(detector).__name__}")
        
        if hasattr(detector, 'keywords'):
            print(f"   Keywords: {detector.keywords}")
        if hasattr(detector, 'sensitivities'):
            print(f"   Sensitivities: {detector.sensitivities}")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"\n5. Model info:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Start detector
        print("\n6. Starting detector...")
        await detector.start()
        print("   Started successfully!")
        
        # Test with very simple audio
        print("\n7. Testing with simple audio pulse...")
        
        # Create a simple audio pulse
        sample_rate = 48000  # Main app uses 48kHz
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        # Create a sine wave burst
        freq = 1000  # 1kHz
        t = np.linspace(0, duration, samples)
        audio = (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)
        
        # Process it
        print(f"   Sending {len(audio)} samples at {sample_rate}Hz")
        detector.process_audio(audio.tobytes(), sample_rate)
        
        # Wait a bit
        await asyncio.sleep(1.0)
        
        # Check if audio made it through
        if hasattr(detector, 'audio_queue'):
            print(f"   Audio queue size: {detector.audio_queue.qsize()}")
        if hasattr(detector, 'audio_buffer'):
            print(f"   Audio buffer size: {len(detector.audio_buffer)}")
        
        # Stop
        await detector.stop()
        print("\n8. Detector stopped successfully")
        
    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Diagnostics complete")
    
    # Recommendations
    print("\nRecommendations:")
    sensitivity = wake_config.get('sensitivity', 0)
    if isinstance(sensitivity, (int, float)) and sensitivity < 0.1:
        print("- [WARNING] Sensitivity is extremely low. Try 0.5 for normal use")
    if sensitivity > 1.0:
        print("- [WARNING] Sensitivity is above 1.0. Maximum valid value is 1.0")


if __name__ == "__main__":
    # Import numpy here since it's used in the test
    import numpy as np
    asyncio.run(diagnose_wake_word())