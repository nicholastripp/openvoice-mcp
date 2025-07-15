#!/usr/bin/env python3
"""
Quick test of wake word detection with current config
"""
import asyncio
import sys
import time
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config
from wake_word import create_wake_word_detector

# Track detections
detections = []

def on_detection(keyword, confidence):
    detections.append((keyword, time.time()))
    print(f"\n*** WAKE WORD DETECTED: '{keyword}' ***\n")

async def test_wake_word():
    print("Wake Word Detection Test")
    print("=" * 50)
    
    # Load config
    config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')
    print(f"Configuration:")
    print(f"  Engine: {config.wake_word.engine}")
    print(f"  Model: {config.wake_word.model}")
    print(f"  Sensitivity: {config.wake_word.sensitivity}")
    print(f"  Audio gain: {config.wake_word.audio_gain}")
    
    # Create detector
    detector = create_wake_word_detector(config.wake_word)
    detector.add_detection_callback(on_detection)
    
    print("\nStarting detector...")
    await detector.start()
    
    print(f"\n*** LISTENING FOR '{config.wake_word.model.upper()}' ***")
    print("Speak the wake word clearly...")
    print("Press Ctrl+C to stop\n")
    
    # Get model info
    info = detector.get_model_info()
    print(f"Model info: {info}")
    
    # Simulate audio input from microphone
    import pyaudio
    pa = pyaudio.PyAudio()
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,  # Match main app sample rate
        input=True,
        frames_per_buffer=600  # Match main app chunk size
    )
    
    chunk_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read audio
            audio_data = stream.read(600, exception_on_overflow=False)
            
            # Process through detector
            detector.process_audio(audio_data, 48000)
            
            chunk_count += 1
            if chunk_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Listening... ({elapsed:.1f}s, {len(detections)} detections)", end='\r')
            
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    pa.terminate()
    await detector.stop()
    
    print(f"\nTest complete. Total detections: {len(detections)}")
    for keyword, timestamp in detections:
        print(f"  - '{keyword}' at {timestamp - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(test_wake_word())