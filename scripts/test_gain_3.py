#!/usr/bin/env python3
"""
Quick test of gain 3.0 with TONOR microphone
"""
import asyncio
import sys
import time
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config
from wake_word import create_wake_word_detector

print("Testing Audio Gain 3.0")
print("=" * 50)

detections = []

def on_detection(keyword, confidence):
    detections.append(keyword)
    print(f"\n*** DETECTED: '{keyword}' ***\n")

async def test():
    config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')
    print(f"Wake word: {config.wake_word.model}")
    print(f"Gain: {config.wake_word.audio_gain}")
    
    detector = create_wake_word_detector(config.wake_word)
    detector.add_detection_callback(on_detection)
    
    await detector.start()
    
    print("\nListening for 30 seconds...")
    print("Say 'picovoice' clearly\n")
    
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        input=True,
        frames_per_buffer=600
    )
    
    start = time.time()
    while time.time() - start < 30:
        audio = stream.read(600, exception_on_overflow=False)
        detector.process_audio(audio, 48000)
        await asyncio.sleep(0.001)
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    await detector.stop()
    
    print(f"\nTest complete. Detections: {len(detections)}")

asyncio.run(test())