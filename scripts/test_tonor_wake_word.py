#!/usr/bin/env python3
"""
Test wake word detection with TONOR microphone
"""
import asyncio
import sys
import time
import numpy as np
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config
from wake_word import create_wake_word_detector

# Track detections and audio stats
detections = []
audio_stats = {'max': 0, 'clipped': 0, 'frames': 0}

def on_detection(keyword, confidence):
    detections.append((keyword, time.time()))
    print(f"\n*** WAKE WORD DETECTED: '{keyword}' ***")
    print(f"Total detections: {len(detections)}\n")

async def test_wake_word():
    print("TONOR Microphone Wake Word Test")
    print("=" * 50)
    
    # Load config
    config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')
    print(f"Configuration:")
    print(f"  Wake word: {config.wake_word.model}")
    print(f"  Sensitivity: {config.wake_word.sensitivity}")
    print(f"  Audio gain: {config.wake_word.audio_gain}")
    
    # Create detector
    detector = create_wake_word_detector(config.wake_word)
    detector.add_detection_callback(on_detection)
    
    # Override process_audio to add stats
    original_process = detector.process_audio
    
    def process_with_stats(audio_data, sample_rate):
        # Get audio stats
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        max_level = np.max(np.abs(audio_array))
        
        # After gain
        gained = max_level * config.wake_word.audio_gain
        if gained > 32767:
            audio_stats['clipped'] += 1
        
        audio_stats['max'] = max(audio_stats['max'], gained)
        audio_stats['frames'] += 1
        
        # Call original
        return original_process(audio_data, sample_rate)
    
    detector.process_audio = process_with_stats
    
    print("\nStarting detector...")
    await detector.start()
    
    print(f"\n*** LISTENING FOR '{config.wake_word.model.upper()}' ***")
    print("Speak clearly into the TONOR microphone")
    print("Running for 30 seconds...\n")
    
    # Use TONOR microphone specifically
    import pyaudio
    pa = pyaudio.PyAudio()
    
    # Find TONOR
    tonor_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if 'TONOR' in info['name'] and info['maxInputChannels'] > 0:
            tonor_index = i
            print(f"Using TONOR microphone: {info['name']}")
            break
    
    if tonor_index is None:
        print("ERROR: TONOR microphone not found!")
        return
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        input=True,
        input_device_index=tonor_index,
        frames_per_buffer=600
    )
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 30:
            # Read audio
            audio_data = stream.read(600, exception_on_overflow=False)
            
            # Process through detector
            detector.process_audio(audio_data, 48000)
            
            # Status update
            if audio_stats['frames'] % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Status: {elapsed:.1f}s | Max level: {int(audio_stats['max'])} | Clipped: {audio_stats['clipped']} | Detections: {len(detections)}", end='\r')
            
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    pa.terminate()
    await detector.stop()
    
    print(f"\n\nTest Results:")
    print(f"  Duration: {time.time() - start_time:.1f} seconds")
    print(f"  Max audio level: {int(audio_stats['max'])}")
    print(f"  Clipped frames: {audio_stats['clipped']}")
    print(f"  Total detections: {len(detections)}")
    
    if detections:
        print("\nDetections:")
        for keyword, timestamp in detections:
            print(f"  - '{keyword}' at {timestamp - start_time:.2f}s")
    else:
        print("\nNo detections - troubleshooting tips:")
        print("  1. Say 'picovoice' clearly and loudly")
        print("  2. Check microphone is not muted")
        print("  3. Audio levels are good, so issue may be pronunciation")

if __name__ == "__main__":
    asyncio.run(test_wake_word())