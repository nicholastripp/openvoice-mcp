#!/usr/bin/env python3
"""
Test Porcupine with actual microphone recording
"""
import os
import sys
import time
import threading
import queue
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

import pvporcupine
import pyaudio
import numpy as np

# Configuration
SAMPLE_RATE = 16000
FRAME_LENGTH = 512
CHANNELS = 1
FORMAT = pyaudio.paInt16

def test_realtime_detection():
    """Test Porcupine with real microphone input"""
    
    # Get access key
    access_key = os.getenv('PICOVOICE_ACCESS_KEY')
    if not access_key:
        print("ERROR: PICOVOICE_ACCESS_KEY not set")
        return
    
    print("Porcupine Real-time Detection Test")
    print("=" * 50)
    
    # Test both keywords
    keywords = ['jarvis', 'picovoice']
    sensitivity = 1.0
    
    print(f"Keywords: {keywords}")
    print(f"Sensitivity: {sensitivity}")
    print("\nCreating Porcupine...")
    
    try:
        # Create Porcupine
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=keywords,
            sensitivities=[sensitivity] * len(keywords)
        )
        
        print(f"[OK] Porcupine created")
        print(f"Sample rate: {porcupine.sample_rate}")
        print(f"Frame length: {porcupine.frame_length}")
        
        # Initialize PyAudio
        pa = pyaudio.PyAudio()
        
        # Find default input device
        default_device = pa.get_default_input_device_info()
        print(f"\nUsing audio device: {default_device['name']}")
        print(f"Device sample rate: {int(default_device['defaultSampleRate'])}")
        
        # Audio queue
        audio_queue = queue.Queue()
        
        def audio_callback(in_data, frame_count, time_info, status):
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)
        
        # Open audio stream
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_LENGTH,
            stream_callback=audio_callback
        )
        
        print("\nStarting audio stream...")
        stream.start_stream()
        
        print("\n*** LISTENING ***")
        print("Say 'Hey Jarvis' or 'Picovoice'...")
        print("Press Ctrl+C to stop\n")
        
        frames_processed = 0
        detection_count = 0
        
        try:
            while True:
                # Get audio frame
                try:
                    audio_data = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Convert to numpy array
                audio_frame = np.frombuffer(audio_data, dtype=np.int16)
                
                # Log every 50th frame
                frames_processed += 1
                if frames_processed % 50 == 0:
                    max_level = np.max(np.abs(audio_frame))
                    print(f"Frame {frames_processed}: max level = {max_level}", end='\\r')
                
                # Process with Porcupine (expects list)
                keyword_index = porcupine.process(audio_frame.tolist())
                
                # Check for detection
                if keyword_index >= 0:
                    detection_count += 1
                    keyword = keywords[keyword_index]
                    print(f"\\n[DETECTED #{detection_count}] Wake word: '{keyword}' at frame {frames_processed}")
                    print(f"Listening again...\\n")
                
        except KeyboardInterrupt:
            print("\\n\\nStopping...")
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()
        
        print(f"\\nTest complete. Total detections: {detection_count}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realtime_detection()