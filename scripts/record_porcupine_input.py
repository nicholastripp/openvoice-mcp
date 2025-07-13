#!/usr/bin/env python3
"""
Record what Porcupine is actually receiving for debugging
"""
import os
import sys
import wave
import time
import numpy as np
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config
from wake_word import create_wake_word_detector

# Buffer to store audio frames
recorded_frames = []
detection_occurred = False

def on_detection(keyword, confidence):
    global detection_occurred
    detection_occurred = True
    print(f"\n[DETECTION] Wake word '{keyword}' detected!")

async def record_test():
    """Record audio that's being sent to Porcupine"""
    global recorded_frames
    
    print("Audio Recording Test for Porcupine")
    print("=" * 50)
    
    # Load config
    config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')
    print(f"Wake word model: {config.wake_word.model}")
    print(f"Sensitivity: {config.wake_word.sensitivity}")
    print(f"Audio gain: {config.wake_word.audio_gain}")
    
    # Create detector
    detector = create_wake_word_detector(config.wake_word)
    
    # Monkey-patch the process method to record frames
    original_process = detector.porcupine.process
    
    def recording_process(frame):
        recorded_frames.append(frame.copy() if isinstance(frame, list) else frame.tolist())
        return original_process(frame)
    
    detector.porcupine.process = recording_process
    
    # Add detection callback
    detector.add_detection_callback(on_detection)
    
    print("\nStarting detector...")
    await detector.start()
    
    print("\n*** RECORDING ***")
    print(f"Say '{config.wake_word.model}' into the microphone...")
    print("Recording for 10 seconds...\n")
    
    # Simulate audio input (48kHz, 600 samples per chunk)
    import asyncio
    start_time = time.time()
    chunk_count = 0
    
    # Use actual microphone input
    import pyaudio
    pa = pyaudio.PyAudio()
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        input=True,
        frames_per_buffer=600
    )
    
    try:
        while time.time() - start_time < 10:
            audio_data = stream.read(600, exception_on_overflow=False)
            detector.process_audio(audio_data, 48000)
            chunk_count += 1
            
            if chunk_count % 50 == 0:
                print(f"Processed {chunk_count} chunks ({len(recorded_frames)} frames to Porcupine)", end='\r')
            
            await asyncio.sleep(0.001)  # Small delay
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    await detector.stop()
    
    print(f"\n\nRecording complete!")
    print(f"Total chunks processed: {chunk_count}")
    print(f"Total frames sent to Porcupine: {len(recorded_frames)}")
    print(f"Detection occurred: {detection_occurred}")
    
    # Save recorded frames
    if recorded_frames:
        print("\nSaving recorded audio...")
        
        # Flatten all frames into one array
        all_samples = []
        for frame in recorded_frames:
            all_samples.extend(frame)
        
        # Convert to numpy array
        audio_data = np.array(all_samples, dtype=np.int16)
        
        # Save as WAV file
        output_file = "/tmp/porcupine_input.wav"
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)  # Porcupine's sample rate
            wf.writeframes(audio_data.tobytes())
        
        print(f"Audio saved to: {output_file}")
        print(f"Duration: {len(audio_data) / 16000:.2f} seconds")
        print(f"You can play it with: aplay {output_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(record_test())