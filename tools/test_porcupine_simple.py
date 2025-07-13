#!/usr/bin/env python3
"""
Simple Porcupine wake word test with visual feedback
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pvporcupine
import pyaudio


def main():
    """Simple Porcupine test"""
    print("="*60)
    print("Porcupine Wake Word Test - Simple Version")
    print("="*60)
    
    # Get access key
    access_key = os.getenv('PICOVOICE_ACCESS_KEY')
    if not access_key:
        print("\n[ERROR] PICOVOICE_ACCESS_KEY not set!")
        return 1
    
    # Create Porcupine
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=['picovoice', 'alexa', 'hey google']
        )
        print(f"\n[OK] Porcupine initialized")
        print(f"  Sample rate: {porcupine.sample_rate}Hz")
        print(f"  Frame length: {porcupine.frame_length} samples")
        print(f"  Wake words: picovoice, alexa, hey google")
    except Exception as e:
        print(f"\n[ERROR] Failed to create Porcupine: {e}")
        return 1
    
    # Setup audio
    pa = pyaudio.PyAudio()
    
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    
    print("\n" + "="*60)
    print("LISTENING FOR WAKE WORDS:")
    print("  - 'picovoice'")
    print("  - 'alexa'") 
    print("  - 'hey google'")
    print("\nSpeak clearly near the microphone")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    keywords = ['picovoice', 'alexa', 'hey google']
    detection_count = 0
    last_detection_time = 0
    
    try:
        print("Listening", end="", flush=True)
        dot_count = 0
        
        while True:
            # Read audio
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            
            # Process
            keyword_index = porcupine.process(pcm)
            
            # Check detection
            if keyword_index >= 0:
                current_time = time.time()
                
                # Cooldown check (2 seconds)
                if current_time - last_detection_time > 2.0:
                    detection_count += 1
                    keyword = keywords[keyword_index]
                    
                    print(f"\n\n*** WAKE WORD DETECTED: '{keyword}' ***")
                    print(f"    Detection #{detection_count}")
                    print(f"    Time: {time.strftime('%H:%M:%S')}\n")
                    
                    last_detection_time = current_time
                    print("Listening", end="", flush=True)
                    dot_count = 0
            else:
                # Show activity
                dot_count += 1
                if dot_count % 20 == 0:
                    print(".", end="", flush=True)
                if dot_count % 200 == 0:
                    print(" (still listening)", end="", flush=True)
                    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        # Cleanup
        if audio_stream is not None:
            audio_stream.close()
        
        if pa is not None:
            pa.terminate()
        
        if porcupine is not None:
            porcupine.delete()
        
        print(f"\nTotal detections: {detection_count}")
        print("Test complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())