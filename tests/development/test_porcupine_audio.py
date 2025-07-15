#!/usr/bin/env python3
"""
Test Porcupine with actual audio capture
"""
import sys
import os
import time
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from wake_word import create_wake_word_detector
from audio.capture import AudioCapture
from utils.logger import setup_logging, get_logger


async def test_porcupine_with_audio():
    """Test Porcupine with real audio input"""
    logger = get_logger("PorcupineAudioTest")
    
    print("\n=== Testing Porcupine with Audio Capture ===")
    
    # Load config
    config = load_config()
    
    # Force Porcupine settings
    config.wake_word.engine = 'porcupine'
    config.wake_word.model = 'picovoice'
    config.wake_word.sensitivity = 0.5
    config.wake_word.enabled = True
    
    print(f"\nConfiguration:")
    print(f"  Engine: {config.wake_word.engine}")
    print(f"  Wake word: {config.wake_word.model}")
    print(f"  Sensitivity: {config.wake_word.sensitivity}")
    print(f"  Audio device: {config.audio.input_device}")
    
    # Create components
    audio_capture = AudioCapture(config.audio)
    wake_detector = create_wake_word_detector(config.wake_word)
    
    # Track detections
    detections = []
    
    def on_wake_word(model, confidence):
        detections.append((model, confidence, time.time()))
        print(f"\n[DETECTED] Wake word '{model}' detected! (confidence: {confidence})")
        print(f"Total detections: {len(detections)}")
    
    wake_detector.add_detection_callback(on_wake_word)
    
    # Audio callback
    def audio_callback(audio_data: bytes, sample_rate: int):
        # Feed audio to wake word detector
        wake_detector.process_audio(audio_data, sample_rate)
    
    audio_capture.set_callback(audio_callback)
    
    try:
        # Start components
        print("\nStarting audio capture...")
        await audio_capture.start()
        
        print("Starting wake word detector...")
        await wake_detector.start()
        
        print("\n" + "="*50)
        print("[READY] Say 'picovoice' to test wake word detection")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        # Run for 60 seconds or until interrupted
        start_time = time.time()
        last_status = 0
        
        while True:
            await asyncio.sleep(0.1)
            
            elapsed = time.time() - start_time
            
            # Show status every 5 seconds
            if int(elapsed / 5) > last_status:
                last_status = int(elapsed / 5)
                if not detections:
                    print(f"... listening ({int(elapsed)}s elapsed, no detections yet)")
                else:
                    print(f"... listening ({int(elapsed)}s elapsed, {len(detections)} detections so far)")
            
            # Stop after 60 seconds
            if elapsed > 60:
                print("\nTest duration complete (60s)")
                break
                
    except KeyboardInterrupt:
        print("\n\nStopping test...")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop components
        print("\nStopping components...")
        await audio_capture.stop()
        await wake_detector.stop()
        
        # Report results
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        if detections:
            print(f"\nTotal detections: {len(detections)}")
            for i, (model, conf, timestamp) in enumerate(detections):
                print(f"{i+1}. {model} at {timestamp:.2f}s (confidence: {conf})")
                
            # Calculate detection rate
            if len(detections) > 1:
                intervals = []
                for i in range(1, len(detections)):
                    interval = detections[i][2] - detections[i-1][2]
                    intervals.append(interval)
                avg_interval = sum(intervals) / len(intervals)
                print(f"\nAverage interval between detections: {avg_interval:.2f}s")
        else:
            print("\nNo wake words detected")
            print("\nPossible issues:")
            print("- Microphone not working or too quiet")
            print("- Wake word 'picovoice' not spoken clearly")
            print("- Sensitivity too low (current: 0.5)")
            print("- Audio gain may need adjustment")


def main():
    """Main test function"""
    setup_logging("INFO")
    
    print("="*60)
    print("Porcupine Wake Word Detection - Live Audio Test")
    print("="*60)
    
    # Check access key
    if not os.getenv('PICOVOICE_ACCESS_KEY'):
        print("\n[ERROR] PICOVOICE_ACCESS_KEY not set!")
        print("Please set: export PICOVOICE_ACCESS_KEY='your-key'")
        return 1
    
    try:
        asyncio.run(test_porcupine_with_audio())
        return 0
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())