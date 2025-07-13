#!/usr/bin/env python3
"""
Test Picovoice Porcupine wake word detection
"""
import sys
import os
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from wake_word import create_wake_word_detector
from utils.logger import setup_logging, get_logger


def test_porcupine_installation():
    """Test basic Porcupine installation"""
    print("\n=== Testing Porcupine Installation ===")
    
    try:
        import pvporcupine
        print(f"[OK] Porcupine module imported successfully")
        print(f"  Version available via module")
        
        # Check for access key
        access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        if access_key:
            print(f"[OK] PICOVOICE_ACCESS_KEY environment variable is set")
        else:
            print("[FAIL] PICOVOICE_ACCESS_KEY environment variable not found")
            print("  Please set it with: export PICOVOICE_ACCESS_KEY='your-key-here'")
            print("  Get a free key at: https://console.picovoice.ai/")
            return False
            
        # Try to create instance
        try:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=['picovoice']
            )
            print(f"[OK] Successfully created Porcupine instance")
            print(f"  Sample rate: {porcupine.sample_rate}Hz")
            print(f"  Frame length: {porcupine.frame_length} samples")
            porcupine.delete()
            return True
        except Exception as e:
            print(f"[FAIL] Failed to create Porcupine instance: {e}")
            return False
            
    except ImportError as e:
        print(f"[FAIL] Failed to import pvporcupine: {e}")
        print("  Install with: pip install pvporcupine")
        return False


async def test_wake_word_detector():
    """Test wake word detector with Porcupine engine"""
    print("\n=== Testing Wake Word Detector ===")
    
    # Load config
    config = load_config()
    
    # Override to use Porcupine
    config.wake_word.engine = 'porcupine'
    config.wake_word.model = 'picovoice'
    config.wake_word.sensitivity = 0.5
    
    print(f"Configuration:")
    print(f"  Engine: {config.wake_word.engine}")
    print(f"  Model: {config.wake_word.model}")
    print(f"  Sensitivity: {config.wake_word.sensitivity}")
    
    try:
        # Create detector
        detector = create_wake_word_detector(config.wake_word)
        print(f"[OK] Created {detector.__class__.__name__}")
        
        # Get model info
        info = detector.get_model_info()
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Start detector
        await detector.start()
        print(f"\n[OK] Detector started successfully")
        
        # Set up callback
        detections = []
        def on_detection(model, confidence):
            detections.append((model, confidence))
            print(f"\n[DETECTED] WAKE WORD DETECTED: {model} (confidence: {confidence})")
        
        detector.add_detection_callback(on_detection)
        
        print("\n[LISTENING] Listening for wake word 'picovoice'...")
        print("   Say 'picovoice' to test detection")
        print("   Press Ctrl+C to stop\n")
        
        # Listen for 30 seconds
        start_time = time.time()
        try:
            while time.time() - start_time < 30:
                await asyncio.sleep(0.1)
                
                # Simulate some audio input (in real usage, audio capture would feed this)
                # For now, just show it's running
                if int(time.time() - start_time) % 5 == 0:
                    elapsed = int(time.time() - start_time)
                    if elapsed > 0:
                        print(f"   ... still listening ({elapsed}s elapsed)")
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        # Stop detector
        await detector.stop()
        print(f"\n[OK] Detector stopped")
        
        # Report results
        if detections:
            print(f"\n[STATS] Detected {len(detections)} wake word(s):")
            for model, conf in detections:
                print(f"   - {model}: {conf}")
        else:
            print(f"\n[STATS] No wake words detected")
            
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Error testing detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    setup_logging("INFO")
    logger = get_logger("PorcupineTest")
    
    print("=" * 50)
    print("Porcupine Wake Word Detection Test")
    print("=" * 50)
    
    # Test installation
    if not test_porcupine_installation():
        print("\n[ERROR] Porcupine installation test failed")
        print("\nTroubleshooting:")
        print("1. Make sure pvporcupine is installed: pip install pvporcupine")
        print("2. Set PICOVOICE_ACCESS_KEY environment variable")
        print("3. Get a free key at: https://console.picovoice.ai/")
        return 1
    
    # Test detector
    print("\nPreparing to test wake word detector...")
    print("Note: This test requires audio input to actually detect wake words")
    
    try:
        success = asyncio.run(test_wake_word_detector())
        if success:
            print("\n[SUCCESS] All tests passed!")
            return 0
        else:
            print("\n[ERROR] Detector test failed")
            return 1
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())