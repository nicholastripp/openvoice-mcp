#!/usr/bin/env python3
"""
Test wake word detector threshold to verify it matches config
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from wake_word.detector import WakeWordDetector

def test_detector_threshold():
    """Test that wake word detector uses correct threshold from config"""
    print("Testing wake word detector threshold...")
    
    try:
        # Load configuration (same as main app)
        config = load_config("config/config.yaml")
        
        print(f"Config wake word sensitivity: {config.wake_word.sensitivity}")
        print(f"Config type: {type(config.wake_word.sensitivity)}")
        
        # Create detector (same as main app)
        detector = WakeWordDetector(config.wake_word)
        
        print(f"Detector sensitivity: {detector.sensitivity}")
        print(f"Detector type: {type(detector.sensitivity)}")
        
        # Check if they match
        if detector.sensitivity == config.wake_word.sensitivity:
            print(f"✓ THRESHOLD CORRECT: detector sensitivity = {detector.sensitivity}")
            return True
        else:
            print(f"✗ THRESHOLD MISMATCH:")
            print(f"  Config: {config.wake_word.sensitivity}")
            print(f"  Detector: {detector.sensitivity}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detector_threshold()
    sys.exit(0 if success else 1)