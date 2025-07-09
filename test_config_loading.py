#!/usr/bin/env python3
"""
Test configuration loading to verify wake word sensitivity value
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config

def test_config_loading():
    """Test that configuration loads correctly"""
    print("Testing configuration loading...")
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        print(f"Wake word config loaded:")
        print(f"  enabled: {config.wake_word.enabled}")
        print(f"  model: {config.wake_word.model}")
        print(f"  sensitivity: {config.wake_word.sensitivity}")
        print(f"  sensitivity type: {type(config.wake_word.sensitivity)}")
        print(f"  timeout: {config.wake_word.timeout}")
        print(f"  vad_enabled: {config.wake_word.vad_enabled}")
        print(f"  cooldown: {config.wake_word.cooldown}")
        
        # Check the sensitivity value specifically
        expected_sensitivity = 0.00005
        actual_sensitivity = config.wake_word.sensitivity
        
        if actual_sensitivity == expected_sensitivity:
            print(f"✓ Configuration loading SUCCESS: sensitivity = {actual_sensitivity}")
            return True
        else:
            print(f"✗ Configuration loading FAILED: expected {expected_sensitivity}, got {actual_sensitivity}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration loading ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)