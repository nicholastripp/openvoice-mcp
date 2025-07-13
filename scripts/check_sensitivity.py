#!/usr/bin/env python3
"""
Check the sensitivity value being used by Porcupine
"""
import sys
import os
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config

# Load config
config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')

print("Wake Word Configuration Check")
print("=" * 40)
print(f"Engine: {config.wake_word.engine}")
print(f"Model: {config.wake_word.model}")
print(f"Sensitivity: {config.wake_word.sensitivity}")
print(f"Sensitivity type: {type(config.wake_word.sensitivity)}")
print(f"Cooldown: {config.wake_word.cooldown}")

# Check if sensitivity is in valid range
sensitivity = config.wake_word.sensitivity
if sensitivity < 0.0 or sensitivity > 1.0:
    print(f"\n[WARNING] Sensitivity {sensitivity} is outside valid range [0.0, 1.0]")
    print("Porcupine will clamp this to valid range")
    clamped = max(0.0, min(1.0, sensitivity))
    print(f"Actual value used: {clamped}")
    
if sensitivity < 0.1:
    print(f"\n[WARNING] Sensitivity {sensitivity} is extremely low")
    print("This may prevent any detections. Try 0.5 for normal use")

# Check access key
access_key = os.getenv('PICOVOICE_ACCESS_KEY')
print(f"\nAccess key present: {'Yes' if access_key else 'No'}")
if access_key:
    print(f"Access key length: {len(access_key)}")

print("\nRecommendation:")
if sensitivity < 0.1:
    print("- Change sensitivity in config.yaml to 0.5")
    print("- Current value 0.000001 is too low for any detections")