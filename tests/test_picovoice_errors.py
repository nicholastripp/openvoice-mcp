#!/usr/bin/env python3
"""
Test script to simulate Picovoice error messages
Useful for testing error handling without triggering actual Picovoice errors
"""

import sys
import os

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logger import get_logger

def simulate_device_limit_error():
    """Simulate the device activation limit error"""
    logger = get_logger("PicovoiceTest")

    print("\n" + "=" * 70)
    print("SIMULATING: Device Activation Limit Error")
    print("=" * 70 + "\n")

    # This is the exact error handling code from porcupine_detector.py
    logger.error("━" * 70)
    logger.error("⚠️  PICOVOICE DEVICE LIMIT REACHED")
    logger.error("━" * 70)
    logger.error("")
    logger.error("Your Picovoice account has reached its device activation limit.")
    logger.error("")
    logger.error("To fix this:")
    logger.error("  1. Visit: https://console.picovoice.ai/")
    logger.error("  2. Remove old/unused devices from your account")
    logger.error("  3. Or generate a new access key")
    logger.error("")
    logger.error("Alternatively, disable wake word detection:")
    logger.error("  • Edit config/config.yaml")
    logger.error("  • Set wake_word.enabled to false")
    logger.error("━" * 70)

def simulate_invalid_key_error():
    """Simulate the invalid access key error"""
    logger = get_logger("PicovoiceTest")

    print("\n" + "=" * 70)
    print("SIMULATING: Invalid Access Key Error")
    print("=" * 70 + "\n")

    # This is the exact error handling code from porcupine_detector.py
    logger.error("━" * 70)
    logger.error("⚠️  INVALID PICOVOICE ACCESS KEY")
    logger.error("━" * 70)
    logger.error("")
    logger.error("Your Picovoice access key is invalid or has expired.")
    logger.error("")
    logger.error("To fix this:")
    logger.error("  1. Visit: https://console.picovoice.ai/")
    logger.error("  2. Sign in or create a free account")
    logger.error("  3. Generate a new access key")
    logger.error("  4. Update PICOVOICE_ACCESS_KEY in your .env file")
    logger.error("━" * 70)

def main():
    """Run all error simulations"""
    print("\n" + "=" * 70)
    print("PICOVOICE ERROR MESSAGE SIMULATION TEST")
    print("=" * 70)
    print()
    print("This script simulates the error messages that would be displayed")
    print("when Picovoice encounters issues.")
    print()
    print("Check the output below to verify Unicode characters render correctly.")
    print("=" * 70)

    try:
        # Test device limit error
        simulate_device_limit_error()

        # Test invalid key error
        simulate_invalid_key_error()

        print("\n" + "=" * 70)
        print("✓ ALL ERROR MESSAGES RENDERED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("If you see:")
        print("  ✓ Clean box lines (━)")
        print("  ✓ Warning symbol (⚠️)")
        print("  ✓ Bullet points (•)")
        print()
        print("Then Unicode support is working correctly!")
        print()

        return True

    except UnicodeEncodeError as e:
        print("\n" + "=" * 70)
        print("✗ UNICODE ENCODING ERROR DETECTED")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Your terminal does not support UTF-8 encoding.")
        print("The error messages will not display correctly.")
        print()
        print("Recommended fixes:")
        print("  1. Run: export PYTHONIOENCODING=utf-8")
        print("  2. Or configure locale: sudo raspi-config")
        print("  3. Or implement ASCII fallback in code")
        print()

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
