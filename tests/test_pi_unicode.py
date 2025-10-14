#!/usr/bin/env python3
"""
Test script to verify Unicode support on Raspberry Pi
Run this on the Pi to check if the Picovoice error messages will display correctly
"""

import sys
import locale

def test_encoding():
    """Test system encoding support"""
    print("=" * 70)
    print("RASPBERRY PI UNICODE SUPPORT TEST")
    print("=" * 70)
    print()

    # Check Python encoding
    print("Python Encoding Info:")
    print(f"  sys.getdefaultencoding(): {sys.getdefaultencoding()}")
    print(f"  sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"  sys.stderr.encoding: {sys.stderr.encoding}")
    print()

    # Check locale
    print("Locale Info:")
    try:
        current_locale = locale.getlocale()
        print(f"  Current locale: {current_locale}")
        print(f"  Preferred encoding: {locale.getpreferredencoding()}")
    except Exception as e:
        print(f"  Error getting locale: {e}")
    print()

    # Test Unicode characters used in error messages
    print("Unicode Character Tests:")
    print()

    test_cases = [
        ("Box drawing (━)", "━" * 70),
        ("Warning emoji (⚠️)", "⚠️  WARNING"),
        ("Bullet (•)", "  • Item 1\n  • Item 2"),
        ("Check mark (✓)", "✓ Success"),
        ("Cross mark (✗)", "✗ Failed"),
    ]

    for name, test_str in test_cases:
        print(f"Testing {name}:")
        try:
            print(f"  {test_str}")
            print(f"  ✓ Rendered successfully")
        except UnicodeEncodeError as e:
            print(f"  ✗ FAILED: {e}")
        print()

    # Simulate the actual Picovoice error message
    print("=" * 70)
    print("SIMULATED PICOVOICE ERROR MESSAGE:")
    print("=" * 70)
    print()

    try:
        print("━" * 70)
        print("⚠️  PICOVOICE DEVICE LIMIT REACHED")
        print("━" * 70)
        print("")
        print("Your Picovoice account has reached its device activation limit.")
        print("")
        print("To fix this:")
        print("  1. Visit: https://console.picovoice.ai/")
        print("  2. Remove old/unused devices from your account")
        print("  3. Or generate a new access key")
        print("")
        print("Alternatively, disable wake word detection:")
        print("  • Edit config/config.yaml")
        print("  • Set wake_word.enabled to false")
        print("━" * 70)
        print()
        print("✓ Full error message rendered successfully!")
    except UnicodeEncodeError as e:
        print(f"✗ ERROR: Cannot render error message: {e}")
        print()
        print("ASCII FALLBACK VERSION:")
        print("-" * 70)
        print("[!] PICOVOICE DEVICE LIMIT REACHED")
        print("-" * 70)
        print()

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()

    # Recommendations
    if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
        print("⚠️  WARNING: Terminal encoding is not UTF-8")
        print()
        print("Recommended fixes:")
        print("  1. Set environment variable: export PYTHONIOENCODING=utf-8")
        print("  2. Configure locale: sudo raspi-config → Localisation Options → en_US.UTF-8")
        print("  3. Add to ~/.bashrc: export LC_ALL=en_US.UTF-8")
        print()
        return False
    else:
        print("✓ Terminal encoding supports UTF-8")
        print("✓ Picovoice error messages should display correctly")
        print()
        return True

if __name__ == "__main__":
    success = test_encoding()
    sys.exit(0 if success else 1)
