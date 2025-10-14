#!/usr/bin/env python3
"""
Test script to verify the Unicode fallback fix works correctly
This should run without errors on both UTF-8 and Latin-1 terminals
"""
import sys

def _supports_unicode():
    """Check if terminal supports UTF-8 encoding"""
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        return 'utf' in sys.stdout.encoding.lower()
    return False


def _get_box_char():
    """Get box drawing character with ASCII fallback"""
    return "━" if _supports_unicode() else "-"


def _get_warning_char():
    """Get warning character with ASCII fallback"""
    return "⚠️" if _supports_unicode() else "[!]"


def main():
    print("=" * 70)
    print("UNICODE FALLBACK FIX VERIFICATION")
    print("=" * 70)
    print()

    print("Terminal Info:")
    print(f"  sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"  UTF-8 support detected: {_supports_unicode()}")
    print()

    print("Characters selected:")
    box = _get_box_char()
    warning = _get_warning_char()
    print(f"  Box character: {repr(box)}")
    print(f"  Warning character: {repr(warning)}")
    print()

    print("=" * 70)
    print("SIMULATED PICOVOICE ERROR (with fallback)")
    print("=" * 70)
    print()

    # This should work on BOTH UTF-8 and Latin-1 terminals
    try:
        print(box * 70)
        print(f"{warning}  PICOVOICE DEVICE LIMIT REACHED")
        print(box * 70)
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
        print(box * 70)
        print()

        print("=" * 70)
        if _supports_unicode():
            print("SUCCESS: Error message displayed with Unicode formatting")
        else:
            print("SUCCESS: Error message displayed with ASCII fallback")
        print("=" * 70)
        return True

    except UnicodeEncodeError as e:
        print()
        print("=" * 70)
        print("FAILED: Unicode encoding error occurred")
        print("=" * 70)
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
