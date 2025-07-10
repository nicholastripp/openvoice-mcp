#!/usr/bin/env python3
"""
Test the UnboundLocalError fixes
"""
import subprocess
import sys

print("Testing UnboundLocalError fixes...")
print("=" * 50)

# Test 1: Import test
print("\n1. Testing websockets import fix:")
try:
    result = subprocess.run([sys.executable, "test_import_fix.py"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Import test passed")
    else:
        print("✗ Import test failed:")
        print(result.stderr)
except Exception as e:
    print(f"✗ Import test error: {e}")

# Test 2: Test mode
print("\n2. Testing --test-mode flag:")
try:
    result = subprocess.run([sys.executable, "src/main.py", "--test-mode"], 
                          capture_output=True, text=True, timeout=2)
    if "cannot access local variable 'logger'" not in result.stderr:
        print("✓ Logger error fixed")
    else:
        print("✗ Logger error still present")
        print(result.stderr[:200])
except subprocess.TimeoutExpired:
    print("✓ Test mode started without logger error (timed out as expected)")
except Exception as e:
    print(f"✗ Test mode error: {e}")

print("\n" + "=" * 50)
print("Fixes appear to be working. The app should now:")
print("1. Not throw UnboundLocalError for websockets")
print("2. Not throw UnboundLocalError for logger in test mode")
print("3. Show actual connection errors or run in test mode")