#!/usr/bin/env python3
"""
Test Picovoice pronunciation with real speech
"""
import subprocess
import time

print("Testing Picovoice pronunciations...")

# Different ways to pronounce "picovoice"
pronunciations = [
    ("picovoice", "picovoice as one word"),
    ("pico voice", "pico voice as two words"),
    ("peek oh voice", "phonetic: peek-oh-voice"),
    ("pea co voice", "phonetic: pea-co-voice"),
    ("pick oh voice", "phonetic: pick-oh-voice"),
]

# Generate audio files
for filename, text in pronunciations:
    safe_filename = filename.replace(" ", "_")
    cmd = f'say -o {safe_filename}.aiff "{text}"'
    subprocess.run(cmd, shell=True)
    print(f"Created: {safe_filename}.aiff")

# Now test each one
print("\nStarting Pi test...")
subprocess.Popen(
    'ssh ansible@williams "cd ~/ha-realtime-assist && source venv/bin/activate && source .env && '
    'timeout 60 python scripts/test_wake_word_quick.py > /tmp/pronunciation_test.log 2>&1"',
    shell=True
)

time.sleep(5)  # Let it initialize

# Play each pronunciation
subprocess.run(['osascript', '-e', 'set volume output volume 40'])

for filename, text in pronunciations:
    safe_filename = filename.replace(" ", "_")
    print(f"\nTesting: {text}")
    subprocess.run(['afplay', f'{safe_filename}.aiff'])
    time.sleep(3)

# Wait and check results
time.sleep(5)

# Get results
result = subprocess.run(
    ['ssh', 'ansible@williams', 'grep -c "DETECTED" /tmp/pronunciation_test.log 2>/dev/null || echo 0'],
    capture_output=True, text=True
)
detections = int(result.stdout.strip())

print(f"\nTotal detections: {detections}")

# Show which ones worked
if detections > 0:
    result = subprocess.run(
        ['ssh', 'ansible@williams', 'grep -B5 "DETECTED" /tmp/pronunciation_test.log | grep "Testing:"'],
        capture_output=True, text=True
    )
    print("Successful pronunciations:")
    print(result.stdout)