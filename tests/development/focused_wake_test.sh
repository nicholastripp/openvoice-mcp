#!/bin/bash
# Focused wake word test with gain 1.0

echo "Focused Wake Word Test (Gain 1.0)"
echo "================================="

# Start Pi test
echo "Starting Pi listener..."
ssh ansible@williams "cd ~/ha-realtime-assist && source venv/bin/activate && source .env && python scripts/test_wake_word_quick.py" &
PI_PID=$!

# Wait for initialization
sleep 5

# Set moderate volume
osascript -e 'set volume output volume 35'

echo -e "\nTesting pronunciations:"

# Test different pronunciations
echo -n "1. 'pea co voice'... "
say "pea co voice"
sleep 3
echo "done"

echo -n "2. 'peek oh voice'... "
say "peek oh voice"
sleep 3
echo "done"

echo -n "3. 'pico voice' (two words)... "
say "pico voice"
sleep 3
echo "done"

echo -n "4. 'picovoice' (one word)... "
say "picovoice"
sleep 3
echo "done"

# Higher volume test
osascript -e 'set volume output volume 50'
echo -e "\nHigher volume test:"

echo -n "5. 'pea co voice' (louder)... "
say "pea co voice"
sleep 3
echo "done"

# Wait for completion
echo -e "\nWaiting for results..."
wait $PI_PID

echo "Test complete!"