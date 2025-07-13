#!/bin/bash
# Quick audio test for wake word detection

echo "Quick Wake Word Audio Test"
echo "========================="

# Set volume
osascript -e 'set volume output volume 40'

echo "Starting Pi listener..."
ssh ansible@williams "cd ~/ha-realtime-assist && source venv/bin/activate && source .env && timeout 30 python scripts/test_wake_word_quick.py" &
PI_PID=$!

# Wait for initialization
sleep 5

echo -e "\nPlaying wake words..."

# Play picovoice
echo -n "Playing 'picovoice'... "
afplay picovoice_normal.aiff
echo "done"
sleep 2

echo -n "Playing 'picovoice' (slow)... "
afplay picovoice_slow.aiff
echo "done"
sleep 2

echo -n "Playing 'picovoice' (clear)... "
afplay picovoice_clear.aiff
echo "done"
sleep 2

# Test phrase to check audio levels
echo -n "Playing test phrase... "
afplay test_phrase.aiff
echo "done"

# Wait for Pi test to complete
echo -e "\nWaiting for results..."
wait $PI_PID

echo "Test complete!"