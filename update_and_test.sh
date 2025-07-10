#!/bin/bash
# Update code and test wake word detection

echo "Home Assistant Realtime Voice Assistant - Update and Test"
echo "========================================================="
echo ""

# Update code
echo "1. Updating code from GitHub..."
git pull origin main

if [ $? -ne 0 ]; then
    echo "[FAIL] Could not update code. Check your internet connection."
    exit 1
fi

echo "[OK] Code updated successfully"
echo ""

# Check Python version
echo "2. Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "[WARNING] Not in virtual environment!"
    echo "Please activate with: source venv/bin/activate"
    echo ""
fi

# Test wake word mode
echo "3. Testing wake word detection mode..."
echo "   This will run in test mode (no network required)"
echo ""
echo "   Say 'Alexa' to test wake word detection"
echo "   Press Ctrl+C to stop"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 src/main.py --test-mode