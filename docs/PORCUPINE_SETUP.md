# Picovoice Porcupine Setup Guide

This guide explains how to set up Picovoice Porcupine wake word detection for the Home Assistant Realtime Voice Assistant.

## Why Porcupine?

We're switching from OpenWakeWord to Porcupine because:
- **11x more accurate** than competitors on Raspberry Pi
- **6.5x faster** performance
- No stuck model issues (which plague OpenWakeWord with TensorFlow Lite)
- Proven reliability in production environments
- Works great on Raspberry Pi Zero and up

## Getting Started

### 1. Get a Free Picovoice Access Key

1. Visit [Picovoice Console](https://console.picovoice.ai/)
2. Sign up for a free account (no credit card required)
3. Copy your AccessKey from the dashboard

### 2. Set the Access Key

Add to your environment:

```bash
export PICOVOICE_ACCESS_KEY="your-access-key-here"
```

Or add to your `.bashrc`:

```bash
echo 'export PICOVOICE_ACCESS_KEY="your-access-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install Porcupine

```bash
pip install pvporcupine
```

### 4. Configure Your Assistant

Update your `config/config.yaml`:

```yaml
wake_word:
  enabled: true
  engine: "porcupine"              # Switch from "openwakeword"
  model: "picovoice"               # See list above for valid options
  sensitivity: 0.5                 # 0.0-1.0 (0.5 is a good default)
  porcupine_access_key: ${PICOVOICE_ACCESS_KEY}  # From environment
```

## Available Wake Words

Porcupine free tier includes these pre-trained models:

### Confirmed Working Keywords
- `alexa`
- `picovoice`
- `computer`
- `americano`
- `blueberry`
- `bumblebee`
- `grapefruit`
- `grasshopper`
- `porcupine`
- `terminator`

### Aliases
- `hey_picovoice` (maps to picovoice)
- `ok_picovoice` (maps to picovoice)

**Important Notes:**
- 'jarvis' and 'hey_jarvis' are NOT built-in wake words! These require creating custom models.
- Multi-word wake words like 'hey google', 'ok google', 'hey siri' may not work as expected with the current implementation.

## Testing

Run the test script to verify your setup:

```bash
python tools/test_porcupine.py
```

## Troubleshooting

### "Initialization failed" Error
- Make sure your access key is valid
- Check that PICOVOICE_ACCESS_KEY is set correctly
- Ensure you're using the key from your Picovoice Console account

### "Invalid wake word" Error
- This happens when you specify a wake word that isn't built-in
- Check the list above for valid wake words
- Common mistake: 'jarvis' is NOT a built-in wake word
- For custom wake words, you need to create them at console.picovoice.ai

### Audio Not Detected
- Porcupine requires 16kHz audio at 512 samples per frame
- The detector automatically handles resampling from 24kHz
- Try adjusting sensitivity (lower = more sensitive)

### Performance Issues
- Porcupine is highly optimized and should use <10% CPU on RPi 4
- If experiencing issues, check other processes using `top`

## Custom Wake Words (Advanced)

The free tier allows training custom wake words:
1. Log into Picovoice Console
2. Click "Create Wake Word"
3. Enter your phrase and select "Raspberry Pi" as platform
4. Download the model file
5. Use the custom model path in your config

Note: Custom models on free tier expire after 30 days and need regeneration.

## Comparison with OpenWakeWord

| Feature | OpenWakeWord | Porcupine |
|---------|--------------|-----------|
| Accuracy | Moderate | Excellent |
| CPU Usage | High | Low |
| Stuck States | Frequent | None |
| Custom Words | Free | Free (30-day limit) |
| Languages | English only | 9 languages |

## License

Porcupine is free for personal use. Commercial use requires a license from Picovoice.