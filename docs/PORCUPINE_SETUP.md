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
- For custom wake words, see the Custom Wake Words section below

### "Custom wake word file not found" Error
- Make sure your `.ppn` file is in `config/wake_words/`
- Check that the filename in config.yaml matches exactly (case-sensitive)
- Ensure the file has the `.ppn` extension

### Audio Not Detected
- Porcupine requires 16kHz audio at 512 samples per frame
- The detector automatically handles resampling from 24kHz
- Try adjusting sensitivity (lower = more sensitive)

### Performance Issues
- Porcupine is highly optimized and should use <10% CPU on RPi 4
- If experiencing issues, check other processes using `top`

## Custom Wake Words (Advanced)

The free tier allows training custom wake words. Here's how to use them:

### Creating Custom Wake Words

1. Log into [Picovoice Console](https://console.picovoice.ai/)
2. Click "Create Wake Word" 
3. Enter your phrase (e.g., "Hey Assistant", "Computer", etc.)
4. Select "Raspberry Pi" as the target platform
5. Click "Train" and wait for the model to be generated
6. Download the `.ppn` file from the email or console

### Using Custom Wake Words

1. Place the downloaded `.ppn` file in the `config/wake_words/` directory
2. Update your `config.yaml`:

```yaml
wake_word:
  enabled: true
  model: "my_custom_wake.ppn"  # Use the filename of your .ppn file
  sensitivity: 0.5
```

### Examples

```yaml
# Custom wake word examples
model: "hey_assistant.ppn"     # Custom "Hey Assistant" wake word
model: "jarvis.ppn"           # Custom "Jarvis" wake word  
model: "smart_home.ppn"       # Custom "Smart Home" wake word
```

### Important Notes

- Custom wake word files must have the `.ppn` extension
- Files must be placed in `config/wake_words/` directory
- The free tier models expire after 30 days and need regeneration
- You can create up to 3 custom wake words on the free tier
- For production use, consider the paid tier for permanent models

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