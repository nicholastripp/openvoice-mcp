# Wake Word Setup Guide

Configure wake word detection using Picovoice Porcupine for reliable hands-free activation.

## Overview

The assistant uses Picovoice Porcupine for wake word detection. Porcupine offers:
- High accuracy with low false positive rates
- Built-in keywords that work out of the box
- Support for custom wake words
- Optimized for Raspberry Pi

## Getting Started

### 1. Get a Picovoice Access Key

Porcupine requires a free access key:

1. Sign up at [console.picovoice.ai](https://console.picovoice.ai)
2. Create a new project
3. Copy your access key
4. Add to `.env` file:
   ```bash
   PICOVOICE_ACCESS_KEY=your-access-key-here
   ```

### 2. Basic Configuration

In `config/config.yaml`:

```yaml
wake_word:
  enabled: true
  model: "picovoice"    # Built-in keyword
  sensitivity: 1.0      # 0.0-1.0 (1.0 = most sensitive)
```

## Built-in Keywords

Porcupine includes these keywords without any downloads:

| Keyword | Example Phrase |
|---------|----------------|
| `alexa` | "Alexa" |
| `americano` | "Americano" |
| `blueberry` | "Blueberry" |
| `bumblebee` | "Bumblebee" |
| `computer` | "Computer" |
| `grapefruit` | "Grapefruit" |
| `grasshopper` | "Grasshopper" |
| `hey google` | "Hey Google" |
| `hey siri` | "Hey Siri" |
| `jarvis` | "Jarvis" |
| `ok google` | "Ok Google" |
| `picovoice` | "Picovoice" (default) |
| `porcupine` | "Porcupine" |
| `terminator` | "Terminator" |

## Sensitivity Tuning

The sensitivity parameter controls how easily the wake word triggers:

- **1.0** (Maximum): Most sensitive, may have occasional false positives
- **0.5** (Balanced): Good balance for most environments
- **0.1** (Minimum): Requires very clear pronunciation

### Testing Sensitivity

```bash
# Activate virtual environment
source venv/bin/activate

# Test wake word detection
python examples/test_wake_word.py
```

Adjust sensitivity based on your environment:
- Quiet room: 0.3 - 0.5
- Normal environment: 0.5 - 0.7
- Noisy environment: 0.7 - 1.0

## Audio Configuration

### High-Pass Filter (Required)

Porcupine requires a high-pass filter to work properly:

```yaml
wake_word:
  highpass_filter_enabled: true   # Must be true
  highpass_filter_cutoff: 80.0    # Hz
```

This removes low-frequency noise and DC offset that can interfere with detection.

### Audio Gain

Adjust the wake word audio gain if needed:

```yaml
wake_word:
  audio_gain: 1.0        # 1.0 = no change
  audio_gain_mode: "fixed"
```

- Increase if wake word is hard to trigger
- Decrease if you get false positives
- Start with 1.0 and adjust as needed

## Custom Wake Words

### Creating Custom Keywords

1. Visit [Picovoice Console](https://console.picovoice.ai)
2. Click "Train Custom Wake Word"
3. Enter your phrase (e.g., "Hey Assistant")
4. Record training samples
5. Download the `.ppn` file

### Using Custom Keywords

1. Create directory for custom models:
   ```bash
   mkdir -p config/wake_words
   ```

2. Place your `.ppn` file in the directory

3. Update configuration:
   ```yaml
   wake_word:
     model: "config/wake_words/Hey-Assistant_en_raspberry-pi_v3_0_0.ppn"
   ```

## Troubleshooting

### Wake Word Not Detecting

1. **Check Access Key**:
   ```bash
   echo $PICOVOICE_ACCESS_KEY
   ```
   Ensure it's set in your `.env` file

2. **Verify Audio Input**:
   ```bash
   python examples/test_audio_devices.py
   ```

3. **Test with Maximum Sensitivity**:
   ```yaml
   sensitivity: 1.0
   audio_gain: 1.5
   ```

4. **Check Logs**:
   ```bash
   grep -i porcupine logs/assistant.log
   ```

### Too Many False Positives

1. **Reduce Sensitivity**:
   ```yaml
   sensitivity: 0.3
   ```

2. **Increase Cooldown**:
   ```yaml
   cooldown: 3.0  # Seconds between detections
   ```

3. **Try Different Keyword**:
   - Longer phrases work better
   - Unique sounds reduce false positives

### Platform-Specific Issues

#### Raspberry Pi
- Use keywords ending with `_raspberry-pi` for best performance
- Ensure you have sufficient CPU headroom
- Consider using a USB microphone for better quality

#### Access Key Errors
- "Invalid access key": Check key is correct and active
- "Exceeded quota": Free tier allows generous usage, check console
- "Unsupported platform": Ensure using correct platform-specific model

## Performance Optimization

### CPU Usage
- Porcupine uses ~5-10% CPU on Raspberry Pi 4
- Less on more powerful systems

### Memory Usage
- ~10MB per keyword
- Very efficient compared to alternatives

### Reduce Latency
```yaml
wake_word:
  vad_enabled: false  # Disable if not needed
```

## Best Practices

1. **Choose Distinctive Keywords**: Avoid common words in conversation
2. **Test in Your Environment**: What works in one room may not in another
3. **Consider Multiple Users**: Test with different voices
4. **Monitor Logs**: Check for false positives and missed detections
5. **Start Conservative**: Begin with lower sensitivity and increase as needed

## Example Configurations

### Living Room Assistant
```yaml
wake_word:
  model: "alexa"        # Familiar to family
  sensitivity: 0.7      # Balanced for normal conversation
  cooldown: 2.0
```

### Workshop Assistant
```yaml
wake_word:
  model: "computer"     # Clear, distinctive
  sensitivity: 0.9      # High for noisy environment
  audio_gain: 1.5       # Boost for distance
```

### Bedroom Assistant
```yaml
wake_word:
  model: "jarvis"       # Fun choice
  sensitivity: 0.4      # Low to avoid false triggers
  cooldown: 3.0         # Prevent accidental triggers
```

## Need Help?

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [Audio Setup](AUDIO_SETUP.md) for microphone configuration
- Visit [Picovoice Docs](https://picovoice.ai/docs/) for advanced features