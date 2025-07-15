# Wake Word Setup Guide

Configure wake word detection for reliable hands-free activation of your assistant.

## Wake Word Engines

The assistant supports two wake word engines:

### 1. Porcupine (Recommended)
- **Pros**: More accurate, better noise handling, built-in keywords
- **Cons**: Requires access key (free tier available)
- **Keywords**: picovoice, alexa, computer, terminator, and more

### 2. OpenWakeWord
- **Pros**: Completely free, no API key needed
- **Cons**: Requires model downloads, higher CPU usage
- **Keywords**: hey_jarvis, alexa, hey_mycroft, ok_nabu

## Basic Configuration

### Using Porcupine (Default)

1. Get a free access key from [console.picovoice.ai](https://console.picovoice.ai)

2. Add to `.env`:
   ```bash
   PICOVOICE_ACCESS_KEY=your-access-key-here
   ```

3. Configure in `config.yaml`:
   ```yaml
   wake_word:
     enabled: true
     engine: "porcupine"
     model: "picovoice"  # or any built-in keyword
     sensitivity: 1.0    # 0.0-1.0 (1.0 = most sensitive)
   ```

### Using OpenWakeWord

1. Configure in `config.yaml`:
   ```yaml
   wake_word:
     enabled: true
     engine: "openwakeword"
     model: "hey_jarvis"  # or other available model
     sensitivity: 0.5     # 0.0-1.0 (0.5 = balanced)
     auto_download: true  # Download models automatically
   ```

2. Models will download on first run, or manually:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate
   
   # Then download models
   python download_wake_word_models.py --download-all
   ```

## Available Wake Words

### Porcupine Built-in Keywords
- `alexa` - "Alexa"
- `americano` - "Americano" 
- `blueberry` - "Blueberry"
- `bumblebee` - "Bumblebee"
- `computer` - "Computer"
- `grapefruit` - "Grapefruit"
- `grasshopper` - "Grasshopper"
- `picovoice` - "Picovoice" (default)
- `porcupine` - "Porcupine"
- `terminator` - "Terminator"

### OpenWakeWord Models
- `alexa` - "Alexa"
- `hey_jarvis` - "Hey Jarvis"
- `hey_mycroft` - "Hey Mycroft"
- `hey_rhasspy` - "Hey Rhasspy"
- `ok_nabu` - "Ok Nabu"

## Sensitivity Tuning

### Understanding Sensitivity

- **Higher sensitivity** (0.8-1.0): Triggers more easily, may have false positives
- **Lower sensitivity** (0.1-0.3): Requires clearer pronunciation, fewer false triggers
- **Balanced** (0.4-0.6): Good for most environments

### Tuning Process

1. Start with default sensitivity:
   ```yaml
   sensitivity: 1.0  # Porcupine
   # or
   sensitivity: 0.5  # OpenWakeWord
   ```

2. Test in your environment:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate
   
   # Then test interactively
   python examples/test_wake_word.py --interactive
   ```

3. Adjust based on results:
   - **Too many false triggers**: Decrease by 0.1
   - **Hard to trigger**: Increase by 0.1

4. Test with different:
   - Distances (1-10 feet)
   - Background noise levels
   - Different speakers/voices

## Audio Settings for Wake Words

### Porcupine-Specific Settings

```yaml
wake_word:
  # Audio preprocessing
  highpass_filter_enabled: true  # REQUIRED for Porcupine
  highpass_filter_cutoff: 80.0   # Hz - removes low rumble
  
  # Gain settings
  audio_gain: 1.0               # Amplification (1.0 = no change)
  audio_gain_mode: "fixed"      # or "dynamic"
```

### OpenWakeWord Settings

```yaml
wake_word:
  # Noise suppression (if available)
  speex_noise_suppression: true  # Requires speexdsp
  vad_enabled: true             # Voice activity detection
  
  # Gain settings
  audio_gain: 1.0
  audio_gain_mode: "fixed"
```

## Custom Wake Words

### Creating Custom Porcupine Wake Words

1. Visit [Picovoice Console](https://console.picovoice.ai)
2. Click "Create Wake Word"
3. Enter your phrase (e.g., "Hey Assistant")
4. Train the model with voice samples
5. Download the `.ppn` file
6. Place in `config/wake_words/`
7. Update configuration:
   ```yaml
   wake_word:
     model: "config/wake_words/Hey-Assistant_en_rpi_v2_1_0.ppn"
   ```

### Creating Custom OpenWakeWord Models

Currently, OpenWakeWord doesn't support easy custom model creation. Use pre-trained models or consider Porcupine for custom words.

## Troubleshooting Wake Word Detection

### Problem: Wake Word Not Detecting

**Check these in order**:

1. **Audio Input Working**:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate
   
   # Then test audio
   python examples/test_audio_devices.py
   ```

2. **Wake Word Test Mode**:
   ```yaml
   wake_word:
     test_mode: true  # Prints detection scores
   ```

3. **Audio Gain**:
   ```yaml
   wake_word:
     audio_gain: 1.5  # Increase if too quiet
   ```

4. **High-Pass Filter** (Porcupine only):
   ```yaml
   wake_word:
     highpass_filter_enabled: true  # MUST be true
   ```

### Problem: Too Many False Positives

1. **Decrease Sensitivity**:
   ```yaml
   sensitivity: 0.3  # Lower values = fewer false triggers
   ```

2. **Increase Cooldown**:
   ```yaml
   cooldown: 3.0  # Seconds between detections
   ```

3. **Try Different Wake Word**:
   - Longer phrases work better
   - Avoid common words

### Problem: Specific Voice Not Working

1. **Test Multiple Speakers**:
   - Different accents
   - Male/female voices
   - Children vs adults

2. **Adjust Audio Settings**:
   ```yaml
   audio:
     input_volume: 1.5  # Boost quiet voices
     agc_enabled: true  # Auto-adjust levels
   ```

3. **Consider Custom Wake Word**:
   - Train with specific voice
   - Include variations

## Performance Optimization

### CPU Usage

Monitor CPU during detection:
```bash
top -p $(pgrep -f main.py)
```

Reduce CPU usage:
```yaml
wake_word:
  # Reduce processing frequency
  chunk_size: 1600  # Larger chunks = less CPU
```

### Memory Usage

- Porcupine: ~10MB per keyword
- OpenWakeWord: ~50MB per model

### Latency

Reduce detection latency:
```yaml
wake_word:
  # Smaller chunks = faster response
  chunk_size: 960   # 60ms chunks
  
  # Disable unnecessary features
  vad_enabled: false
  speex_noise_suppression: false
```

## Advanced Configuration

### Multiple Wake Words

For Porcupine with multiple keywords:
```yaml
wake_word:
  model: "picovoice,alexa,computer"  # Comma-separated
  # Each can have different sensitivity
```

### Environment-Specific Settings

Create different configs for different rooms:
```yaml
# config/kitchen.yaml - Noisy environment
wake_word:
  sensitivity: 0.8
  audio_gain: 2.0
  speex_noise_suppression: true

# config/bedroom.yaml - Quiet environment  
wake_word:
  sensitivity: 0.3
  audio_gain: 1.0
```

### Debugging Wake Word

Enable detailed logging:
```yaml
system:
  log_level: "DEBUG"

wake_word:
  test_mode: true  # Shows confidence scores
```

Watch the logs:
```bash
tail -f logs/assistant.log | grep -i wake
```

## Best Practices

1. **Choose Distinctive Wake Words**: Avoid common words in conversation
2. **Test Thoroughly**: Test with multiple people and environments
3. **Start Sensitive**: Begin with high sensitivity and reduce if needed
4. **Position Matters**: Place microphone centrally with clear line of sight
5. **Regular Testing**: Periodically verify detection is working well

## Quick Reference

### Porcupine Setup
```yaml
wake_word:
  enabled: true
  engine: "porcupine"
  model: "computer"
  sensitivity: 1.0
  highpass_filter_enabled: true
  porcupine_access_key: ${PICOVOICE_ACCESS_KEY}
```

### OpenWakeWord Setup
```yaml
wake_word:
  enabled: true
  engine: "openwakeword"
  model: "hey_jarvis"
  sensitivity: 0.5
  auto_download: true
  speex_noise_suppression: true
```

## Getting Help

- Test wake word: 
  ```bash
  # Activate virtual environment first
  source venv/bin/activate
  python examples/test_wake_word.py
  ```
- Check [Audio Setup](AUDIO_SETUP.md) for microphone issues
- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
- Enable debug logging for detailed diagnostics