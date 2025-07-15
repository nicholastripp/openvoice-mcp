# Audio Setup Guide

Proper audio configuration is crucial for reliable wake word detection and clear communication with the assistant.

## Audio Hardware Recommendations

### USB Microphones (Recommended)
- **Budget**: TONOR TC-777, Blue Snowball iCE
- **Mid-range**: Jabra Speak 410/510, Anker PowerConf
- **Premium**: Blue Yeti, RODE NT-USB Mini

### Microphone Arrays
- **ReSpeaker 2-Mic HAT** - Designed for Raspberry Pi
- **ReSpeaker 4-Mic Array** - Better noise cancellation
- **Matrix Voice** - Advanced 8-mic array

### Speakers
- **3.5mm Jack**: Any powered speakers
- **USB Speakers**: Better quality, independent volume
- **Bluetooth**: Supported but may add latency

## Finding Your Audio Devices

List all available audio devices:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then run the test
python examples/test_audio_devices.py
```

Example output:
```
Input Devices:
  0: USB Microphone (2 channels, 48000Hz)
  1: Built-in Microphone (1 channel, 44100Hz)

Output Devices:
  2: USB Speaker (2 channels, 48000Hz)
  3: Built-in Output (2 channels, 44100Hz)
```

## Basic Configuration

Edit `config/config.yaml`:

```yaml
audio:
  input_device: "USB Microphone"  # or device index like 0
  output_device: "USB Speaker"    # or device index like 2
  sample_rate: 48000             # Match your device's native rate
  channels: 1                    # 1 for mono, 2 for stereo
  input_volume: 1.0             # Start with 1.0, adjust as needed
  output_volume: 2.0            # Speaker volume multiplier
```

## Volume Configuration

### Manual Volume Control

The `input_volume` setting controls microphone gain:
- **0.1 - 0.9**: Reduces volume (for loud mics)
- **1.0**: No change (default)
- **1.1 - 5.0**: Amplifies volume (for quiet mics)

#### Finding the Right Volume

1. Start with `input_volume: 1.0`
2. Run the assistant with debug logging
3. Watch for clipping warnings:
   ```
   *** AUDIO CLIPPING: 15.2% of samples clipped at input ***
   ```
4. If clipping occurs, reduce volume (e.g., 0.7)
5. If wake word doesn't trigger, increase volume (e.g., 1.5)

### Automatic Gain Control (AGC)

AGC automatically adjusts volume to maintain optimal levels:

```yaml
audio:
  agc_enabled: true              # Enable AGC
  agc_target_rms: 0.3           # Target level (30% of max)
  agc_max_gain: 3.0             # Maximum amplification
  agc_min_gain: 0.1             # Minimum gain
  agc_attack_time: 0.5          # Fast response to loud sounds
  agc_release_time: 2.0         # Slow recovery for quiet
  agc_clipping_threshold: 0.05  # Max 5% clipping allowed
```

#### When to Use AGC
- Multiple users with different voice levels
- Varying distances from microphone
- Changing background noise conditions
- USB microphones with inconsistent gain

## Wake Word Audio Settings

Wake word detection has separate gain control:

```yaml
wake_word:
  audio_gain: 1.0              # Wake word specific gain
  audio_gain_mode: "fixed"     # or "dynamic" for auto-adjust
  
  # Porcupine-specific settings
  highpass_filter_enabled: true  # Required for Porcupine
  highpass_filter_cutoff: 80.0   # Hz - removes rumble/DC offset
```

### Wake Word Gain Tuning

1. **Too Sensitive** (false triggers):
   - Reduce `audio_gain` to 0.8 or 0.7
   - Increase `sensitivity` threshold

2. **Not Sensitive Enough**:
   - Increase `audio_gain` to 1.2 or 1.5
   - Decrease `sensitivity` threshold
   - Ensure `highpass_filter_enabled: true`

## Audio Quality Diagnostics

### Run Audio Diagnostics
```bash
# Activate virtual environment first
source venv/bin/activate

# Then run diagnostics
python tests/development/test_audio_diagnostics.py
```

This will show:
- Current audio levels
- Clipping detection
- Suggested gain adjustments
- Audio quality metrics

### Monitor Audio in Real-Time

Enable debug logging to see audio statistics:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then run with debug logging
python src/main.py --log-level DEBUG
```

Look for:
```
Audio level: RMS=0.045, Peak=0.234
AGC: gain=1.5, clipping=0.0%
Audio format conversion - Float32[-0.123, 0.456] -> PCM16[-4029, 14959]
```

## Troubleshooting Audio Issues

### Problem: Audio Clipping/Distortion

**Symptoms**: Crackling, distorted audio, wake word only works with soft speech

**Solutions**:
1. Reduce `input_volume` below 1.0 (e.g., 0.5)
2. Enable AGC for automatic adjustment
3. Move microphone further from mouth
4. Check for multiple gain stages in your setup

### Problem: Wake Word Not Detecting

**Symptoms**: Have to shout or repeat wake word multiple times

**Solutions**:
1. Increase `input_volume` (e.g., 2.0)
2. Ensure `highpass_filter_enabled: true` for Porcupine
3. Test with different wake words
4. Check microphone placement

### Problem: Echo/Feedback

**Symptoms**: Assistant hears itself, continuous triggering

**Solutions**:
1. Enable feedback prevention:
   ```yaml
   audio:
     feedback_prevention: true
     mute_during_response: true
   ```
2. Increase speaker distance from microphone
3. Reduce `output_volume`
4. Use headphones for testing

### Problem: Latency/Delay

**Symptoms**: Long delay between speaking and response

**Solutions**:
1. Use USB audio devices (lower latency)
2. Set optimal `chunk_size` (1200 recommended)
3. Use wired Ethernet instead of WiFi
4. Check CPU usage (`top` command)

## Advanced Audio Configuration

### Sample Rate Optimization

Match your hardware's native rate:
```yaml
audio:
  sample_rate: 48000  # Common for USB devices
  # sample_rate: 44100  # Common for built-in audio
  # sample_rate: 16000  # Some voice-specific mics
```

### Buffer Size Tuning

For Raspberry Pi optimization:
```yaml
audio:
  chunk_size: 1200  # 50ms at 24kHz (recommended)
  # chunk_size: 2400  # 100ms - higher latency, more stable
  # chunk_size: 600   # 25ms - lower latency, may cause issues
```

### Multi-Microphone Setup

For microphone arrays:
```yaml
audio:
  channels: 1  # Still use mono
  input_device: "ReSpeaker 4 Mic Array"
  # Array will handle beam-forming internally
```

## Testing Your Configuration

1. **Test Audio Levels**:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate
   
   # Then test levels
   python examples/test_audio_devices.py --test-levels
   ```

2. **Test Wake Word**:
   ```bash
   # With virtual environment activated
   python examples/test_wake_word.py --interactive
   ```

3. **Full Integration Test**:
   ```bash
   # With virtual environment activated
   python examples/test_full_integration.py
   ```

## Best Practices

1. **Start Conservative**: Begin with default settings and adjust gradually
2. **One Change at a Time**: Only modify one setting per test
3. **Document Working Config**: Save your working configuration
4. **Environment Matters**: Test in your actual use environment
5. **Monitor Logs**: Watch for clipping and level warnings

## Need Help?

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [Wake Word Setup](WAKE_WORD_SETUP.md) for detection issues
- See example configurations in `config/config.yaml.example`