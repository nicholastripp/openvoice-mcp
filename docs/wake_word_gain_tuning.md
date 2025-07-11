# Wake Word Gain Tuning Guide

## Overview
The wake word detection system uses audio gain (amplification) to improve detection reliability. This guide helps you find the optimal gain setting for your environment.

## Configuration
In your `config.yaml` file, under the `wake_word` section:

```yaml
wake_word:
  audio_gain: 3.5                # Audio amplification factor (1.0-5.0)
  audio_gain_mode: "fixed"       # "fixed" or "dynamic"
```

## Gain Settings

### Fixed Gain Mode (Recommended)
- **Default**: 3.5x
- **Range**: 1.0 - 5.0
- Provides consistent amplification regardless of input volume
- Most predictable behavior

### Dynamic Gain Mode (Advanced)
- Automatically adjusts gain based on input level
- Prevents the inverse threshold relationship issue
- Currently uses hardcoded parameters (2.0x - 5.0x range)

## Tuning Process

1. **Start with default (3.5x)**
   - This has been tested to work well in most environments

2. **If wake word not detected reliably:**
   - Increase to 4.0x
   - Maximum recommended: 4.5x

3. **If getting false positives:**
   - Decrease to 3.0x or 2.5x
   - Or increase the sensitivity threshold

## Testing Your Settings

Use the provided test script:
```bash
python test_gain_settings.py
```

This will show:
- Gain calculations for different input levels
- Actual model predictions (on Raspberry Pi only)

## Optimal Values

Based on testing:
- **Final RMS after gain**: 0.1 - 0.2 (optimal range)
- **Below 0.05**: May not detect reliably
- **Above 0.2**: May cause clipping/distortion

## Example Configurations

### Quiet Environment
```yaml
audio_gain: 4.0
audio_gain_mode: "fixed"
```

### Normal Environment (Default)
```yaml
audio_gain: 3.5
audio_gain_mode: "fixed"
```

### Noisy Environment
```yaml
audio_gain: 3.0
audio_gain_mode: "fixed"
```

### Variable Conditions
```yaml
audio_gain: 3.5
audio_gain_mode: "dynamic"
```

## Troubleshooting

- **Still not detecting?** Check your microphone input volume in system settings
- **Too many false positives?** Reduce gain or increase sensitivity threshold
- **Inconsistent detection?** Try dynamic mode or adjust your speaking volume