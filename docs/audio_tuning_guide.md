# Audio Tuning Guide

This guide helps you optimize audio settings to prevent distortion and improve wake word detection and speech recognition accuracy.

## Common Audio Issues

### 1. Wake Word Not Detecting Normal Speech
**Symptoms**: Only detects when speaking very softly, misses normal volume speech

**Cause**: Audio distortion from multiple gain stages causing clipping

**Solution**:
```yaml
# In config.yaml
audio:
  input_volume: 1.0  # Reduce to 0.8 or 0.5 if still clipping

wake_word:
  audio_gain: 1.0  # Keep at 1.0 (no amplification)
  audio_gain_mode: "fixed"  # Avoid dynamic gain
```

### 2. OpenAI Misunderstanding Speech
**Symptoms**: AI responses indicate it couldn't understand, asks to repeat

**Cause**: Distorted audio being sent to OpenAI from excessive normalization

**Solution**: The code has been updated to reduce aggressive normalization. Ensure you're using the latest version.

### 3. Audio Clipping/Distortion
**Symptoms**: Harsh, distorted sound in recordings or playback

**Cause**: Input level too high, causing clipping at multiple stages

**Solution**:
1. Run the audio pipeline test:
   ```bash
   python tools/test_audio_pipeline.py
   ```
2. Adjust `input_volume` based on results
3. If your microphone is very loud, reduce system microphone gain

## Recommended Settings

### For Most Users
```yaml
audio:
  input_volume: 1.0  # No gain by default
  
wake_word:
  audio_gain: 1.0  # No additional gain
  audio_gain_mode: "fixed"
```

### For Quiet Microphones
```yaml
audio:
  input_volume: 1.5  # Gentle boost at input
  
wake_word:
  audio_gain: 1.0  # Still no wake word gain
  audio_gain_mode: "fixed"
```

### For Loud/Sensitive Microphones
```yaml
audio:
  input_volume: 0.5  # Reduce input level
  
wake_word:
  audio_gain: 1.0
  audio_gain_mode: "fixed"
```

## Audio Processing Pipeline

Understanding the audio flow helps diagnose issues:

1. **Microphone Input** → Captured at device rate (e.g., 48kHz)
2. **Input Gain** → Multiplied by `audio.input_volume`
3. **Resampling** → Converted to 24kHz for OpenAI
4. **PCM16 Conversion** → Float to 16-bit integer
5. **Wake Word Processing**:
   - Normalized to float
   - Gain applied (if configured)
   - Soft limiting to prevent clipping
   - Resampled to 16kHz
6. **OpenAI Processing**:
   - Normalized for consistent levels
   - Limited gain (max 3x) 
   - Soft limiting applied

## Testing Your Configuration

1. **Test Audio Levels**:
   ```bash
   python tools/test_audio_pipeline.py
   ```
   This shows audio levels at each processing stage.

2. **Test Wake Word**:
   ```bash
   python -m src.main --test-mode
   ```
   This runs wake word detection only (no OpenAI connection).

3. **Monitor Logs**:
   Watch for clipping warnings:
   ```
   *** AUDIO CLIPPING: X.X% of samples clipped at input ***
   ```

## Troubleshooting Tips

1. **Start Conservative**: Begin with all gains at 1.0 and only increase if needed
2. **One Change at a Time**: Adjust only one setting and test thoroughly
3. **Check System Audio**: Ensure system microphone gain isn't too high
4. **Use Fixed Gain**: Dynamic gain can cause unpredictable behavior
5. **Monitor CPU**: High gain + resampling can increase CPU usage on Pi

## Advanced Tuning

### Wake Word Sensitivity
```yaml
wake_word:
  sensitivity: 0.001  # Lower = fewer false positives, higher = more sensitive
```

### Sample Rate Optimization
If CPU usage is high, consider using 16kHz throughout:
```yaml
audio:
  sample_rate: 16000  # Capture at 16kHz to avoid resampling
```

Note: This requires your audio device to support 16kHz capture.

## Changes in This Update

1. **Reduced default gains** to prevent distortion
2. **Soft limiting** instead of hard clipping for better audio quality
3. **Consistent normalization** values across the pipeline
4. **Lower OpenAI normalization** target to prevent over-amplification
5. **Better clipping detection** and logging

These changes should resolve issues with wake word detection and speech recognition accuracy.