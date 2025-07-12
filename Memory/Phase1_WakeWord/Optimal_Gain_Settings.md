# Optimal Audio Gain Settings

## Final Configuration
- **Microphone**: TONOR G11 USB 
- **Audio Gain**: 3.0
- **Result**: Optimal balance - no clipping, good signal levels

## Test Results with Gain 3.0
- **Audio Levels**: 3,000-22,000 (excellent range)
- **Peak Levels**: ~17,000-22,000 (strong signal without clipping)
- **Clipping**: 0% (no frames clipped)
- **Headroom**: ~10,000 below maximum (good safety margin)

## Why 3.0 is Better Than 5.0
1. **No Clipping**: Gain 5.0 caused frequent clipping at 32767
2. **Clean Audio**: No distortion from clipped peaks
3. **Better Detection**: Porcupine works better with clean, undistorted audio
4. **Safety Margin**: Leaves room for louder speech without distortion

## Configuration
```yaml
wake_word:
  engine: "porcupine"
  model: "picovoice"
  sensitivity: 1.0
  audio_gain: 3.0  # Optimal for TONOR G11
```

## Summary
The TONOR G11 microphone with 3.0x gain provides:
- Strong, clear audio signal
- No clipping or distortion
- Optimal levels for wake word detection
- Good balance between sensitivity and quality