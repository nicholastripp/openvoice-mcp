# TONOR G11 USB Microphone Test Results

## Microphone Identification
- **Device**: TONOR G11 USB microphone
- **Card**: 3
- **ALSA Device**: plughw:3,0
- **Manufacturer**: C-Media Electronics Inc.

## Audio Level Testing

### Raw Microphone Levels (no gain)
- **Peak level at 48kHz**: 2075
- **Average max**: 518
- **Assessment**: Low gain, needs amplification

### With Audio Gain Applied
- **Configuration**: audio_gain: 5.0 (maximum allowed)
- **Resulting levels**: 
  - Average: 10,000-20,000
  - Peaks: 32,765 (maximum before clipping)
  - Some clipping detected at peaks

## Current Status
1. TONOR microphone successfully detected and configured
2. Audio levels with 5.0x gain are adequate for wake word detection
3. Audio is reaching Porcupine at appropriate levels
4. System is ready for wake word detection testing

## Configuration Used
```yaml
wake_word:
  engine: "porcupine"
  model: "picovoice"
  sensitivity: 1.0
  audio_gain: 5.0
```

## Next Steps
1. Test wake word detection by saying "picovoice" clearly
2. Monitor test.log for detection events
3. If no detections, try other wake words like "alexa" or "computer"