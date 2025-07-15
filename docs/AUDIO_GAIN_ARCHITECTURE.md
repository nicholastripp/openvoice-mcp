# Audio Gain Architecture

## Overview

The audio processing pipeline has multiple gain stages that can compound and cause distortion if not configured properly.

## Automatic Gain Control (AGC)

The system now includes Automatic Gain Control (AGC) that can automatically adjust input volume to:
- Prevent clipping distortion
- Maintain optimal audio levels
- Adapt to different microphone types and environments

### AGC Configuration
```yaml
audio:
  agc_enabled: true        # Enable automatic gain adjustment
  agc_target_rms: 0.3     # Target level (30% of max)
  agc_max_gain: 3.0       # Maximum amplification allowed
  agc_min_gain: 0.1       # Minimum gain (can attenuate to 10%)
```

### When to Use AGC
- **Enable AGC** if:
  - You're unsure about the correct input_volume setting
  - Your environment has varying noise levels
  - Multiple people with different voice levels use the system
  - You want "set and forget" operation

- **Disable AGC** if:
  - You have a consistent environment
  - You've already found the perfect input_volume
  - You need predictable gain levels

## Gain Stages

### 1. Input Capture Gain (`audio.input_volume`)
- **Location**: Applied at audio capture in `src/audio/capture.py`
- **Default**: 1.0 (no change)
- **Range**: 0.1 - 5.0
  - **< 1.0**: Attenuates (reduces) volume - use for loud microphones
  - **= 1.0**: No change to input level
  - **> 1.0**: Amplifies volume - use for quiet microphones
- **Safe Range**: 0.5 - 2.0
- **Warning**: Values > 2.0 often cause clipping/distortion

### 2. Wake Word Gain (`wake_word.audio_gain`)
- **Location**: Applied in wake word detectors
- **Default**: 1.0 (no gain)
- **Purpose**: Boost quiet audio for wake word detection
- **Note**: Usually not needed if input_volume is set correctly

### 3. OpenAI Audio Normalization
- **Location**: `src/openai_client/realtime.py`
- **Target RMS**: 0.05 (normalized range)
- **Max Gain**: 3.0x
- **Purpose**: Normalize audio for consistent VAD detection

## Total Amplification

The total gain applied is multiplicative:
```
Total Gain = input_volume × wake_word.audio_gain × openai_gain
```

Example with defaults:
- input_volume: 1.0
- wake_word.audio_gain: 1.0
- openai_gain: up to 3.0
- **Total**: up to 3.0x

Example with high input_volume (problematic):
- input_volume: 5.0
- wake_word.audio_gain: 1.0
- openai_gain: up to 3.0
- **Total**: up to 15.0x (causes severe distortion!)

## Troubleshooting

### Symptoms of Too Much Gain:
1. Wake word only triggers with soft speech
2. Normal/loud speech doesn't trigger wake word
3. OpenAI misunderstands commands
4. Audio logs show frequent max values (32767)
5. Clipping warnings in logs (e.g., "50% of samples clipped")

### Symptoms of Microphone Too Loud (Even at 1.0 Gain):
1. Clipping occurs even with input_volume: 1.0
2. Wake word works better with soft speech
3. Need to use input_volume < 1.0 to attenuate

### Recommended Settings:
```yaml
audio:
  input_volume: 1.0  # Start here, increase only if needed

wake_word:
  audio_gain: 1.0    # Keep at 1.0 unless input is very quiet
```

### If Audio is Too Quiet:
1. First, check your microphone placement and settings
2. Try input_volume: 1.5 or 2.0
3. Only increase wake_word.audio_gain if input_volume alone isn't enough
4. Never set input_volume > 3.0

### If Audio is Too Loud (Clipping at 1.0):
1. Your microphone gain may be set too high in the OS
2. Try input_volume: 0.5 or 0.7 to attenuate
3. Monitor logs for clipping warnings
4. Adjust until clipping is minimal (<5%)

## Important Notes

- The high-pass filter (80Hz) is REQUIRED for Porcupine wake word detection
- Multiple gain stages compound - be conservative with each stage
- Soft clipping is applied to prevent harsh distortion, but it's better to avoid clipping altogether
- Monitor audio levels in logs to ensure you're not hitting max values frequently