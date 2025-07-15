# Wake Word Sensitivity Fix

## Issue
Wake words were not being detected despite audio flowing correctly through the system.

## Root Cause
The sensitivity in config.yaml was set to `0.000001` (1e-06), which is extremely low. This made the detector require an impossibly strong signal to trigger.

## Solution
Changed sensitivity from `0.000001` to `0.5` in config.yaml:

```yaml
wake_word:
  sensitivity: 0.5  # Was 0.000001
```

## Technical Details
- Porcupine expects sensitivity values between 0.0 and 1.0
- Lower values = more sensitive (opposite of what the comment suggested)
- The value 0.000001 effectively disabled detection
- Value of 0.5 provides balanced detection

## Verification
After fix:
- Audio processing confirmed working (48kHz → 16kHz resampling)
- Porcupine processing frames correctly
- Detection ready for wake word "jarvis" (from hey_jarvis model)

## Recommendation
For production use:
- 0.3-0.5 for normal environments
- 0.1-0.3 for noisy environments (more false positives)
- 0.5-0.7 for quiet environments (fewer false positives)