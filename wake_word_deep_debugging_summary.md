# Wake Word Deep Debugging Summary

## Problem Statement
Both hey_jarvis and alexa wake word models return extremely low confidence values (~1e-06), far below the threshold of 0.0001, even with loud speech input (RMS 0.108). This indicates a fundamental issue with audio processing or model initialization, not model corruption.

## Debugging Enhancements Applied

### 1. Audio Resampling Verification
- Added detailed logging for 48kHz → 16kHz resampling process
- Logs show: original samples, target samples, pre/post resample levels
- Verifies if resampling is destroying the audio signal

### 2. Model File Integrity Check  
- Enhanced model availability check to verify file size
- TFLite models should be >10KB - will error if too small
- Logs exact path and size of model files

### 3. Sensitivity Threshold Adjustment
- Lowered threshold from 0.0001 to 0.000001 (1e-06)
- This matches the confidence values being returned
- Temporary change to test if model responds at all

### 4. Audio Format Validation
- Added comprehensive format checks before model prediction:
  - Chunk size must be exactly 1280 samples
  - Data type must be float32
  - Audio range must be [-1.0, 1.0]
- Logs format details every 100 chunks

### 5. Queue Chunk Verification
- Added critical check when pulling chunks from queue
- Ensures chunks are exactly 1280 samples before processing
- Will error and skip invalid chunks

### 6. Raw OpenWakeWord Test Script
- Created `test_openwakeword_raw.py` to test models directly
- Bypasses our entire audio pipeline
- Tests various audio patterns and checks for stuck states

## Expected Debug Output

When running with these changes, look for:

1. **Resampling logs**:
   ```
   DETECTOR: RESAMPLING 48000Hz → 16000Hz, samples: 200 → 67
   Post-resample: level=0.xxxxx, RMS=0.xxxxx
   ```

2. **Model file check**:
   ```
   Model file found: /path/to/alexa_v0.1.tflite (size: XXXX bytes)
   ```

3. **Format validation**:
   ```
   DETECTOR: AUDIO FORMAT: (1280,) float32 [-0.xxxxx, 0.xxxxx] RMS=0.xxxxx
   ```

4. **Chunk size errors** (if any):
   ```
   DETECTOR: CRITICAL ERROR - Wrong chunk size from queue: XXX
   ```

## Next Steps

1. **Run the test script**: 
   ```bash
   python test_openwakeword_raw.py
   ```
   This will show if OpenWakeWord works at all on the system.

2. **Check the logs** for:
   - Resampling destroying audio (RMS dropping to near 0)
   - Wrong chunk sizes in queue
   - Model file size issues

3. **If test script shows same issue**:
   - OpenWakeWord installation may be broken
   - Try reinstalling: `pip install --force-reinstall openwakeword`
   - Check TensorFlow Lite compatibility

4. **If test script works**:
   - Issue is in our audio pipeline
   - Focus on resampling and chunk size calculations
   - Check if 48kHz audio is being properly converted

## Root Cause Hypothesis

The cycling pattern of values (9.904526e-07 → 1.0394664e-06 → 1.1165829e-06) suggests:
1. Model is initialized but not processing audio correctly
2. These may be default/initialization values
3. Audio format mismatch preventing proper inference

The fact that both models show identical behavior confirms this is a systemic issue, not model-specific corruption.