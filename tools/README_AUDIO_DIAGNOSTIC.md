# Audio Pipeline Diagnostic Tool

## Overview

The Audio Pipeline Diagnostic Tool is a comprehensive analysis system designed to identify and diagnose audio quality issues in the HA Realtime Voice Assistant. It captures and analyzes audio at each of the 7 transformation stages in the pipeline to pinpoint exactly where distortion occurs.

## Key Features

- **Multi-Stage Capture**: Captures audio at all 7 pipeline transformation stages
- **Comprehensive Metrics**: Calculates RMS, Peak, THD, SNR, clipping ratio, DC offset, and frequency response
- **Real-time Monitoring**: Live display of audio metrics during capture
- **Visualization Suite**: Generates waveforms, spectrograms, frequency response plots, and gain cascade charts
- **Automated Reports**: Produces detailed JSON and CSV reports with recommendations
- **Device Profiling**: Creates optimized configuration profiles for different audio hardware

## Pipeline Stages Analyzed

1. **raw_input**: Raw microphone input at device sample rate (48kHz)
2. **volume_adjusted**: After input volume multiplication (0.1-5.0x)
3. **agc_processed**: After Automatic Gain Control processing
4. **resampled_24k**: After resampling to 24kHz for OpenAI
5. **pcm16_converted**: After Float32 to PCM16 conversion
6. **wake_word_gain**: After wake word gain application (1.0-5.0x)
7. **highpass_filtered**: After high-pass filter (80Hz cutoff)

## Installation

The tool uses the existing project dependencies. No additional installation required.

## Usage

### Basic Commands

```bash
# Run real-time diagnostic (recommended for first analysis)
python tools/audio_pipeline_diagnostic.py --mode realtime --duration 10

# Record audio for later analysis
python tools/audio_pipeline_diagnostic.py --mode record --duration 30

# Analyze previously recorded audio
python tools/audio_pipeline_diagnostic.py --mode analyze --input reports/recording_*.npz

# Compare two configurations
python tools/audio_pipeline_diagnostic.py --mode compare --config1 config1.yaml --config2 config2.yaml
```

### Command Line Options

- `--mode`: Diagnostic mode (realtime, record, analyze, compare)
- `--duration`: Capture duration in seconds (default: 10)
- `--config`: Path to configuration file (default: config/config.yaml)
- `--input`: Input file for analyze mode
- `--output-dir`: Output directory for reports (default: reports/)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Understanding the Output

### Real-time Display

During real-time analysis, the tool displays:
```
Stage: volume_adjusted
  RMS: 0.227334 | Peak: 0.425144
  THD: 37.25% | SNR: 18.9 dB
  Clipping: 0.00% (0 samples)
  DC Offset: 0.000099
  Level: |#####################.............................|
```

### Key Metrics Explained

- **RMS (Root Mean Square)**: Average signal level (0.0-1.0)
  - Optimal: 0.2-0.4 for normal speech
  - Too low (<0.1): Weak signal, poor wake word detection
  - Too high (>0.7): Risk of clipping

- **Peak**: Maximum amplitude reached
  - Optimal: 0.5-0.8
  - >0.99: Clipping occurring

- **THD (Total Harmonic Distortion)**: Percentage of harmonic distortion
  - <1%: Excellent
  - 1-3%: Good
  - 3-5%: Acceptable
  - >5%: Poor, audible distortion

- **SNR (Signal-to-Noise Ratio)**: Signal quality in dB
  - >40 dB: Excellent
  - 30-40 dB: Good
  - 20-30 dB: Acceptable
  - <20 dB: Poor

- **Clipping Ratio**: Percentage of samples at maximum amplitude
  - 0%: Perfect
  - <0.1%: Acceptable
  - >1%: Severe distortion

- **DC Offset**: Bias in the signal
  - Should be close to 0
  - >0.01: May indicate hardware issues

## Interpreting Results

### Common Issues and Solutions

#### Issue 1: High Clipping at volume_adjusted Stage
**Symptom**: Clipping >1% after volume adjustment
**Cause**: Input volume setting too high
**Solution**: Reduce `audio.input_volume` in config.yaml (try 0.5-1.0)

#### Issue 2: Distortion Increases at resampled_24k
**Symptom**: THD jumps significantly after resampling
**Cause**: Poor resampling algorithm or aliasing
**Solution**: The tool recommends using scipy.signal.resample_poly

#### Issue 3: Low Signal Levels Throughout
**Symptom**: RMS <0.05 at all stages
**Cause**: Microphone gain too low in ALSA
**Solution**: 
```bash
alsamixer  # Press F4, increase Mic level to 80-90%
```

#### Issue 4: Cumulative Gain Too High
**Symptom**: Signal grows >10x from input to output
**Cause**: Multiple gain stages compounding
**Solution**: Balance gains across stages (reduce both input_volume and wake_word.audio_gain)

## Generated Reports

### JSON Report Structure
```json
{
  "device_info": {...},
  "configuration": {...},
  "stages": [
    {
      "stage_name": "raw_input",
      "rms": 0.227,
      "peak": 0.425,
      "thd": 37.2,
      "snr": 19.0,
      "clipping_ratio": 0.0,
      ...
    }
  ],
  "recommendations": [...],
  "overall_assessment": "WARNING: 3 warnings found...",
  "timestamp": "2024-01-20T10:30:00"
}
```

### Visualization Outputs

1. **Waveform Plots** (`waveform_*.png`): Time-domain signal and RMS envelope
2. **Spectrograms** (`spectrogram_*.png`): Frequency content over time
3. **Frequency Response** (`freq_response_*.png`): Power distribution across frequencies
4. **Gain Cascade** (`gain_cascade_*.png`): Signal level changes through pipeline
5. **Metrics Comparison** (`metrics_comparison_*.png`): Multi-metric analysis dashboard

## Optimization Workflow

1. **Initial Diagnosis**: Run real-time mode to identify issues
   ```bash
   python tools/audio_pipeline_diagnostic.py --mode realtime
   ```

2. **Adjust Configuration**: Based on recommendations, modify config.yaml

3. **Test Changes**: Run diagnostic again to verify improvements

4. **Create Profile**: Save optimal settings for your hardware
   ```bash
   cp config/config.yaml config/profiles/my_device.yaml
   ```

5. **Document Findings**: Update Memory Bank with results

## Advanced Features

### Custom Test Signals

Modify `stage_capture.py` to inject specific test signals:
- Pure tones for THD analysis
- White/pink noise for frequency response
- Speech samples for real-world testing

### Parallel Stage Capture

For simultaneous capture of all stages (requires modification of actual pipeline):
```python
captured_data = await stage_capture.capture_parallel(
    stages=["raw_input", "volume_adjusted", "agc_processed"],
    duration=10.0
)
```

### Stage Difference Analysis

Compare two stages to isolate transformation effects:
```python
diff_metrics = stage_capture.analyze_stage_difference(
    stage1_audio=raw_audio,
    stage2_audio=processed_audio
)
```

## Troubleshooting

### ImportError for scipy or matplotlib
```bash
pip install scipy matplotlib
```

### No audio devices found
Ensure audio system is properly configured:
```bash
arecord -l  # List recording devices
python examples/test_audio_devices.py --list
```

### Permission denied accessing audio device
Add user to audio group:
```bash
sudo usermod -a -G audio $USER
# Log out and back in
```

## Integration with Main Application

To use diagnostic insights in production:

1. **Apply Recommended Settings**: Update config.yaml with optimal values
2. **Enable/Disable Features**: Based on findings (e.g., disable AGC if causing issues)
3. **Monitor Continuously**: Add diagnostic metrics to regular logging
4. **Create Alerts**: Set thresholds for critical metrics

## Contributing

When adding new features to the diagnostic tool:

1. Add new metrics to `AudioMetrics` class
2. Add visualizations to `AudioVisualizer` class
3. Update stage definitions in `PipelineStageCapture`
4. Document new features in this README

## Related Documentation

- [Audio Setup Guide](../docs/AUDIO_SETUP.md)
- [Configuration Reference](../docs/CONFIG.md)
- [APM Memory Bank](../APM_Memory_Bank.md)
- [Implementation Plan](../APM_Implementation_Plan.md)