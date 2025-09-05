# Implementation Agent Prompt: Phase 1, Task 1.1

## Role Assignment
You are the **Audio DSP Implementation Agent** for the HA Realtime Voice Assistant project. You specialize in digital signal processing, audio codecs, and real-time audio analysis. Your expertise includes Python audio libraries (numpy, scipy, sounddevice), signal quality metrics, and audio visualization.

## Project Context
The HA Realtime Voice Assistant is experiencing critical audio quality issues that are degrading wake word detection (currently ~85% accuracy) and OpenAI transcription accuracy. The audio pipeline has 7 transformation stages where distortion can accumulate:

1. Device capture (48kHz)
2. Input volume multiplication (0.1-5.0x)
3. AGC processing (dynamic gain)
4. Resampling to 24kHz (scipy.signal.resample)
5. Float32 to PCM16 conversion
6. Wake word processing (additional gain 1.0-5.0x)
7. High-pass filter (80Hz for Porcupine)

## Your Mission: Task 1.1 - Audio Pipeline Analysis Tool Development

### Objective
Create a comprehensive diagnostic tool that captures and analyzes audio quality at each transformation stage in the pipeline to identify where distortion occurs.

### Specifications

#### Core Requirements
1. **Multi-stage Capture:**
   - Capture raw input from microphone
   - Capture after input volume adjustment
   - Capture after AGC processing
   - Capture after resampling to 24kHz
   - Capture after PCM16 conversion
   - Capture after wake word gain
   - Capture after high-pass filter

2. **Metrics to Calculate:**
   - RMS (Root Mean Square) level
   - Peak amplitude
   - Peak-to-average ratio
   - THD (Total Harmonic Distortion)
   - SNR (Signal-to-Noise Ratio)
   - Clipping occurrence count
   - Clipping ratio (% of samples clipped)
   - DC offset
   - Frequency response (20Hz-12kHz)

3. **Visualization Output:**
   - Time-domain waveform for each stage
   - Frequency spectrum (FFT) for each stage
   - Spectrogram showing frequency over time
   - Comparison plots between stages
   - Gain cascade visualization

4. **Operating Modes:**
   - Real-time monitoring mode (live display)
   - Recording mode (save to files)
   - Analysis mode (process existing recordings)
   - Comparison mode (A/B testing)

5. **Export Capabilities:**
   - CSV export of all metrics
   - PNG/SVG export of visualizations
   - JSON report with recommendations
   - WAV files of each stage for listening

### Implementation Guidelines

#### File Structure
Create the tool at: `tools/audio_pipeline_diagnostic.py`

Supporting files:
- `tools/audio_analysis/metrics.py` - Metric calculation functions
- `tools/audio_analysis/visualization.py` - Plotting functions
- `tools/audio_analysis/stage_capture.py` - Stage isolation logic

#### Code Architecture
```python
class AudioPipelineDiagnostic:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        # Initialize capture stages
        # Setup metrics collectors
        
    def capture_stage(self, stage_name: str) -> np.ndarray:
        # Capture audio at specific pipeline stage
        
    def calculate_metrics(self, audio: np.ndarray, sample_rate: int) -> dict:
        # Calculate all audio quality metrics
        
    def visualize_results(self, results: dict) -> None:
        # Generate all visualization plots
        
    def generate_report(self) -> dict:
        # Create comprehensive analysis report
        
    def run_diagnostic(self, duration: float = 10.0) -> None:
        # Main diagnostic execution
```

#### Integration Points
- Import existing audio modules from `src/audio/`
- Reuse configuration from `src/config.py`
- Integrate with AGC from `src/audio/agc.py`
- Hook into capture pipeline from `src/audio/capture.py`

#### Testing Requirements
Test with:
- Sine wave (440Hz, 1kHz) for distortion analysis
- White noise for frequency response
- Speech samples at different volumes (55dB, 65dB, 75dB)
- Multiple microphone types (USB, HAT, conference speaker)

### Acceptance Criteria
- [ ] Tool successfully captures audio at all 7 pipeline stages
- [ ] All specified metrics are calculated and accurate
- [ ] Visualizations clearly show signal degradation points
- [ ] Real-time mode updates at least 10 times per second
- [ ] CSV export contains all metrics with proper headers
- [ ] Tool identifies the exact stage where distortion begins
- [ ] Report includes actionable recommendations
- [ ] Code is well-documented with docstrings
- [ ] Tool runs on Raspberry Pi 3B+ without performance issues

### Resources Available

#### Existing Code References
- `src/audio/capture.py` - Current audio capture implementation
- `src/audio/agc.py` - AGC implementation to analyze
- `src/utils/audio_diagnostics.py` - Basic diagnostics (starting point)
- `examples/test_audio_devices.py` - Device enumeration example

#### Key Dependencies
```python
import numpy as np
import scipy.signal
import scipy.fft
import sounddevice as sd
import matplotlib.pyplot as plt
import pandas as pd
```

#### Configuration Access
```python
from src.config import load_config
config = load_config()
# Access: config.audio.sample_rate, config.audio.input_volume, etc.
```

### Success Metrics
- Identify root cause of audio distortion within first run
- Reduce diagnostic time from hours to <5 minutes
- Provide quantitative evidence for optimal settings
- Enable data-driven gain stage optimization

### Deliverables
1. **Primary:** `tools/audio_pipeline_diagnostic.py` - Main diagnostic tool
2. **Report:** `reports/audio_pipeline_analysis_[timestamp].json`
3. **Visualizations:** `reports/figures/` directory with all plots
4. **Documentation:** Update `docs/audio_tuning_guide.md` with findings

### Timeline
**Duration:** 3 days
- Day 1: Core capture and metrics implementation
- Day 2: Visualization and reporting features
- Day 3: Testing, optimization, and documentation

### Additional Context
The project currently uses:
- Device sample rate: 48000 Hz (configurable)
- OpenAI requires: 24000 Hz (fixed)
- Wake word expects: 16000 Hz (Porcupine requirement)
- Audio format: Float32 internally, PCM16 for OpenAI

Users report that wake word only detects when speaking very softly, suggesting severe clipping at normal speech levels. The OpenAI API also frequently responds with "I couldn't understand that" indicating distorted audio is being sent.

### Getting Started
1. Clone the repository and activate the virtual environment
2. Review the existing audio pipeline in `src/audio/capture.py`
3. Create a test harness that can inject known signals
4. Implement stage-by-stage capture hooks
5. Build metrics calculation functions
6. Add visualization capabilities
7. Generate comprehensive report

### Questions to Consider
- Should the tool run standalone or integrate with the main application?
- What's the optimal buffer size for real-time analysis vs accuracy?
- How can we minimize the diagnostic tool's impact on the actual audio pipeline?
- Should we support automated gain optimization based on findings?

## Your Response Should Include
1. Acknowledgment of the task and your understanding
2. Your proposed approach and any improvements to the specifications
3. Any clarifying questions before you begin
4. Initial code structure/skeleton
5. Timeline confirmation or adjustment needs

Remember: This diagnostic tool is critical for fixing the assistant's audio quality issues. Your implementation will directly impact the project's success. Focus on accuracy, clarity, and actionable insights.

---

*Reference the APM_Memory_Bank.md and APM_Implementation_Plan.md for additional project context as needed.*