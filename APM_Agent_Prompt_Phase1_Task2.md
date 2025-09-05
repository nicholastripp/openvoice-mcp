# Implementation Agent Prompt: Phase 1, Task 1.2

## Role Assignment
You are the **Audio DSP Implementation Agent** continuing work on the HA Realtime Voice Assistant project. Having successfully completed the audio pipeline diagnostic tool (Task 1.1), you now focus on analyzing and improving the resampling quality in the audio pipeline.

## Project Context Update
Task 1.1 revealed critical findings:
- Multiple gain stages can compound to 25x amplification
- PCM16 conversion using 32767 multiplier may cause asymmetry  
- Current resampling method (scipy.signal.resample) lacks anti-aliasing configuration
- THD measurements show significant distortion potential
- Wake word detection accuracy remains degraded at ~85%

## Your Mission: Task 1.2 - Resampling Quality Assessment

### Objective
Compare the current scipy.signal.resample method with alternative resampling approaches to identify the optimal solution for voice audio quality. The pipeline requires resampling from 48kHz (device) to 24kHz (OpenAI) and potentially to 16kHz (Porcupine).

### Specifications

#### Core Requirements

1. **Resampling Methods to Test:**
   - Current: `scipy.signal.resample` (FFT-based)
   - Alternative 1: `scipy.signal.resample_poly` (polyphase filtering)
   - Alternative 2: `librosa.resample` (Kaiser window)
   - Alternative 3: `soxr` Python bindings (SoX resampler)
   - Alternative 4: `resampy.resample` (band-limited sinc)

2. **Test Scenarios:**
   - **Chunk Sizes:** 1200, 2400, 4800 samples (50ms, 100ms, 200ms at 24kHz)
   - **Sample Rates:** 48000→24000, 48000→16000, 44100→24000
   - **Signal Types:**
     - Pure tones (440Hz, 1kHz, 3kHz, 8kHz)
     - Speech samples (male, female, child voices)
     - White/pink noise for frequency response
     - Multi-tone test signals

3. **Quality Metrics to Measure:**
   - **Frequency Response:** 20Hz-12kHz (voice range)
   - **Aliasing Rejection:** Measure energy above Nyquist
   - **Phase Response:** Group delay consistency
   - **THD+N:** Total Harmonic Distortion plus Noise
   - **Processing Latency:** Time per chunk
   - **CPU Usage:** Especially on Raspberry Pi

4. **Analysis Outputs:**
   - Frequency response plots for each method
   - Aliasing artifact measurements
   - Quality vs. performance tradeoff matrix
   - Recommended settings per use case

### Implementation Guidelines

#### File Structure
Create test suite at: `tools/audio_resampling_analysis.py`

Supporting files:
- `tools/resampling_tests/quality_metrics.py` - Quality measurements
- `tools/resampling_tests/performance_bench.py` - Performance testing
- `tools/resampling_tests/comparison_report.py` - Comparative analysis

#### Code Architecture
```python
class ResamplingAnalyzer:
    def __init__(self):
        self.methods = {
            'scipy_fft': self.resample_scipy_fft,
            'scipy_poly': self.resample_scipy_poly,
            'librosa': self.resample_librosa,
            'soxr': self.resample_soxr,
            'resampy': self.resample_resampy
        }
        
    def benchmark_quality(self, method: str, signal: np.ndarray, 
                         orig_sr: int, target_sr: int) -> dict:
        # Measure quality metrics for resampling method
        
    def benchmark_performance(self, method: str, chunk_size: int,
                            iterations: int = 1000) -> dict:
        # Measure CPU and latency performance
        
    def analyze_frequency_response(self, method: str) -> dict:
        # Analyze frequency domain characteristics
        
    def generate_comparison_report(self) -> pd.DataFrame:
        # Create comprehensive comparison matrix
```

#### Integration with Existing Tool
Leverage the diagnostic tool from Task 1.1:
```python
from tools.audio_pipeline_diagnostic import AudioPipelineDiagnostic
from tools.audio_analysis.metrics import calculate_thd, calculate_snr

# Use existing metrics infrastructure
diagnostic = AudioPipelineDiagnostic()
# Hook into stage 4 (resampling stage)
```

#### Testing Protocol
1. **Baseline Measurement:**
   - Capture current scipy.resample behavior
   - Document existing artifacts

2. **Controlled Testing:**
   - Use identical input signals for all methods
   - Test at multiple amplitude levels
   - Include edge cases (silence, clipping)

3. **Real-world Testing:**
   - Process actual speech recordings
   - Test with wake word audio samples
   - Validate with OpenAI API responses

### Acceptance Criteria
- [ ] All 5 resampling methods implemented and tested
- [ ] Frequency response measured for voice range (20Hz-12kHz)
- [ ] Aliasing measured and below -60dB for best method
- [ ] Performance benchmarked on Raspberry Pi 3B+ and 4
- [ ] Optimal chunk size identified for latency/quality balance
- [ ] Clear recommendation with quantitative justification
- [ ] Configuration code for selected method ready to integrate
- [ ] Documentation includes before/after spectrograms

### Resources Available

#### Existing Code References
- `tools/audio_pipeline_diagnostic.py` - Completed diagnostic tool
- `src/audio/capture.py:319` - Current resampling implementation
- `tools/audio_analysis/metrics.py` - Quality metric calculations

#### Additional Dependencies to Test
```bash
pip install librosa soxr-python resampy
# Note: Some may require additional system libraries
```

#### Test Audio Files
Place in `test_audio/` directory:
- `speech_male_48k.wav` - Male speech sample
- `speech_female_48k.wav` - Female speech sample  
- `tone_sweep_48k.wav` - Frequency sweep 20Hz-20kHz
- `wake_word_samples_48k.wav` - "Picovoice" utterances

### Success Metrics
- Identify resampling method with <1% THD for voice
- Achieve >60dB aliasing rejection
- Maintain <5ms processing latency per chunk
- Reduce CPU usage by >20% if possible
- Improve wake word detection accuracy as result

### Deliverables
1. **Primary:** `tools/audio_resampling_analysis.py` - Analysis tool
2. **Report:** `reports/resampling_comparison.json` - Full metrics
3. **Visualizations:** `reports/resampling/` - Comparison plots
4. **Configuration:** `config/optimal_resampling.yaml` - Recommended settings
5. **Integration PR:** Code changes for `src/audio/capture.py`

### Timeline
**Duration:** 2 days
- Day 1: Implement all resampling methods and quality tests
- Day 2: Performance benchmarking and report generation

### Critical Considerations

#### Current Problem Evidence
From Task 1.1 findings:
```python
# Current implementation in src/audio/capture.py:319
resampled = signal.resample(audio_data, new_length)
# No anti-aliasing filter configuration
# No window function specified
# May introduce artifacts at high frequencies
```

#### Raspberry Pi Constraints
- Limited CPU for real-time processing
- Memory bandwidth considerations
- Must maintain low latency for conversation flow
- Power consumption on battery-powered setups

### Getting Started
1. Review Task 1.1 diagnostic tool outputs
2. Set up test environment with all resampling libraries
3. Create controlled test signals
4. Implement method wrappers with consistent interface
5. Run quality and performance benchmarks
6. Generate comparison visualizations
7. Make recommendation based on data

### Questions to Address
- What's the optimal tradeoff between quality and CPU usage?
- Should we use different methods for different stages (wake word vs. OpenAI)?
- Can we pre-calculate filter coefficients for efficiency?
- Is the quality improvement worth additional dependencies?

## Your Response Should Include
1. Acknowledgment of Task 1.1 completion and findings
2. Your approach to resampling comparison testing
3. Any additional methods or metrics to consider
4. Preliminary hypothesis based on theory
5. Risk assessment for production integration

Remember: The resampling stage is critical for maintaining voice quality while meeting the different sample rate requirements of OpenAI (24kHz) and Porcupine (16kHz). Your analysis will directly impact audio quality throughout the pipeline.

---

*Reference the APM_Memory_Bank.md (updated with Task 1.1 results) and APM_Implementation_Plan.md for additional context.*