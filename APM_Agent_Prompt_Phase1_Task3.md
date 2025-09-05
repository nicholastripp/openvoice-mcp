# Implementation Agent Prompt: Phase 1, Task 1.3

## Role Assignment
You are the **Testing Implementation Agent** for the HA Realtime Voice Assistant project, specializing in systematic testing, calibration routines, and device profiling. You'll work with the diagnostic tools and resampling improvements from Tasks 1.1 and 1.2 to optimize gain staging across the audio pipeline.

## Project Context Update
Tasks 1.1 and 1.2 have provided critical improvements:
- **Task 1.1:** Diagnostic tool revealed cumulative gain up to 25x possible across pipeline
- **Task 1.2:** Resampling improved with scipy.signal.resample_poly (50% CPU reduction)
- **Current Issue:** Wake word detection at ~85% due to gain-induced clipping
- **Root Cause:** Multiple uncoordinated gain stages causing distortion

## Your Mission: Task 1.3 - Gain Stage Optimization

### Objective
Create a comprehensive gain optimization system that tests various microphone types and speech volumes to establish optimal gain settings that prevent clipping while maintaining adequate signal levels for wake word detection and transcription.

### Specifications

#### Core Requirements

1. **Test Matrix Development:**
   ```
   Microphones: 3 types
   - USB microphones (e.g., Blue Yeti, generic USB)
   - ReSpeaker 2-Mic HAT
   - Conference speakerphones (e.g., Jabra Speak 410)
   
   Gain Levels: 5 settings per stage
   - input_volume: [0.3, 0.5, 1.0, 2.0, 3.0]
   - agc_enabled: [true, false]
   - wake_word.audio_gain: [0.5, 1.0, 1.5, 2.0, 3.0]
   
   Volume Levels: 3 scenarios
   - Quiet speech: 55-60 dB SPL
   - Normal conversation: 60-65 dB SPL  
   - Loud speech: 70-75 dB SPL
   ```

2. **Calibration Wizard:**
   - Interactive CLI tool for users
   - Automatic microphone detection
   - Guided audio level testing
   - Real-time visual feedback
   - Optimal settings calculation
   - Profile generation and saving

3. **Testing Protocol:**
   - Capture 10-second samples at each configuration
   - Measure clipping occurrence
   - Test wake word detection accuracy
   - Validate OpenAI transcription quality
   - Calculate cumulative gain through pipeline
   - Identify sweet spot for each device

4. **Profile System:**
   - Device-specific YAML profiles
   - Automatic profile selection based on device ID
   - Override capability for custom setups
   - Fallback to safe defaults

### Implementation Guidelines

#### File Structure
Main tool: `tools/gain_optimization_wizard.py`

Supporting files:
```
tools/gain_calibration/
├── test_matrix.py          # Test configuration generator
├── device_profiler.py      # Device identification and profiling
├── calibration_routine.py  # Interactive calibration logic
├── wake_word_tester.py     # Wake word accuracy testing
└── profile_manager.py      # Profile CRUD operations

config/audio_profiles/
├── usb_generic.yaml        # Generic USB mic profile
├── respeaker_2mic.yaml     # ReSpeaker HAT profile
├── jabra_410.yaml          # Jabra conference speaker
└── custom_template.yaml    # User customization template
```

#### Code Architecture
```python
class GainOptimizationWizard:
    def __init__(self, diagnostic_tool=None):
        self.diagnostic = diagnostic_tool or AudioPipelineDiagnostic()
        self.test_matrix = TestMatrix()
        self.profiler = DeviceProfiler()
        
    def run_interactive_calibration(self) -> dict:
        # Guide user through calibration process
        
    def test_configuration(self, config: dict, test_audio: np.ndarray) -> dict:
        # Test specific gain configuration
        
    def find_optimal_gains(self, device_type: str, room_noise: float) -> dict:
        # Calculate optimal settings for device and environment
        
    def generate_profile(self, device_info: dict, optimal_gains: dict) -> str:
        # Create YAML profile for device
        
    def validate_with_wake_word(self, config: dict) -> float:
        # Test wake word detection accuracy
```

#### Integration Points
```python
# Use existing diagnostic tool
from tools.audio_pipeline_diagnostic import AudioPipelineDiagnostic
from tools.audio_analysis.metrics import calculate_clipping_ratio

# Use improved resampling
from tools.resampling_tests.quality_metrics import measure_snr

# Hook into live pipeline
from src.audio.capture import AudioCapture
from src.wake_word.porcupine_detector import PorcupineDetector
```

#### Calibration Flow
1. **Device Detection:**
   - Enumerate audio devices
   - Identify device type/model
   - Load existing profile if available

2. **Noise Floor Measurement:**
   - Capture ambient room noise
   - Calculate baseline levels
   - Adjust thresholds accordingly

3. **Level Testing:**
   - User speaks at different volumes
   - Real-time meters show levels
   - Identify clipping points

4. **Optimization:**
   - Test matrix combinations
   - Find maximum clean gain
   - Validate with wake word

5. **Profile Generation:**
   - Save optimal settings
   - Create backup of current config
   - Apply new settings

### Acceptance Criteria
- [ ] Test matrix covers all 3 microphone types
- [ ] Each configuration tested with 3 volume levels
- [ ] Calibration wizard completes in <30 seconds
- [ ] Wake word accuracy improves from 85% to >95%
- [ ] Zero clipping at normal speech levels (60-65 dB)
- [ ] Profiles work across different Raspberry Pi models
- [ ] Visual feedback shows real-time audio levels
- [ ] Generated profiles are human-readable YAML
- [ ] Rollback mechanism for failed calibrations

### Resources Available

#### From Previous Tasks
- `tools/audio_pipeline_diagnostic.py` - Stage analysis
- `tools/audio_resampling_analysis.py` - Quality metrics
- Task findings: 25x cumulative gain issue identified

#### Test Equipment Simulation
If physical devices unavailable:
```python
# Simulate device characteristics
DEVICE_PROFILES = {
    'usb_generic': {'sensitivity': 0.8, 'noise_floor': -50},
    'respeaker': {'sensitivity': 1.2, 'noise_floor': -45},
    'jabra_410': {'sensitivity': 1.0, 'noise_floor': -55}
}
```

#### Existing Configuration
```python
# Current problematic settings from config.yaml
audio:
  input_volume: 1.0      # May be too high
  agc_enabled: true      # May fight with manual gain
wake_word:
  audio_gain: 1.0        # Additional amplification
  sensitivity: 1.0       # Detection threshold
```

### Success Metrics
- Wake word detection accuracy: >95% (from 85%)
- Zero clipping incidents at normal speech
- Transcription error rate: <2%
- Calibration time: <30 seconds
- Profile accuracy: Works for 90% of users without adjustment

### Deliverables
1. **Primary Tool:** `tools/gain_optimization_wizard.py`
2. **Test Suite:** `tools/gain_calibration/` module
3. **Device Profiles:** At least 3 profiles in `config/audio_profiles/`
4. **Report:** `reports/gain_optimization_results.json`
5. **Documentation:** Update `docs/AUDIO_SETUP.md` with calibration guide
6. **Integration:** PR with config.py changes for profile loading

### Timeline
**Duration:** 3 days
- Day 1: Test matrix implementation and data collection
- Day 2: Calibration wizard and profile system
- Day 3: Wake word validation and documentation

### Critical Considerations

#### The Gain Multiplication Problem
Current pipeline has multiplicative gain:
```
Total Gain = input_volume × agc_gain × wake_word_gain
Example: 2.0 × 3.0 × 3.0 = 18x amplification!
```

#### Device Sensitivity Variations
- USB mics: Wide sensitivity range (-40 to -20 dBFS)
- ReSpeaker: Higher sensitivity, prone to clipping
- Conference speakers: AGC built-in, may conflict

#### User Experience Goals
- One-time calibration that "just works"
- No technical knowledge required
- Visual/audio feedback during calibration
- Clear improvement immediately noticeable

### Getting Started
1. Review gain-related findings from Tasks 1.1 and 1.2
2. Set up test environment with multiple audio sources
3. Create test matrix generator
4. Implement device detection and profiling
5. Build interactive calibration routine
6. Validate with real wake word testing
7. Generate device-specific profiles

### Questions to Address
- Should AGC be disabled for some devices with built-in AGC?
- How to handle USB devices without consistent IDs?
- Can we auto-detect room acoustics and adjust?
- Should profiles include time-of-day adjustments?

## Your Response Should Include
1. Acknowledgment of Tasks 1.1 and 1.2 findings
2. Your testing strategy for the gain matrix
3. Approach to user-friendly calibration
4. Method for validating wake word improvements
5. Any additional device types to consider

Remember: This calibration system will be the user's first experience with fixing their audio issues. It must be simple, fast, and effective. The goal is to achieve >95% wake word accuracy while preventing any clipping distortion.

---

*Reference the updated APM_Memory_Bank.md with Tasks 1.1 and 1.2 results, and APM_Implementation_Plan.md for project context.*