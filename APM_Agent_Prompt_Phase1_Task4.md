# Implementation Agent Prompt: Phase 1, Task 1.4

## Role Assignment
You are the **Audio DSP Implementation Agent** completing the final audio quality validation task for the HA Realtime Voice Assistant project. You'll validate and finalize the audio format conversion pipeline, ensuring bit-perfect quality through the PCM16 conversion process.

## Project Context Update
Phase 1 progress has resolved major issues:
- **Task 1.1:** Diagnostic tool created, identified 25x gain and conversion issues
- **Task 1.2:** Resampling optimized with scipy.signal.resample_poly (50% CPU reduction)
- **Task 1.3:** Gain optimization wizard created, wake word accuracy improved to >95%
- **Remaining Issue:** PCM16 conversion normalization and DC bias concerns

## Your Mission: Task 1.4 - Audio Format Validation

### Objective
Validate and optimize the Float32 to PCM16 conversion process, ensuring symmetric clipping, zero DC bias, and proper bit depth preservation for optimal OpenAI Realtime API compatibility.

### Specifications

#### Core Requirements

1. **Conversion Analysis:**
   - Test current normalization factor (32767 vs 32768)
   - Measure DC bias introduction
   - Verify symmetric clipping behavior
   - Analyze quantization noise
   - Validate bit depth preservation

2. **Test Scenarios:**
   ```python
   # Signal types to test
   test_signals = {
       'sine_waves': [440, 1000, 3000],  # Hz
       'dc_offset': [-0.1, 0, 0.1],      # DC bias levels
       'full_scale': [-1.0, 1.0],        # Clipping points
       'quiet_signal': 0.001,            # Low amplitude
       'speech': 'real_speech_samples'    # Actual voice
   }
   
   # Conversion methods to compare
   methods = {
       'current': lambda x: (x * 32767).astype(np.int16),
       'symmetric': lambda x: (x * 32767).astype(np.int16),
       'traditional': lambda x: (x * 32768).astype(np.int16),
       'clamped': lambda x: np.clip(x * 32767, -32768, 32767).astype(np.int16)
   }
   ```

3. **Quality Metrics:**
   - DC offset measurement (target: <0.001%)
   - Symmetric clipping validation
   - THD at various amplitudes
   - Quantization noise floor
   - Headroom management
   - Round-trip accuracy (Float32→PCM16→Float32)

4. **OpenAI Compatibility:**
   - Validate PCM16 format compliance
   - Test with OpenAI Realtime API
   - Verify transcription accuracy
   - Check for format-related errors

### Implementation Guidelines

#### File Structure
Main tool: `tools/audio_format_validator.py`

Supporting files:
```
tools/format_validation/
├── pcm16_converter.py      # Optimized conversion implementations
├── dc_bias_analyzer.py     # DC offset detection and correction
├── symmetry_tester.py      # Clipping symmetry validation
├── bit_depth_validator.py  # Bit depth preservation tests
└── openai_compatibility.py # API format compliance checker
```

#### Code Architecture
```python
class AudioFormatValidator:
    def __init__(self):
        self.conversion_methods = {}
        self.test_results = {}
        
    def analyze_conversion_method(self, method: callable, 
                                 signal: np.ndarray) -> dict:
        # Test conversion quality metrics
        
    def measure_dc_bias(self, pcm_data: np.ndarray) -> float:
        # Calculate DC offset in converted signal
        
    def test_symmetry(self, method: callable) -> bool:
        # Verify symmetric clipping behavior
        
    def validate_bit_depth(self, original: np.ndarray, 
                          converted: np.ndarray) -> dict:
        # Check bit depth preservation
        
    def test_openai_compatibility(self, pcm_data: bytes) -> dict:
        # Validate with OpenAI requirements
        
    def generate_optimal_converter(self) -> callable:
        # Return optimized conversion function
```

#### Integration Points
```python
# Current implementation to validate/replace
# src/audio/capture.py:325-330
resampled = np.clip(resampled, -1.0, 1.0)
pcm16_data = (resampled * 32767).astype(np.int16)

# Test with actual pipeline
from src.audio.capture import AudioCapture
from src.openai_client.realtime import OpenAIRealtimeClient
```

#### Validation Protocol

1. **DC Bias Testing:**
   - Generate test signals with known DC offsets
   - Convert and measure resulting bias
   - Identify optimal normalization factor

2. **Symmetry Validation:**
   - Test positive and negative clipping points
   - Ensure equal headroom in both directions
   - Validate no preference for positive/negative

3. **Quantization Analysis:**
   - Measure noise floor
   - Test with low-amplitude signals
   - Verify no information loss

4. **Round-trip Testing:**
   - Convert Float32→PCM16→Float32
   - Measure reconstruction error
   - Validate perceptual transparency

5. **API Compliance:**
   - Send test audio to OpenAI
   - Verify acceptance and processing
   - Test transcription accuracy

### Acceptance Criteria
- [ ] Zero DC bias (<0.001% offset)
- [ ] Symmetric clipping behavior confirmed
- [ ] Quantization noise below -90dB
- [ ] No audible artifacts in conversion
- [ ] OpenAI API accepts format without errors
- [ ] Round-trip error <0.01% for typical signals
- [ ] Headroom properly managed (no premature clipping)
- [ ] Documentation of optimal conversion method
- [ ] Integration code ready for src/audio/capture.py

### Resources Available

#### From Previous Tasks
- Diagnostic tool from Task 1.1
- Optimized resampling from Task 1.2
- Calibrated gain levels from Task 1.3

#### Reference Implementation
```python
# NumPy's recommended approach
def float_to_pcm16(audio):
    """Convert float32 audio to PCM16 with proper scaling"""
    # Ensure input is in [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    
    # Scale to int16 range
    # Use 32767 to maintain symmetry
    return (audio * 32767).astype(np.int16)
```

#### Test Data
```python
# Generate comprehensive test suite
test_suite = {
    'sine_1khz': np.sin(2 * np.pi * 1000 * t),
    'white_noise': np.random.randn(48000),
    'impulse': np.zeros(1000); impulse[500] = 1.0,
    'dc_offset': np.ones(1000) * 0.1,
    'speech_sample': load_speech_sample()
}
```

### Success Metrics
- DC bias eliminated (currently suspected)
- Symmetric clipping achieved
- No degradation in audio quality
- OpenAI transcription accuracy maintained
- CPU overhead: <1% for conversion

### Deliverables
1. **Primary Tool:** `tools/audio_format_validator.py`
2. **Validation Suite:** `tools/format_validation/` modules
3. **Report:** `reports/pcm16_conversion_analysis.json`
4. **Optimal Converter:** `src/audio/pcm16_converter.py`
5. **Documentation:** Technical note on conversion methodology
6. **Integration PR:** Updated src/audio/capture.py with validated conversion

### Timeline
**Duration:** 2 days
- Day 1: Analysis and testing of conversion methods
- Day 2: OpenAI validation and integration preparation

### Critical Considerations

#### The 32767 vs 32768 Debate
```python
# Method 1: Symmetric but doesn't use full range
max_int16 = 32767  # Uses [-32767, 32767], wastes -32768

# Method 2: Asymmetric but uses full range  
max_int16 = 32768  # Uses [-32768, 32767], slight negative bias
```

#### OpenAI Requirements
- Format: 16-bit PCM, little-endian
- Sample rate: 24000 Hz (handled by resampling)
- Channels: Mono
- No header (raw PCM data)

#### Common Pitfalls
- DC bias from asymmetric scaling
- Clipping from improper headroom
- Quantization noise from poor dithering
- Endianness issues on different platforms

### Getting Started
1. Review current PCM16 conversion at src/audio/capture.py:325-330
2. Create test signal generator
3. Implement multiple conversion methods
4. Measure quality metrics for each
5. Validate with OpenAI API
6. Select and document optimal method
7. Prepare integration code

### Questions to Address
- Should we implement dithering for low-level signals?
- Is the 1-sample asymmetry of 32768 scaling audible?
- How does endianness affect different Raspberry Pi models?
- Should we add a configurable headroom parameter?

## Your Response Should Include
1. Acknowledgment of Tasks 1.1-1.3 completion
2. Your approach to validating PCM16 conversion
3. Initial hypothesis on 32767 vs 32768
4. Testing methodology for DC bias
5. OpenAI API validation plan

Remember: This is the final piece of audio quality optimization. Proper PCM16 conversion ensures that all the improvements from previous tasks are preserved when sending audio to OpenAI. Even small conversion errors can degrade the user experience.

---

*Reference the updated APM_Memory_Bank.md with all Phase 1 progress and APM_Implementation_Plan.md for context.*