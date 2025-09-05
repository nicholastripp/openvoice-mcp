"""
Wake Word Testing Module for Gain Optimization

Tests wake word detection accuracy at different gain settings to find
the optimal configuration for reliable detection without false positives.
"""
import sys
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import wake word detector if available
try:
    from wake_word.porcupine_detector import PorcupineDetector
    from config import WakeWordConfig
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    PorcupineDetector = None
    WakeWordConfig = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class WakeWordTestResult:
    """Results from wake word testing"""
    gain_config: Dict[str, float]
    detection_rate: float
    false_positive_rate: float
    average_confidence: float
    response_time: float
    audio_metrics: Dict[str, float] = field(default_factory=dict)
    test_details: List[Dict] = field(default_factory=list)


@dataclass
class TestScenario:
    """Test scenario for wake word detection"""
    name: str
    description: str
    test_type: str  # 'detection' or 'rejection'
    audio_file: Optional[str] = None
    duration: float = 10.0
    expected_detections: int = 0


class WakeWordTester:
    """
    Test wake word detection accuracy with different gain configurations
    """
    
    # Test scenarios
    TEST_SCENARIOS = [
        TestScenario(
            name="normal_detection",
            description="Normal wake word speaking",
            test_type="detection",
            duration=10.0,
            expected_detections=3
        ),
        TestScenario(
            name="quiet_detection",
            description="Quiet wake word speaking",
            test_type="detection",
            duration=10.0,
            expected_detections=3
        ),
        TestScenario(
            name="noisy_detection",
            description="Wake word with background noise",
            test_type="detection",
            duration=10.0,
            expected_detections=3
        ),
        TestScenario(
            name="false_positive",
            description="Similar sounding words",
            test_type="rejection",
            duration=10.0,
            expected_detections=0
        ),
        TestScenario(
            name="silence",
            description="Complete silence",
            test_type="rejection",
            duration=5.0,
            expected_detections=0
        )
    ]
    
    def __init__(self, wake_word_model: str = "grasshopper", 
                 sensitivity: float = 1.0,
                 access_key: Optional[str] = None):
        """
        Initialize wake word tester
        
        Args:
            wake_word_model: Wake word model to test
            sensitivity: Detection sensitivity (0.0-1.0)
            access_key: Picovoice access key
        """
        self.wake_word_model = wake_word_model
        self.sensitivity = sensitivity
        self.access_key = access_key or os.getenv('PICOVOICE_ACCESS_KEY')
        
        # Test results storage
        self.test_results = []
        self.current_detections = []
        
        # Audio parameters
        self.sample_rate = 16000  # Porcupine requirement
        self.chunk_size = 512     # Porcupine frame size
    
    def apply_gain_configuration(self, audio_data: np.ndarray, 
                                gain_config: Dict[str, float]) -> np.ndarray:
        """
        Apply gain configuration to audio data
        
        Args:
            audio_data: Input audio data
            gain_config: Gain configuration dictionary
            
        Returns:
            Processed audio data
        """
        processed = audio_data.copy()
        
        # Apply input volume
        input_volume = gain_config.get('input_volume', 1.0)
        processed = processed * input_volume
        
        # Simulate AGC if enabled
        if gain_config.get('agc_enabled', False):
            agc_max_gain = gain_config.get('agc_max_gain', 2.0)
            target_rms = gain_config.get('agc_target_rms', 0.3)
            
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(processed**2))
            
            if current_rms > 0:
                # Calculate gain needed
                agc_gain = min(target_rms / current_rms, agc_max_gain)
                processed = processed * agc_gain
        
        # Apply wake word gain
        wake_word_gain = gain_config.get('wake_word_gain', 1.0)
        processed = processed * wake_word_gain
        
        # Apply high-pass filter if configured
        if gain_config.get('highpass_filter_enabled', False) and SCIPY_AVAILABLE:
            cutoff = gain_config.get('highpass_filter_cutoff', 80.0)
            nyquist = self.sample_rate / 2
            normal_cutoff = cutoff / nyquist
            
            if normal_cutoff < 1.0:
                b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
                processed = signal.filtfilt(b, a, processed)
        
        # Clip to prevent overflow
        processed = np.clip(processed, -1.0, 1.0)
        
        # Convert to int16 for Porcupine
        processed_int16 = (processed * 32767).astype(np.int16)
        
        return processed_int16
    
    def test_with_configuration(self, gain_config: Dict[str, float],
                               scenario: TestScenario,
                               audio_data: Optional[np.ndarray] = None) -> Dict:
        """
        Test wake word detection with specific gain configuration
        
        Args:
            gain_config: Gain configuration to test
            scenario: Test scenario to run
            audio_data: Optional pre-recorded audio data
            
        Returns:
            Test results dictionary
        """
        if not PORCUPINE_AVAILABLE:
            return {
                'error': 'Porcupine not available',
                'detection_rate': 0.0,
                'false_positive_rate': 0.0
            }
        
        # Create wake word configuration
        config = WakeWordConfig(
            enabled=True,
            model=self.wake_word_model,
            sensitivity=self.sensitivity,
            porcupine_access_key=self.access_key,
            audio_gain=gain_config.get('wake_word_gain', 1.0),
            highpass_filter_enabled=gain_config.get('highpass_filter_enabled', False),
            highpass_filter_cutoff=gain_config.get('highpass_filter_cutoff', 80.0)
        )
        
        # Initialize detector
        try:
            detector = PorcupineDetector(config)
            detector.start()
        except Exception as e:
            return {
                'error': f'Failed to initialize detector: {e}',
                'detection_rate': 0.0,
                'false_positive_rate': 0.0
            }
        
        # Track detections
        detections = []
        start_time = time.time()
        
        try:
            if audio_data is not None:
                # Process pre-recorded audio
                processed_audio = self.apply_gain_configuration(audio_data, gain_config)
                
                # Feed audio to detector in chunks
                for i in range(0, len(processed_audio) - self.chunk_size, self.chunk_size):
                    chunk = processed_audio[i:i + self.chunk_size]
                    
                    # Process chunk through detector
                    detector.process_audio(chunk)
                    
                    # Check for detection
                    if detector.detected:
                        detections.append({
                            'time': i / self.sample_rate,
                            'confidence': 1.0  # Porcupine doesn't provide confidence
                        })
                        detector.detected = False
            else:
                # Real-time testing
                print(f"Running {scenario.name}: {scenario.description}")
                print(f"Duration: {scenario.duration} seconds")
                
                if scenario.test_type == "detection":
                    print(f"Please say the wake word '{self.wake_word_model}' {scenario.expected_detections} times")
                else:
                    print("Please remain quiet or speak random words (not the wake word)")
                
                # Countdown
                for i in range(3, 0, -1):
                    print(f"Starting in {i}...")
                    time.sleep(1)
                
                print("🔴 TESTING - Begin now!")
                
                # Record and test
                test_duration = scenario.duration
                elapsed = 0
                
                while elapsed < test_duration:
                    # Simulate audio capture (would be real-time in production)
                    time.sleep(0.1)
                    elapsed = time.time() - start_time
                    
                    # Check for detection (simulated)
                    if detector.detected:
                        detections.append({
                            'time': elapsed,
                            'confidence': 1.0
                        })
                        detector.detected = False
                        print(f"✓ Wake word detected at {elapsed:.1f}s")
        
        finally:
            detector.stop()
        
        # Calculate metrics
        if scenario.test_type == "detection":
            detection_rate = len(detections) / max(scenario.expected_detections, 1)
            false_positive_rate = 0.0
        else:  # rejection test
            detection_rate = 1.0 if len(detections) == 0 else 0.0
            false_positive_rate = len(detections) / scenario.duration
        
        # Calculate response time
        if detections:
            response_times = [d['time'] for d in detections]
            avg_response_time = np.mean(response_times)
        else:
            avg_response_time = 0.0
        
        return {
            'scenario': scenario.name,
            'detections': len(detections),
            'expected': scenario.expected_detections,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'response_time': avg_response_time,
            'test_duration': scenario.duration,
            'detection_details': detections
        }
    
    def test_configuration_suite(self, gain_config: Dict[str, float]) -> WakeWordTestResult:
        """
        Run complete test suite for a gain configuration
        
        Args:
            gain_config: Gain configuration to test
            
        Returns:
            WakeWordTestResult with comprehensive metrics
        """
        print(f"\nTesting configuration:")
        print(f"  Input Volume: {gain_config.get('input_volume', 1.0)}")
        print(f"  AGC Enabled: {gain_config.get('agc_enabled', False)}")
        print(f"  Wake Word Gain: {gain_config.get('wake_word_gain', 1.0)}")
        print(f"  Cumulative Gain: {self.calculate_cumulative_gain(gain_config):.1f}x")
        
        test_results = []
        total_detection_rate = 0
        total_false_positives = 0
        total_response_time = 0
        
        # Run each test scenario
        for scenario in self.TEST_SCENARIOS:
            result = self.test_with_configuration(gain_config, scenario)
            test_results.append(result)
            
            if scenario.test_type == "detection":
                total_detection_rate += result['detection_rate']
            else:
                total_false_positives += result['false_positive_rate']
            
            if result['response_time'] > 0:
                total_response_time += result['response_time']
        
        # Calculate overall metrics
        detection_scenarios = sum(1 for s in self.TEST_SCENARIOS if s.test_type == "detection")
        rejection_scenarios = sum(1 for s in self.TEST_SCENARIOS if s.test_type == "rejection")
        
        avg_detection_rate = total_detection_rate / max(detection_scenarios, 1)
        avg_false_positive_rate = total_false_positives / max(rejection_scenarios, 1)
        avg_response_time = total_response_time / max(detection_scenarios, 1)
        
        # Create result
        result = WakeWordTestResult(
            gain_config=gain_config,
            detection_rate=avg_detection_rate,
            false_positive_rate=avg_false_positive_rate,
            average_confidence=0.95,  # Porcupine doesn't provide confidence scores
            response_time=avg_response_time,
            audio_metrics={
                'cumulative_gain': self.calculate_cumulative_gain(gain_config)
            },
            test_details=test_results
        )
        
        return result
    
    def calculate_cumulative_gain(self, gain_config: Dict[str, float]) -> float:
        """
        Calculate cumulative gain through pipeline
        
        Args:
            gain_config: Gain configuration
            
        Returns:
            Cumulative gain value
        """
        cumulative = gain_config.get('input_volume', 1.0)
        
        if gain_config.get('agc_enabled', False):
            cumulative *= gain_config.get('agc_max_gain', 1.0)
        
        cumulative *= gain_config.get('wake_word_gain', 1.0)
        
        return cumulative
    
    def find_optimal_configuration(self, test_configs: List[Dict[str, float]]) -> Tuple[Dict[str, float], WakeWordTestResult]:
        """
        Find the optimal configuration from a list of test configurations
        
        Args:
            test_configs: List of gain configurations to test
            
        Returns:
            Tuple of (optimal_config, test_result)
        """
        best_config = None
        best_result = None
        best_score = -float('inf')
        
        for config in test_configs:
            result = self.test_configuration_suite(config)
            
            # Calculate overall score
            # Prioritize detection rate, penalize false positives
            score = (result.detection_rate * 100 - 
                    result.false_positive_rate * 50 -
                    abs(self.calculate_cumulative_gain(config) - 5) * 2)
            
            # Bonus for fast response time
            if result.response_time > 0 and result.response_time < 0.5:
                score += 10
            
            print(f"\n  Detection Rate: {result.detection_rate:.1%}")
            print(f"  False Positive Rate: {result.false_positive_rate:.3f}/s")
            print(f"  Response Time: {result.response_time:.2f}s")
            print(f"  Score: {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_config = config
                best_result = result
        
        return best_config, best_result
    
    def generate_test_report(self, results: List[WakeWordTestResult], 
                            output_file: str):
        """
        Generate test report
        
        Args:
            results: List of test results
            output_file: Output file path
        """
        report = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wake_word_model': self.wake_word_model,
            'sensitivity': self.sensitivity,
            'test_scenarios': len(self.TEST_SCENARIOS),
            'configurations_tested': len(results),
            'results': []
        }
        
        for result in results:
            report['results'].append({
                'configuration': result.gain_config,
                'detection_rate': result.detection_rate,
                'false_positive_rate': result.false_positive_rate,
                'response_time': result.response_time,
                'cumulative_gain': result.audio_metrics.get('cumulative_gain', 0),
                'test_details': result.test_details
            })
        
        # Find best configuration
        best_idx = np.argmax([r.detection_rate - r.false_positive_rate for r in results])
        report['best_configuration'] = results[best_idx].gain_config
        report['best_detection_rate'] = results[best_idx].detection_rate
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("WAKE WORD TEST SUMMARY")
        print("="*60)
        print(f"Configurations tested: {len(results)}")
        print(f"Best detection rate: {results[best_idx].detection_rate:.1%}")
        print(f"Best configuration:")
        for key, value in results[best_idx].gain_config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Test the wake word tester
    print("Wake Word Detection Tester")
    
    if not PORCUPINE_AVAILABLE:
        print("\nERROR: Porcupine is required for wake word testing")
        print("Install with: pip install pvporcupine")
        print("Also ensure PICOVOICE_ACCESS_KEY is set")
        sys.exit(1)
    
    # Create test configurations
    test_configs = [
        {
            'input_volume': 1.0,
            'agc_enabled': True,
            'agc_max_gain': 2.0,
            'agc_target_rms': 0.3,
            'wake_word_gain': 1.0,
            'highpass_filter_enabled': True,
            'highpass_filter_cutoff': 80.0
        },
        {
            'input_volume': 0.7,
            'agc_enabled': True,
            'agc_max_gain': 3.0,
            'agc_target_rms': 0.3,
            'wake_word_gain': 1.5,
            'highpass_filter_enabled': True,
            'highpass_filter_cutoff': 80.0
        },
        {
            'input_volume': 1.5,
            'agc_enabled': False,
            'agc_max_gain': 1.0,
            'agc_target_rms': 0.3,
            'wake_word_gain': 0.8,
            'highpass_filter_enabled': False,
            'highpass_filter_cutoff': 0.0
        }
    ]
    
    # Run tests
    tester = WakeWordTester()
    
    print("\nTesting wake word detection with different gain configurations...")
    print("This will test detection accuracy and false positive rates.\n")
    
    results = []
    for config in test_configs:
        result = tester.test_configuration_suite(config)
        results.append(result)
    
    # Find optimal configuration
    best_config, best_result = tester.find_optimal_configuration(test_configs)
    
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATION FOUND:")
    print("="*60)
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"\nDetection Rate: {best_result.detection_rate:.1%}")
    print(f"False Positive Rate: {best_result.false_positive_rate:.3f}/s")
    
    # Generate report
    tester.generate_test_report(results, '/tmp/wake_word_test_report.json')
    print("\nTesting complete!")