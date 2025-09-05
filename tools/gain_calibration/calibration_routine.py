"""
Interactive Calibration Routine for Gain Optimization

Provides an interactive CLI interface for users to calibrate their
audio setup with real-time feedback and visual meters.
"""
import sys
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from queue import Queue, Empty
import os

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
class CalibrationStep:
    """Single step in calibration process"""
    name: str
    description: str
    duration: float
    action: str
    target_level: Optional[float] = None
    
    
@dataclass 
class CalibrationResult:
    """Results from calibration process"""
    device_name: str
    noise_floor: float
    optimal_input_volume: float
    optimal_agc_enabled: bool
    optimal_agc_max_gain: float
    optimal_wake_word_gain: float
    clipping_threshold: float
    test_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    

class CalibrationRoutine:
    """
    Interactive calibration routine with real-time feedback
    """
    
    # Calibration steps
    CALIBRATION_STEPS = [
        CalibrationStep(
            name="noise_floor",
            description="Measuring ambient noise level",
            duration=3.0,
            action="Please remain quiet while we measure the room noise",
            target_level=None
        ),
        CalibrationStep(
            name="quiet_speech",
            description="Testing quiet speech level",
            duration=5.0,
            action="Please speak quietly (as if someone is sleeping nearby)",
            target_level=0.1
        ),
        CalibrationStep(
            name="normal_speech", 
            description="Testing normal conversation level",
            duration=5.0,
            action="Please speak at your normal conversation volume",
            target_level=0.3
        ),
        CalibrationStep(
            name="loud_speech",
            description="Testing loud speech level",
            duration=5.0,
            action="Please speak loudly (as if calling someone in another room)",
            target_level=0.5
        ),
        CalibrationStep(
            name="wake_word_test",
            description="Testing wake word detection",
            duration=10.0,
            action="Please say your wake word 3 times with 2-second pauses",
            target_level=0.3
        )
    ]
    
    def __init__(self, device_id: Optional[int] = None, sample_rate: int = 48000):
        """
        Initialize calibration routine
        
        Args:
            device_id: Audio device ID (None for default)
            sample_rate: Sample rate for audio capture
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        
        # Audio processing state
        self.is_recording = False
        self.audio_queue = Queue()
        self.level_history = []
        self.current_step = None
        
        # Results storage
        self.measurements = {}
        self.optimal_gains = {}
        
        # Visual feedback
        self.meter_width = 50
        self.update_interval = 0.1  # Update display every 100ms
    
    def start_audio_stream(self) -> bool:
        """
        Start audio streaming for calibration
        
        Returns:
            True if stream started successfully
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("ERROR: sounddevice not available. Please install it.")
            return False
        
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio callback status: {status}")
                if self.is_recording:
                    self.audio_queue.put(indata.copy())
            
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype='float32'
            )
            self.stream.start()
            return True
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
    
    def stop_audio_stream(self):
        """Stop audio streaming"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def measure_audio_level(self, duration: float) -> Dict[str, float]:
        """
        Measure audio levels for specified duration
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            Dictionary with level statistics
        """
        self.is_recording = True
        start_time = time.time()
        audio_chunks = []
        
        while time.time() - start_time < duration:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
            except Empty:
                continue
        
        self.is_recording = False
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        if not audio_chunks:
            return {'rms': 0.0, 'peak': 0.0, 'clipping': 0.0}
        
        # Concatenate audio
        audio_data = np.concatenate(audio_chunks)
        
        # Calculate metrics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
        
        # Calculate frequency content if scipy available
        if SCIPY_AVAILABLE:
            freqs, psd = signal.welch(audio_data, self.sample_rate)
            peak_freq_idx = np.argmax(psd)
            peak_freq = freqs[peak_freq_idx]
        else:
            peak_freq = 0.0
        
        return {
            'rms': float(rms),
            'peak': float(peak),
            'clipping': float(clipping_ratio),
            'peak_frequency': float(peak_freq),
            'duration': duration,
            'samples': len(audio_data)
        }
    
    def display_meter(self, level: float, peak: float, target: Optional[float] = None):
        """
        Display audio level meter in terminal
        
        Args:
            level: Current RMS level (0-1)
            peak: Peak level (0-1)
            target: Target level to display
        """
        # Clear line
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        
        # Create meter
        meter_chars = self.meter_width
        filled = int(level * meter_chars)
        peak_pos = int(peak * meter_chars)
        
        # Build meter string
        meter = '['
        for i in range(meter_chars):
            if target and i == int(target * meter_chars):
                meter += '|'  # Target marker
            elif i < filled:
                if i < meter_chars * 0.6:
                    meter += '='  # Green zone
                elif i < meter_chars * 0.8:
                    meter += '≈'  # Yellow zone  
                else:
                    meter += '≡'  # Red zone
            elif i == peak_pos:
                meter += '!'  # Peak marker
            else:
                meter += '-'
        meter += ']'
        
        # Add level values
        level_db = 20 * np.log10(level + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        output = f"Level: {meter} RMS:{level_db:+.1f}dB Peak:{peak_db:+.1f}dB"
        
        # Add status indicators
        if peak > 0.99:
            output += " [CLIPPING!]"
        elif peak > 0.9:
            output += " [HIGH]"
        elif level < 0.05:
            output += " [LOW]"
        else:
            output += " [OK]"
        
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def run_calibration_step(self, step: CalibrationStep) -> Dict[str, float]:
        """
        Run a single calibration step
        
        Args:
            step: Calibration step to run
            
        Returns:
            Measurement results
        """
        print(f"\n{'-' * 60}")
        print(f"Step: {step.description}")
        print(f"{'-' * 60}")
        print(f"\n{step.action}\n")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("\n🔴 RECORDING - Please begin!\n")
        
        # Real-time monitoring
        self.is_recording = True
        start_time = time.time()
        measurements = []
        peak_level = 0.0
        
        while time.time() - start_time < step.duration:
            remaining = step.duration - (time.time() - start_time)
            
            # Get audio chunk
            try:
                chunk = self.audio_queue.get(timeout=0.05)
                rms = np.sqrt(np.mean(chunk**2))
                peak = np.max(np.abs(chunk))
                peak_level = max(peak_level, peak)
                
                measurements.append({
                    'rms': rms,
                    'peak': peak,
                    'time': time.time() - start_time
                })
                
                # Display meter
                self.display_meter(rms, peak_level, step.target_level)
                
            except Empty:
                continue
        
        self.is_recording = False
        print("\n\n✅ Recording complete!\n")
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        # Calculate statistics
        if measurements:
            rms_values = [m['rms'] for m in measurements]
            peak_values = [m['peak'] for m in measurements]
            
            result = {
                'mean_rms': np.mean(rms_values),
                'std_rms': np.std(rms_values),
                'max_peak': np.max(peak_values),
                'clipping_ratio': sum(1 for p in peak_values if p > 0.99) / len(peak_values)
            }
        else:
            result = {'mean_rms': 0.0, 'std_rms': 0.0, 'max_peak': 0.0, 'clipping_ratio': 0.0}
        
        return result
    
    def calculate_optimal_gains(self, measurements: Dict) -> Dict[str, float]:
        """
        Calculate optimal gain settings based on measurements
        
        Args:
            measurements: Dictionary of calibration measurements
            
        Returns:
            Optimal gain settings
        """
        noise_floor = measurements.get('noise_floor', {}).get('mean_rms', 0.01)
        normal_level = measurements.get('normal_speech', {}).get('mean_rms', 0.3)
        loud_peak = measurements.get('loud_speech', {}).get('max_peak', 0.8)
        
        # Calculate signal-to-noise ratio
        snr = normal_level / (noise_floor + 1e-10)
        snr_db = 20 * np.log10(snr)
        
        # Determine base input volume
        # Target: normal speech at 0.3 RMS without clipping on loud speech
        target_normal_rms = 0.3
        target_loud_peak = 0.85  # Leave headroom
        
        # Calculate gain needed for normal speech
        gain_for_normal = target_normal_rms / (normal_level + 1e-10)
        
        # Check if this would cause clipping on loud speech
        loud_with_gain = loud_peak * gain_for_normal
        
        if loud_with_gain > target_loud_peak:
            # Reduce gain to prevent clipping
            gain_for_normal = target_loud_peak / loud_peak
        
        # Determine AGC settings
        if snr_db < 20:
            # Poor SNR, need aggressive AGC
            agc_enabled = True
            agc_max_gain = 3.0
            input_volume = gain_for_normal * 0.7  # Lower input, let AGC do work
        elif snr_db < 30:
            # Moderate SNR, use moderate AGC
            agc_enabled = True
            agc_max_gain = 2.0
            input_volume = gain_for_normal * 0.85
        else:
            # Good SNR, minimal AGC needed
            agc_enabled = True
            agc_max_gain = 1.5
            input_volume = gain_for_normal
        
        # Wake word gain adjustment
        # Start conservative, can be increased if needed
        wake_word_gain = 1.0
        
        # Calculate cumulative gain
        cumulative = input_volume * (agc_max_gain if agc_enabled else 1.0) * wake_word_gain
        
        # Ensure cumulative gain is safe
        if cumulative > 8.0:
            # Scale down wake word gain
            wake_word_gain = 8.0 / (input_volume * (agc_max_gain if agc_enabled else 1.0))
        
        return {
            'input_volume': round(input_volume, 2),
            'agc_enabled': agc_enabled,
            'agc_max_gain': round(agc_max_gain, 1),
            'agc_target_rms': round(target_normal_rms, 2),
            'wake_word_gain': round(wake_word_gain, 2),
            'estimated_snr_db': round(snr_db, 1),
            'cumulative_gain': round(cumulative, 1)
        }
    
    def run_interactive_calibration(self) -> CalibrationResult:
        """
        Run the complete interactive calibration process
        
        Returns:
            CalibrationResult with optimal settings
        """
        print("\n" + "="*60)
        print("   🎤 AUDIO CALIBRATION WIZARD 🎤")
        print("="*60)
        print("\nThis wizard will help optimize your audio settings for")
        print("best wake word detection and speech recognition.")
        print("\nThe process takes about 30 seconds.\n")
        
        input("Press ENTER to begin calibration...")
        
        # Start audio stream
        if not self.start_audio_stream():
            raise RuntimeError("Failed to start audio stream")
        
        try:
            # Run each calibration step
            for step in self.CALIBRATION_STEPS:
                result = self.run_calibration_step(step)
                self.measurements[step.name] = result
                
                # Brief pause between steps
                time.sleep(1)
            
            # Calculate optimal settings
            optimal = self.calculate_optimal_gains(self.measurements)
            
            # Get device info
            if SOUNDDEVICE_AVAILABLE:
                device_info = sd.query_devices(self.device_id)
                device_name = device_info['name']
            else:
                device_name = "Unknown Device"
            
            # Create result
            result = CalibrationResult(
                device_name=device_name,
                noise_floor=self.measurements['noise_floor']['mean_rms'],
                optimal_input_volume=optimal['input_volume'],
                optimal_agc_enabled=optimal['agc_enabled'],
                optimal_agc_max_gain=optimal['agc_max_gain'],
                optimal_wake_word_gain=optimal['wake_word_gain'],
                clipping_threshold=0.95,
                test_scores={
                    'snr_db': optimal['estimated_snr_db'],
                    'cumulative_gain': optimal['cumulative_gain']
                }
            )
            
            # Add warnings if needed
            if optimal['cumulative_gain'] > 10:
                result.warnings.append("High cumulative gain detected - may cause distortion")
            
            if optimal['estimated_snr_db'] < 20:
                result.warnings.append("Low signal-to-noise ratio - consider reducing background noise")
            
            noise_db = 20 * np.log10(result.noise_floor + 1e-10)
            if noise_db > -30:
                result.warnings.append("High ambient noise detected - may affect wake word detection")
            
            # Display results
            self.display_results(result, optimal)
            
            return result
            
        finally:
            self.stop_audio_stream()
    
    def display_results(self, result: CalibrationResult, optimal: Dict):
        """
        Display calibration results to user
        
        Args:
            result: CalibrationResult object
            optimal: Dictionary with additional optimal settings
        """
        print("\n" + "="*60)
        print("   📊 CALIBRATION RESULTS 📊")
        print("="*60)
        
        print(f"\nDevice: {result.device_name}")
        print(f"Noise Floor: {20*np.log10(result.noise_floor + 1e-10):.1f} dB")
        print(f"Signal-to-Noise Ratio: {optimal['estimated_snr_db']:.1f} dB")
        
        print("\n" + "-"*40)
        print("RECOMMENDED SETTINGS:")
        print("-"*40)
        
        print(f"Input Volume: {result.optimal_input_volume}")
        print(f"AGC Enabled: {result.optimal_agc_enabled}")
        if result.optimal_agc_enabled:
            print(f"AGC Max Gain: {result.optimal_agc_max_gain}")
            print(f"AGC Target RMS: {optimal.get('agc_target_rms', 0.3)}")
        print(f"Wake Word Gain: {result.optimal_wake_word_gain}")
        print(f"Cumulative Gain: {optimal['cumulative_gain']}x")
        
        if result.warnings:
            print("\n" + "-"*40)
            print("⚠️  WARNINGS:")
            print("-"*40)
            for warning in result.warnings:
                print(f"• {warning}")
        
        print("\n" + "="*60)
        
        # Quality assessment
        if optimal['estimated_snr_db'] > 30 and optimal['cumulative_gain'] < 8:
            print("✅ EXCELLENT: Your audio setup is well optimized!")
        elif optimal['estimated_snr_db'] > 20 and optimal['cumulative_gain'] < 10:
            print("👍 GOOD: Your audio setup should work well.")
        else:
            print("⚠️  FAIR: Consider the warnings above for better performance.")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Run calibration test
    print("Audio Calibration Routine Test")
    
    if not SOUNDDEVICE_AVAILABLE:
        print("ERROR: sounddevice is required for calibration")
        print("Install with: pip install sounddevice")
        sys.exit(1)
    
    # List available devices
    print("\nAvailable audio input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} ({device['max_input_channels']} channels)")
    
    # Run calibration
    try:
        calibration = CalibrationRoutine()
        result = calibration.run_interactive_calibration()
        
        print("\nCalibration completed successfully!")
        print(f"Results saved for device: {result.device_name}")
        
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled by user.")
    except Exception as e:
        print(f"\nERROR: {e}")