"""
Device Profiler for Audio Input Devices

Identifies and profiles audio input devices, providing device-specific
recommendations for gain optimization.
"""
import os
import re
import json
import hashlib
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class DeviceInfo:
    """Information about an audio device"""
    name: str
    device_id: int
    channels: int
    sample_rate: int
    device_type: str  # Detected type (usb_generic, respeaker, etc.)
    unique_id: str    # Hash of device characteristics
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    driver: Optional[str] = None
    
    def to_profile_name(self) -> str:
        """Generate a profile filename for this device"""
        # Sanitize name for filename
        safe_name = re.sub(r'[^\w\-_]', '_', self.name.lower())
        return f"{self.device_type}_{safe_name[:30]}_{self.unique_id[:8]}.yaml"


class DeviceProfiler:
    """
    Profiles audio input devices and manages device-specific configurations
    """
    
    # Known device patterns for identification
    DEVICE_PATTERNS = {
        'respeaker': {
            'patterns': ['respeaker', '2-mic', '4-mic', 'seeed'],
            'type': 'respeaker_hat',
            'characteristics': {
                'high_sensitivity': True,
                'array_mic': True,
                'recommended_agc': False  # Has hardware processing
            }
        },
        'jabra': {
            'patterns': ['jabra', 'speak'],
            'type': 'conference_speaker',
            'characteristics': {
                'internal_agc': True,
                'echo_cancellation': True,
                'recommended_agc': False
            }
        },
        'blue_yeti': {
            'patterns': ['blue', 'yeti', 'blue yeti'],
            'type': 'studio_mic',
            'characteristics': {
                'high_quality': True,
                'multiple_patterns': True,
                'recommended_agc': True
            }
        },
        'logitech': {
            'patterns': ['logitech', 'webcam', 'c920', 'c922', 'brio'],
            'type': 'webcam_mic',
            'characteristics': {
                'low_quality': True,
                'omnidirectional': True,
                'recommended_agc': True,
                'needs_boost': True
            }
        },
        'usb_audio': {
            'patterns': ['usb audio', 'usb mic', 'generic usb'],
            'type': 'usb_generic',
            'characteristics': {
                'variable_quality': True,
                'recommended_agc': True
            }
        },
        'built_in': {
            'patterns': ['built-in', 'internal', 'macbook', 'laptop'],
            'type': 'built_in_mic',
            'characteristics': {
                'noise_prone': True,
                'recommended_agc': True,
                'needs_noise_reduction': True
            }
        }
    }
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize device profiler
        
        Args:
            profiles_dir: Directory to store/load device profiles
        """
        if profiles_dir:
            self.profiles_dir = Path(profiles_dir)
        else:
            self.profiles_dir = Path(__file__).parent.parent.parent / "config" / "audio_profiles"
        
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.device_cache = {}
    
    def list_audio_devices(self) -> List[DeviceInfo]:
        """
        List all available audio input devices
        
        Returns:
            List of DeviceInfo objects
        """
        devices = []
        
        if SOUNDDEVICE_AVAILABLE:
            # Use sounddevice to enumerate devices
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:  # Input device
                    device_type = self._identify_device_type(device['name'])
                    unique_id = self._generate_device_id(device)
                    
                    info = DeviceInfo(
                        name=device['name'],
                        device_id=i,
                        channels=device['max_input_channels'],
                        sample_rate=int(device['default_samplerate']),
                        device_type=device_type,
                        unique_id=unique_id,
                        driver=device.get('hostapi', 'Unknown')
                    )
                    devices.append(info)
        else:
            # Fallback: Try to get device info from system
            devices = self._get_devices_from_system()
        
        return devices
    
    def _identify_device_type(self, device_name: str) -> str:
        """
        Identify device type based on name patterns
        
        Args:
            device_name: Name of the device
            
        Returns:
            Device type identifier
        """
        name_lower = device_name.lower()
        
        for device_key, config in self.DEVICE_PATTERNS.items():
            for pattern in config['patterns']:
                if pattern in name_lower:
                    return config['type']
        
        # Default to generic USB if not identified
        if 'usb' in name_lower:
            return 'usb_generic'
        
        return 'unknown'
    
    def _generate_device_id(self, device_info: dict) -> str:
        """
        Generate unique ID for a device based on its characteristics
        
        Args:
            device_info: Device information dictionary
            
        Returns:
            Unique device identifier (hash)
        """
        # Create a stable hash based on device characteristics
        characteristics = f"{device_info.get('name', '')}_{device_info.get('default_samplerate', 0)}_{device_info.get('max_input_channels', 0)}"
        return hashlib.md5(characteristics.encode()).hexdigest()
    
    def _get_devices_from_system(self) -> List[DeviceInfo]:
        """
        Fallback method to get devices from system (Linux)
        
        Returns:
            List of DeviceInfo objects
        """
        devices = []
        
        try:
            # Try arecord -l on Linux
            result = subprocess.run(['arecord', '-l'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            
            if result.returncode == 0:
                # Parse arecord output
                current_card = None
                for line in result.stdout.split('\n'):
                    if 'card' in line.lower():
                        # Extract card and device info
                        match = re.search(r'card (\d+).*device (\d+): (.+)', line)
                        if match:
                            card_num = match.group(1)
                            device_num = match.group(2)
                            device_name = match.group(3)
                            
                            device_type = self._identify_device_type(device_name)
                            unique_id = hashlib.md5(f"hw:{card_num},{device_num}".encode()).hexdigest()
                            
                            info = DeviceInfo(
                                name=device_name,
                                device_id=int(f"{card_num}{device_num}"),
                                channels=2,  # Default assumption
                                sample_rate=48000,  # Default assumption
                                device_type=device_type,
                                unique_id=unique_id,
                                driver='ALSA'
                            )
                            devices.append(info)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return devices
    
    def profile_device(self, device: DeviceInfo, test_duration: float = 1.0) -> Dict:
        """
        Create a detailed profile for a specific device
        
        Args:
            device: DeviceInfo object to profile
            test_duration: Duration of test recording in seconds
            
        Returns:
            Device profile dictionary
        """
        profile = {
            'device_info': asdict(device),
            'characteristics': {},
            'recommended_settings': {},
            'test_results': {}
        }
        
        # Get known characteristics if device type is recognized
        for pattern_key, config in self.DEVICE_PATTERNS.items():
            if config['type'] == device.device_type:
                profile['characteristics'] = config.get('characteristics', {})
                break
        
        # Perform basic audio tests if sounddevice is available
        if SOUNDDEVICE_AVAILABLE:
            try:
                import numpy as np
                
                # Record a short sample to analyze noise floor
                print(f"Profiling device: {device.name}")
                print("Please ensure the environment is quiet...")
                
                recording = sd.rec(
                    int(test_duration * device.sample_rate),
                    samplerate=device.sample_rate,
                    channels=1,
                    device=device.device_id,
                    dtype='float32'
                )
                sd.wait()
                
                # Analyze recording
                rms = np.sqrt(np.mean(recording**2))
                peak = np.max(np.abs(recording))
                noise_floor_db = 20 * np.log10(rms + 1e-10)
                
                profile['test_results'] = {
                    'noise_floor_rms': float(rms),
                    'noise_floor_db': float(noise_floor_db),
                    'peak_noise': float(peak)
                }
                
                # Generate recommendations based on noise floor
                if noise_floor_db < -50:
                    quality = 'excellent'
                    base_gain = 1.0
                elif noise_floor_db < -40:
                    quality = 'good'
                    base_gain = 0.8
                elif noise_floor_db < -30:
                    quality = 'fair'
                    base_gain = 0.6
                else:
                    quality = 'poor'
                    base_gain = 0.5
                
                profile['test_results']['quality_assessment'] = quality
                
            except Exception as e:
                print(f"Could not perform audio test: {e}")
        
        # Generate recommended settings
        profile['recommended_settings'] = self._generate_recommendations(device, profile)
        
        return profile
    
    def _generate_recommendations(self, device: DeviceInfo, profile: Dict) -> Dict:
        """
        Generate recommended gain settings for a device
        
        Args:
            device: DeviceInfo object
            profile: Device profile with test results
            
        Returns:
            Dictionary of recommended settings
        """
        recommendations = {
            'input_volume': 1.0,
            'agc_enabled': True,
            'agc_max_gain': 2.0,
            'agc_target_rms': 0.3,
            'wake_word_gain': 1.0,
            'highpass_filter_enabled': True,
            'highpass_filter_cutoff': 80.0
        }
        
        characteristics = profile.get('characteristics', {})
        test_results = profile.get('test_results', {})
        
        # Adjust based on device characteristics
        if characteristics.get('internal_agc'):
            recommendations['agc_enabled'] = False
            recommendations['input_volume'] = 0.7
        
        if characteristics.get('high_sensitivity'):
            recommendations['input_volume'] = 0.5
            recommendations['wake_word_gain'] = 0.8
        
        if characteristics.get('needs_boost'):
            recommendations['input_volume'] = 1.5
            recommendations['agc_max_gain'] = 3.0
        
        if characteristics.get('needs_noise_reduction'):
            recommendations['highpass_filter_cutoff'] = 100.0
        
        # Adjust based on test results
        quality = test_results.get('quality_assessment')
        if quality == 'excellent':
            recommendations['agc_target_rms'] = 0.25
        elif quality == 'poor':
            recommendations['agc_target_rms'] = 0.35
            recommendations['agc_max_gain'] = min(recommendations['agc_max_gain'] * 1.5, 3.0)
        
        # Calculate safe cumulative gain
        cumulative_gain = (recommendations['input_volume'] * 
                          (recommendations['agc_max_gain'] if recommendations['agc_enabled'] else 1.0) *
                          recommendations['wake_word_gain'])
        
        # Ensure cumulative gain doesn't exceed safe threshold
        if cumulative_gain > 8.0:
            scale_factor = 8.0 / cumulative_gain
            recommendations['wake_word_gain'] *= scale_factor
        
        recommendations['estimated_cumulative_gain'] = cumulative_gain
        
        return recommendations
    
    def save_profile(self, device: DeviceInfo, profile: Dict) -> str:
        """
        Save device profile to file
        
        Args:
            device: DeviceInfo object
            profile: Device profile dictionary
            
        Returns:
            Path to saved profile file
        """
        import yaml
        
        filename = device.to_profile_name()
        filepath = self.profiles_dir / filename
        
        # Add metadata
        profile['metadata'] = {
            'created_at': str(Path.cwd()),
            'version': '1.0',
            'device_id': device.unique_id
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False)
        
        return str(filepath)
    
    def load_profile(self, device_id: str) -> Optional[Dict]:
        """
        Load existing profile for a device
        
        Args:
            device_id: Unique device identifier
            
        Returns:
            Profile dictionary or None if not found
        """
        import yaml
        
        # Search for matching profile
        for profile_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(profile_file, 'r') as f:
                    profile = yaml.safe_load(f)
                    if profile.get('metadata', {}).get('device_id') == device_id:
                        return profile
            except Exception:
                continue
        
        return None
    
    def auto_detect_device(self) -> Optional[DeviceInfo]:
        """
        Auto-detect the default or most suitable input device
        
        Returns:
            DeviceInfo for the detected device or None
        """
        devices = self.list_audio_devices()
        
        if not devices:
            return None
        
        # Priority order for auto-detection
        priority_types = [
            'respeaker_hat',      # Best for Raspberry Pi
            'studio_mic',         # High quality USB mics
            'conference_speaker', # Good all-around devices
            'usb_generic',       # Generic USB devices
            'webcam_mic',        # Common but lower quality
            'built_in_mic'       # Last resort
        ]
        
        # Try to find device by priority
        for device_type in priority_types:
            for device in devices:
                if device.device_type == device_type:
                    return device
        
        # Return first available device as fallback
        return devices[0] if devices else None


if __name__ == "__main__":
    # Test device profiling
    profiler = DeviceProfiler()
    
    print("Detecting audio input devices...")
    devices = profiler.list_audio_devices()
    
    if not devices:
        print("No audio input devices found!")
    else:
        print(f"\nFound {len(devices)} input device(s):")
        for i, device in enumerate(devices):
            print(f"{i+1}. {device.name}")
            print(f"   Type: {device.device_type}")
            print(f"   Channels: {device.channels}")
            print(f"   Sample Rate: {device.sample_rate} Hz")
            print(f"   ID: {device.unique_id[:8]}...")
        
        # Auto-detect best device
        best_device = profiler.auto_detect_device()
        if best_device:
            print(f"\nAuto-detected best device: {best_device.name}")
            
            # Profile the device
            print(f"\nProfiling {best_device.name}...")
            profile = profiler.profile_device(best_device, test_duration=0.5)
            
            print("\nDevice Profile:")
            print(f"Characteristics: {profile['characteristics']}")
            print(f"Recommended Settings:")
            for key, value in profile['recommended_settings'].items():
                print(f"  {key}: {value}")
            
            if profile.get('test_results'):
                print(f"\nTest Results:")
                print(f"  Noise floor: {profile['test_results'].get('noise_floor_db', 'N/A'):.1f} dB")
                print(f"  Quality: {profile['test_results'].get('quality_assessment', 'N/A')}")
            
            # Save profile
            profile_path = profiler.save_profile(best_device, profile)
            print(f"\nProfile saved to: {profile_path}")