"""
Profile Manager for Audio Device Configurations

Manages device-specific audio profiles including creation, loading,
validation, and application of optimal gain settings.
"""
import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AudioProfile:
    """Audio profile for a specific device"""
    device_name: str
    device_id: str
    profile_name: str
    created_at: str
    version: str
    
    # Gain settings
    input_volume: float
    agc_enabled: bool
    agc_max_gain: float
    agc_min_gain: float
    agc_target_rms: float
    agc_attack_time: float
    agc_release_time: float
    wake_word_gain: float
    
    # Filter settings
    highpass_filter_enabled: bool
    highpass_filter_cutoff: float
    
    # Performance metrics
    estimated_cumulative_gain: float
    expected_wake_word_accuracy: float
    expected_snr_db: float
    
    # Device characteristics
    device_characteristics: Dict[str, Any]
    
    # Validation scores
    validation_scores: Optional[Dict[str, float]] = None
    
    def to_config_dict(self) -> Dict:
        """Convert profile to config.yaml compatible dictionary"""
        return {
            'audio': {
                'input_volume': self.input_volume,
                'agc_enabled': self.agc_enabled,
                'agc_max_gain': self.agc_max_gain,
                'agc_min_gain': self.agc_min_gain,
                'agc_target_rms': self.agc_target_rms,
                'agc_attack_time': self.agc_attack_time,
                'agc_release_time': self.agc_release_time
            },
            'wake_word': {
                'audio_gain': self.wake_word_gain,
                'highpass_filter_enabled': self.highpass_filter_enabled,
                'highpass_filter_cutoff': self.highpass_filter_cutoff
            }
        }


class ProfileManager:
    """
    Manages audio device profiles for gain optimization
    """
    
    DEFAULT_PROFILE_VERSION = "1.0"
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize profile manager
        
        Args:
            profiles_dir: Directory to store profiles
        """
        if profiles_dir:
            self.profiles_dir = Path(profiles_dir)
        else:
            self.profiles_dir = Path(__file__).parent.parent.parent / "config" / "audio_profiles"
        
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default profiles if they don't exist
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """Create default profiles for common devices"""
        default_profiles = {
            'usb_generic.yaml': {
                'metadata': {
                    'device_name': 'Generic USB Microphone',
                    'device_id': 'usb_generic',
                    'profile_name': 'USB Generic Default',
                    'created_at': datetime.now().isoformat(),
                    'version': self.DEFAULT_PROFILE_VERSION
                },
                'gain_settings': {
                    'input_volume': 1.0,
                    'agc_enabled': True,
                    'agc_max_gain': 2.0,
                    'agc_min_gain': 0.5,
                    'agc_target_rms': 0.3,
                    'agc_attack_time': 0.5,
                    'agc_release_time': 2.0,
                    'wake_word_gain': 1.0
                },
                'filter_settings': {
                    'highpass_filter_enabled': True,
                    'highpass_filter_cutoff': 80.0
                },
                'performance_metrics': {
                    'estimated_cumulative_gain': 2.0,
                    'expected_wake_word_accuracy': 0.92,
                    'expected_snr_db': 30.0
                },
                'device_characteristics': {
                    'sensitivity': 0.8,
                    'noise_floor': -50,
                    'max_spl': 120,
                    'recommended_for': ['general_purpose', 'voice_assistant']
                }
            },
            'respeaker_2mic.yaml': {
                'metadata': {
                    'device_name': 'ReSpeaker 2-Mic HAT',
                    'device_id': 'respeaker_2mic',
                    'profile_name': 'ReSpeaker 2-Mic Optimized',
                    'created_at': datetime.now().isoformat(),
                    'version': self.DEFAULT_PROFILE_VERSION
                },
                'gain_settings': {
                    'input_volume': 0.5,
                    'agc_enabled': False,  # Has hardware processing
                    'agc_max_gain': 1.0,
                    'agc_min_gain': 0.5,
                    'agc_target_rms': 0.3,
                    'agc_attack_time': 0.5,
                    'agc_release_time': 2.0,
                    'wake_word_gain': 0.8
                },
                'filter_settings': {
                    'highpass_filter_enabled': False,  # Hardware handles this
                    'highpass_filter_cutoff': 0.0
                },
                'performance_metrics': {
                    'estimated_cumulative_gain': 0.4,
                    'expected_wake_word_accuracy': 0.95,
                    'expected_snr_db': 35.0
                },
                'device_characteristics': {
                    'sensitivity': 1.2,
                    'noise_floor': -45,
                    'max_spl': 110,
                    'has_hardware_processing': True,
                    'recommended_for': ['raspberry_pi', 'wake_word_detection']
                }
            },
            'jabra_410.yaml': {
                'metadata': {
                    'device_name': 'Jabra Speak 410',
                    'device_id': 'jabra_410',
                    'profile_name': 'Jabra Conference Speaker',
                    'created_at': datetime.now().isoformat(),
                    'version': self.DEFAULT_PROFILE_VERSION
                },
                'gain_settings': {
                    'input_volume': 0.7,
                    'agc_enabled': False,  # Has internal AGC
                    'agc_max_gain': 1.0,
                    'agc_min_gain': 0.5,
                    'agc_target_rms': 0.3,
                    'agc_attack_time': 0.5,
                    'agc_release_time': 2.0,
                    'wake_word_gain': 1.2
                },
                'filter_settings': {
                    'highpass_filter_enabled': False,  # Device handles filtering
                    'highpass_filter_cutoff': 0.0
                },
                'performance_metrics': {
                    'estimated_cumulative_gain': 0.84,
                    'expected_wake_word_accuracy': 0.93,
                    'expected_snr_db': 40.0
                },
                'device_characteristics': {
                    'sensitivity': 1.0,
                    'noise_floor': -55,
                    'max_spl': 115,
                    'has_internal_agc': True,
                    'has_echo_cancellation': True,
                    'recommended_for': ['conference_calls', 'multi_person']
                }
            },
            'custom_template.yaml': {
                'metadata': {
                    'device_name': 'Custom Device Template',
                    'device_id': 'custom',
                    'profile_name': 'Custom Profile Template',
                    'created_at': datetime.now().isoformat(),
                    'version': self.DEFAULT_PROFILE_VERSION,
                    'notes': 'Template for creating custom device profiles'
                },
                'gain_settings': {
                    'input_volume': 1.0,
                    'agc_enabled': True,
                    'agc_max_gain': 2.0,
                    'agc_min_gain': 0.5,
                    'agc_target_rms': 0.3,
                    'agc_attack_time': 0.5,
                    'agc_release_time': 2.0,
                    'wake_word_gain': 1.0
                },
                'filter_settings': {
                    'highpass_filter_enabled': True,
                    'highpass_filter_cutoff': 80.0
                },
                'performance_metrics': {
                    'estimated_cumulative_gain': 2.0,
                    'expected_wake_word_accuracy': 0.90,
                    'expected_snr_db': 25.0
                },
                'device_characteristics': {
                    'sensitivity': 1.0,
                    'noise_floor': -45,
                    'max_spl': 115,
                    'notes': 'Adjust these values based on your device'
                }
            }
        }
        
        # Create default profiles if they don't exist
        for filename, profile_data in default_profiles.items():
            profile_path = self.profiles_dir / filename
            if not profile_path.exists():
                with open(profile_path, 'w') as f:
                    yaml.dump(profile_data, f, default_flow_style=False)
    
    def create_profile(self, 
                      device_name: str,
                      device_id: str,
                      gain_settings: Dict[str, float],
                      performance_metrics: Dict[str, float],
                      device_characteristics: Optional[Dict] = None) -> AudioProfile:
        """
        Create a new audio profile
        
        Args:
            device_name: Name of the device
            device_id: Unique device identifier
            gain_settings: Optimal gain settings
            performance_metrics: Expected performance metrics
            device_characteristics: Device-specific characteristics
            
        Returns:
            Created AudioProfile object
        """
        profile = AudioProfile(
            device_name=device_name,
            device_id=device_id,
            profile_name=f"{device_name} Profile",
            created_at=datetime.now().isoformat(),
            version=self.DEFAULT_PROFILE_VERSION,
            
            # Gain settings with defaults
            input_volume=gain_settings.get('input_volume', 1.0),
            agc_enabled=gain_settings.get('agc_enabled', True),
            agc_max_gain=gain_settings.get('agc_max_gain', 2.0),
            agc_min_gain=gain_settings.get('agc_min_gain', 0.5),
            agc_target_rms=gain_settings.get('agc_target_rms', 0.3),
            agc_attack_time=gain_settings.get('agc_attack_time', 0.5),
            agc_release_time=gain_settings.get('agc_release_time', 2.0),
            wake_word_gain=gain_settings.get('wake_word_gain', 1.0),
            
            # Filter settings
            highpass_filter_enabled=gain_settings.get('highpass_filter_enabled', True),
            highpass_filter_cutoff=gain_settings.get('highpass_filter_cutoff', 80.0),
            
            # Performance metrics
            estimated_cumulative_gain=performance_metrics.get('cumulative_gain', 2.0),
            expected_wake_word_accuracy=performance_metrics.get('wake_word_accuracy', 0.9),
            expected_snr_db=performance_metrics.get('snr_db', 25.0),
            
            # Device characteristics
            device_characteristics=device_characteristics or {}
        )
        
        return profile
    
    def save_profile(self, profile: AudioProfile, filename: Optional[str] = None) -> str:
        """
        Save profile to file
        
        Args:
            profile: AudioProfile to save
            filename: Optional custom filename
            
        Returns:
            Path to saved profile
        """
        if not filename:
            # Generate filename from device name and ID
            safe_name = profile.device_name.lower().replace(' ', '_')
            safe_name = ''.join(c if c.isalnum() or c == '_' else '' for c in safe_name)
            filename = f"{safe_name}_{profile.device_id[:8]}.yaml"
        
        filepath = self.profiles_dir / filename
        
        # Convert profile to dictionary
        profile_dict = {
            'metadata': {
                'device_name': profile.device_name,
                'device_id': profile.device_id,
                'profile_name': profile.profile_name,
                'created_at': profile.created_at,
                'version': profile.version
            },
            'gain_settings': {
                'input_volume': profile.input_volume,
                'agc_enabled': profile.agc_enabled,
                'agc_max_gain': profile.agc_max_gain,
                'agc_min_gain': profile.agc_min_gain,
                'agc_target_rms': profile.agc_target_rms,
                'agc_attack_time': profile.agc_attack_time,
                'agc_release_time': profile.agc_release_time,
                'wake_word_gain': profile.wake_word_gain
            },
            'filter_settings': {
                'highpass_filter_enabled': profile.highpass_filter_enabled,
                'highpass_filter_cutoff': profile.highpass_filter_cutoff
            },
            'performance_metrics': {
                'estimated_cumulative_gain': profile.estimated_cumulative_gain,
                'expected_wake_word_accuracy': profile.expected_wake_word_accuracy,
                'expected_snr_db': profile.expected_snr_db
            },
            'device_characteristics': profile.device_characteristics
        }
        
        if profile.validation_scores:
            profile_dict['validation_scores'] = profile.validation_scores
        
        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(profile_dict, f, default_flow_style=False)
        
        return str(filepath)
    
    def load_profile(self, profile_path: str) -> Optional[AudioProfile]:
        """
        Load profile from file
        
        Args:
            profile_path: Path to profile file
            
        Returns:
            AudioProfile object or None if not found
        """
        filepath = Path(profile_path)
        if not filepath.exists():
            # Try in profiles directory
            filepath = self.profiles_dir / profile_path
            if not filepath.exists():
                return None
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            # Create AudioProfile from data
            profile = AudioProfile(
                device_name=data['metadata']['device_name'],
                device_id=data['metadata']['device_id'],
                profile_name=data['metadata']['profile_name'],
                created_at=data['metadata']['created_at'],
                version=data['metadata']['version'],
                
                # Gain settings
                input_volume=data['gain_settings']['input_volume'],
                agc_enabled=data['gain_settings']['agc_enabled'],
                agc_max_gain=data['gain_settings']['agc_max_gain'],
                agc_min_gain=data['gain_settings']['agc_min_gain'],
                agc_target_rms=data['gain_settings']['agc_target_rms'],
                agc_attack_time=data['gain_settings']['agc_attack_time'],
                agc_release_time=data['gain_settings']['agc_release_time'],
                wake_word_gain=data['gain_settings']['wake_word_gain'],
                
                # Filter settings
                highpass_filter_enabled=data['filter_settings']['highpass_filter_enabled'],
                highpass_filter_cutoff=data['filter_settings']['highpass_filter_cutoff'],
                
                # Performance metrics
                estimated_cumulative_gain=data['performance_metrics']['estimated_cumulative_gain'],
                expected_wake_word_accuracy=data['performance_metrics']['expected_wake_word_accuracy'],
                expected_snr_db=data['performance_metrics']['expected_snr_db'],
                
                # Device characteristics
                device_characteristics=data.get('device_characteristics', {}),
                validation_scores=data.get('validation_scores')
            )
            
            return profile
            
        except Exception as e:
            print(f"Error loading profile: {e}")
            return None
    
    def list_profiles(self) -> List[Dict[str, str]]:
        """
        List all available profiles
        
        Returns:
            List of profile summaries
        """
        profiles = []
        
        for profile_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(profile_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                profiles.append({
                    'filename': profile_file.name,
                    'device_name': data['metadata']['device_name'],
                    'device_id': data['metadata']['device_id'],
                    'profile_name': data['metadata']['profile_name'],
                    'created_at': data['metadata']['created_at'],
                    'cumulative_gain': data['performance_metrics']['estimated_cumulative_gain']
                })
            except Exception:
                continue
        
        return sorted(profiles, key=lambda x: x['device_name'])
    
    def find_profile_for_device(self, device_id: str) -> Optional[AudioProfile]:
        """
        Find existing profile for a device
        
        Args:
            device_id: Device identifier
            
        Returns:
            AudioProfile or None if not found
        """
        for profile_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(profile_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if data['metadata']['device_id'] == device_id:
                    return self.load_profile(str(profile_file))
            except Exception:
                continue
        
        return None
    
    def apply_profile(self, profile: AudioProfile, config_path: str) -> bool:
        """
        Apply profile settings to config.yaml
        
        Args:
            profile: AudioProfile to apply
            config_path: Path to config.yaml
            
        Returns:
            True if successful
        """
        try:
            # Load existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Backup current config
            backup_path = f"{config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(config_path, backup_path)
            
            # Update config with profile settings
            if 'audio' not in config:
                config['audio'] = {}
            if 'wake_word' not in config:
                config['wake_word'] = {}
            
            # Apply audio settings
            config['audio']['input_volume'] = profile.input_volume
            config['audio']['agc_enabled'] = profile.agc_enabled
            config['audio']['agc_max_gain'] = profile.agc_max_gain
            config['audio']['agc_min_gain'] = profile.agc_min_gain
            config['audio']['agc_target_rms'] = profile.agc_target_rms
            config['audio']['agc_attack_time'] = profile.agc_attack_time
            config['audio']['agc_release_time'] = profile.agc_release_time
            
            # Apply wake word settings
            config['wake_word']['audio_gain'] = profile.wake_word_gain
            config['wake_word']['highpass_filter_enabled'] = profile.highpass_filter_enabled
            config['wake_word']['highpass_filter_cutoff'] = profile.highpass_filter_cutoff
            
            # Add profile metadata as comment
            config['_audio_profile'] = {
                'name': profile.profile_name,
                'device': profile.device_name,
                'applied_at': datetime.now().isoformat(),
                'expected_accuracy': profile.expected_wake_word_accuracy
            }
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"Profile applied successfully. Backup saved to: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error applying profile: {e}")
            return False
    
    def validate_profile(self, profile: AudioProfile) -> Dict[str, Any]:
        """
        Validate profile settings
        
        Args:
            profile: AudioProfile to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check cumulative gain
        if profile.estimated_cumulative_gain > 10:
            issues.append(f"Cumulative gain too high: {profile.estimated_cumulative_gain}x (max recommended: 10x)")
        elif profile.estimated_cumulative_gain > 8:
            warnings.append(f"High cumulative gain: {profile.estimated_cumulative_gain}x")
        
        # Check individual gain values
        if profile.input_volume > 3.0:
            issues.append(f"Input volume too high: {profile.input_volume} (max recommended: 3.0)")
        
        if profile.wake_word_gain > 3.0:
            issues.append(f"Wake word gain too high: {profile.wake_word_gain} (max recommended: 3.0)")
        
        # Check AGC settings
        if profile.agc_enabled and profile.agc_max_gain > 5.0:
            warnings.append(f"AGC max gain very high: {profile.agc_max_gain}")
        
        if profile.agc_target_rms > 0.5:
            warnings.append(f"AGC target RMS high: {profile.agc_target_rms} (may cause clipping)")
        
        # Check filter settings
        if profile.highpass_filter_enabled and profile.highpass_filter_cutoff > 200:
            warnings.append(f"High-pass filter cutoff very high: {profile.highpass_filter_cutoff} Hz")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'cumulative_gain': profile.estimated_cumulative_gain
        }


if __name__ == "__main__":
    # Test profile manager
    manager = ProfileManager()
    
    print("Audio Profile Manager Test")
    print("="*60)
    
    # List available profiles
    print("\nAvailable profiles:")
    profiles = manager.list_profiles()
    for i, profile in enumerate(profiles, 1):
        print(f"{i}. {profile['device_name']} ({profile['filename']})")
        print(f"   Cumulative gain: {profile['cumulative_gain']}x")
    
    # Load and validate a profile
    if profiles:
        test_profile = manager.load_profile(profiles[0]['filename'])
        if test_profile:
            print(f"\nLoaded profile: {test_profile.profile_name}")
            
            # Validate
            validation = manager.validate_profile(test_profile)
            print(f"\nValidation results:")
            print(f"Valid: {validation['valid']}")
            if validation['issues']:
                print("Issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
    
    # Create a new profile
    print("\nCreating new test profile...")
    new_profile = manager.create_profile(
        device_name="Test Microphone",
        device_id="test_mic_123",
        gain_settings={
            'input_volume': 1.2,
            'agc_enabled': True,
            'agc_max_gain': 2.5,
            'wake_word_gain': 1.0
        },
        performance_metrics={
            'cumulative_gain': 3.0,
            'wake_word_accuracy': 0.94,
            'snr_db': 32.0
        }
    )
    
    # Save the profile
    saved_path = manager.save_profile(new_profile)
    print(f"Profile saved to: {saved_path}")
    
    print("\nProfile manager test complete!")