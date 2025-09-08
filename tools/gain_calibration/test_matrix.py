"""
Test Matrix Generator for Gain Optimization

Generates comprehensive test configurations for different microphone types
and speech volumes to find optimal gain settings.
"""
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class TestConfiguration:
    """Single test configuration"""
    device_type: str
    input_volume: float
    agc_enabled: bool
    agc_max_gain: float
    wake_word_gain: float
    volume_scenario: str
    expected_spl: float  # Sound pressure level in dB
    
    @property
    def cumulative_gain(self) -> float:
        """Calculate total gain through pipeline"""
        base_gain = self.input_volume
        if self.agc_enabled:
            # AGC can amplify up to max_gain
            base_gain *= self.agc_max_gain
        base_gain *= self.wake_word_gain
        return base_gain
    
    def to_dict(self) -> dict:
        """Convert to dictionary with additional calculated fields"""
        result = asdict(self)
        result['cumulative_gain'] = self.cumulative_gain
        result['risk_level'] = self.get_risk_level()
        return result
    
    def get_risk_level(self) -> str:
        """Assess clipping risk based on cumulative gain"""
        total_gain = self.cumulative_gain
        if total_gain > 10:
            return "high"
        elif total_gain > 5:
            return "medium"
        else:
            return "low"


class TestMatrix:
    """
    Generate and manage test configurations for gain optimization
    """
    
    # Device profiles with typical characteristics
    DEVICE_PROFILES = {
        'usb_generic': {
            'name': 'Generic USB Microphone',
            'sensitivity': 0.8,  # Relative sensitivity
            'noise_floor': -50,  # dB
            'max_spl': 120,      # Maximum SPL before distortion
            'recommended_input_volume': 1.0,
        },
        'respeaker_2mic': {
            'name': 'ReSpeaker 2-Mic HAT',
            'sensitivity': 1.2,  # Higher sensitivity
            'noise_floor': -45,
            'max_spl': 110,
            'recommended_input_volume': 0.5,
        },
        'jabra_410': {
            'name': 'Jabra Speak 410',
            'sensitivity': 1.0,
            'noise_floor': -55,
            'max_spl': 115,
            'recommended_input_volume': 0.7,
            'has_internal_agc': True,
        },
        'blue_yeti': {
            'name': 'Blue Yeti USB Microphone',
            'sensitivity': 1.1,
            'noise_floor': -48,
            'max_spl': 120,
            'recommended_input_volume': 0.8,
        },
        'webcam_mic': {
            'name': 'Generic Webcam Microphone',
            'sensitivity': 0.6,  # Lower sensitivity
            'noise_floor': -40,
            'max_spl': 105,
            'recommended_input_volume': 1.5,
        }
    }
    
    # Volume scenarios for testing
    VOLUME_SCENARIOS = {
        'quiet': {
            'name': 'Quiet Speech',
            'spl_range': (55, 60),
            'typical_spl': 57.5,
            'description': 'Whispering or very soft speech'
        },
        'normal': {
            'name': 'Normal Conversation',
            'spl_range': (60, 65),
            'typical_spl': 62.5,
            'description': 'Regular conversation level'
        },
        'loud': {
            'name': 'Loud Speech',
            'spl_range': (70, 75),
            'typical_spl': 72.5,
            'description': 'Raised voice or enthusiastic speech'
        }
    }
    
    # Gain stage parameters to test
    GAIN_PARAMETERS = {
        'input_volume': [0.3, 0.5, 1.0, 2.0, 3.0],
        'agc_enabled': [True, False],
        'agc_max_gain': [1.0, 2.0, 3.0],  # Only relevant when AGC is enabled
        'wake_word_gain': [0.5, 1.0, 1.5, 2.0, 3.0]
    }
    
    def __init__(self, custom_parameters: Optional[Dict] = None):
        """
        Initialize test matrix generator
        
        Args:
            custom_parameters: Optional custom gain parameters to override defaults
        """
        self.device_profiles = self.DEVICE_PROFILES.copy()
        self.volume_scenarios = self.VOLUME_SCENARIOS.copy()
        self.gain_parameters = self.GAIN_PARAMETERS.copy()
        
        if custom_parameters:
            self.gain_parameters.update(custom_parameters)
    
    def generate_matrix(self, 
                       device_types: Optional[List[str]] = None,
                       volume_levels: Optional[List[str]] = None,
                       optimize_for_speed: bool = False) -> List[TestConfiguration]:
        """
        Generate complete test matrix
        
        Args:
            device_types: Specific devices to test (None = all)
            volume_levels: Specific volume levels to test (None = all)
            optimize_for_speed: If True, use reduced parameter set for faster testing
            
        Returns:
            List of test configurations
        """
        # Select devices and volumes
        devices = device_types or list(self.device_profiles.keys())
        volumes = volume_levels or list(self.volume_scenarios.keys())
        
        # Adjust parameters for speed optimization
        if optimize_for_speed:
            params = {
                'input_volume': [0.5, 1.0, 2.0],
                'agc_enabled': [True, False],
                'agc_max_gain': [2.0],
                'wake_word_gain': [1.0, 2.0]
            }
        else:
            params = self.gain_parameters
        
        configurations = []
        
        for device in devices:
            for volume in volumes:
                for input_vol in params['input_volume']:
                    for agc_enabled in params['agc_enabled']:
                        if agc_enabled:
                            # Test different AGC max gains
                            for agc_gain in params['agc_max_gain']:
                                for ww_gain in params['wake_word_gain']:
                                    config = TestConfiguration(
                                        device_type=device,
                                        input_volume=input_vol,
                                        agc_enabled=True,
                                        agc_max_gain=agc_gain,
                                        wake_word_gain=ww_gain,
                                        volume_scenario=volume,
                                        expected_spl=self.volume_scenarios[volume]['typical_spl']
                                    )
                                    configurations.append(config)
                        else:
                            # AGC disabled, use default max gain
                            for ww_gain in params['wake_word_gain']:
                                config = TestConfiguration(
                                    device_type=device,
                                    input_volume=input_vol,
                                    agc_enabled=False,
                                    agc_max_gain=1.0,
                                    wake_word_gain=ww_gain,
                                    volume_scenario=volume,
                                    expected_spl=self.volume_scenarios[volume]['typical_spl']
                                )
                                configurations.append(config)
        
        return configurations
    
    def generate_quick_test(self, device_type: str) -> List[TestConfiguration]:
        """
        Generate quick test set for a specific device
        
        Args:
            device_type: Device to test
            
        Returns:
            Reduced set of test configurations
        """
        if device_type not in self.device_profiles:
            raise ValueError(f"Unknown device type: {device_type}")
        
        profile = self.device_profiles[device_type]
        
        # Create targeted test based on device characteristics
        configs = []
        
        # Test recommended settings
        base_volume = profile['recommended_input_volume']
        
        # Test with and without AGC at normal volume
        for agc_enabled in [True, False]:
            config = TestConfiguration(
                device_type=device_type,
                input_volume=base_volume,
                agc_enabled=agc_enabled,
                agc_max_gain=2.0 if agc_enabled else 1.0,
                wake_word_gain=1.0,
                volume_scenario='normal',
                expected_spl=62.5
            )
            configs.append(config)
        
        # Test edge cases
        # Low input with high wake word gain
        configs.append(TestConfiguration(
            device_type=device_type,
            input_volume=base_volume * 0.5,
            agc_enabled=True,
            agc_max_gain=3.0,
            wake_word_gain=2.0,
            volume_scenario='quiet',
            expected_spl=57.5
        ))
        
        # High input with low wake word gain
        configs.append(TestConfiguration(
            device_type=device_type,
            input_volume=base_volume * 1.5,
            agc_enabled=False,
            agc_max_gain=1.0,
            wake_word_gain=0.5,
            volume_scenario='loud',
            expected_spl=72.5
        ))
        
        return configs
    
    def filter_safe_configurations(self, 
                                  configurations: List[TestConfiguration],
                                  max_cumulative_gain: float = 10.0) -> List[TestConfiguration]:
        """
        Filter configurations to only include safe gain levels
        
        Args:
            configurations: List of test configurations
            max_cumulative_gain: Maximum allowed cumulative gain
            
        Returns:
            Filtered list of safe configurations
        """
        safe_configs = []
        for config in configurations:
            if config.cumulative_gain <= max_cumulative_gain:
                safe_configs.append(config)
        
        return safe_configs
    
    def get_device_optimal_range(self, device_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Get optimal gain ranges for a specific device
        
        Args:
            device_type: Device type to analyze
            
        Returns:
            Dictionary with recommended gain ranges
        """
        if device_type not in self.device_profiles:
            raise ValueError(f"Unknown device type: {device_type}")
        
        profile = self.device_profiles[device_type]
        sensitivity = profile['sensitivity']
        
        # Calculate optimal ranges based on device sensitivity
        optimal = {
            'input_volume': (
                profile['recommended_input_volume'] * 0.7,
                profile['recommended_input_volume'] * 1.3
            ),
            'agc_max_gain': (1.5, 3.0) if sensitivity < 1.0 else (1.0, 2.0),
            'wake_word_gain': (0.8, 1.5) if sensitivity > 1.0 else (1.0, 2.0)
        }
        
        # Adjust for devices with internal AGC
        if profile.get('has_internal_agc'):
            optimal['agc_max_gain'] = (1.0, 1.5)  # Reduce external AGC
        
        return optimal
    
    def prioritize_configurations(self, 
                                configurations: List[TestConfiguration]) -> List[TestConfiguration]:
        """
        Sort configurations by testing priority
        
        Args:
            configurations: List of test configurations
            
        Returns:
            Sorted list with highest priority first
        """
        def priority_score(config: TestConfiguration) -> float:
            score = 0
            
            # Prioritize normal volume scenarios
            if config.volume_scenario == 'normal':
                score += 10
            
            # Prioritize moderate gain levels
            if 3 < config.cumulative_gain < 7:
                score += 8
            
            # Prioritize common devices
            if config.device_type in ['usb_generic', 'respeaker_2mic']:
                score += 5
            
            # Penalize extreme gains
            if config.cumulative_gain > 15:
                score -= 10
            
            return score
        
        return sorted(configurations, key=priority_score, reverse=True)
    
    def export_matrix(self, configurations: List[TestConfiguration], 
                     output_file: str, format: str = 'json'):
        """
        Export test matrix to file
        
        Args:
            configurations: List of test configurations
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            data = [config.to_dict() for config in configurations]
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            import csv
            if configurations:
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=configurations[0].to_dict().keys())
                    writer.writeheader()
                    for config in configurations:
                        writer.writerow(config.to_dict())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self, configurations: List[TestConfiguration]) -> Dict:
        """
        Get statistics about the test matrix
        
        Args:
            configurations: List of test configurations
            
        Returns:
            Dictionary with matrix statistics
        """
        if not configurations:
            return {}
        
        cumulative_gains = [c.cumulative_gain for c in configurations]
        risk_levels = [c.get_risk_level() for c in configurations]
        
        stats = {
            'total_configurations': len(configurations),
            'unique_devices': len(set(c.device_type for c in configurations)),
            'unique_volumes': len(set(c.volume_scenario for c in configurations)),
            'gain_range': (min(cumulative_gains), max(cumulative_gains)),
            'average_gain': np.mean(cumulative_gains),
            'median_gain': np.median(cumulative_gains),
            'risk_distribution': {
                'low': risk_levels.count('low'),
                'medium': risk_levels.count('medium'),
                'high': risk_levels.count('high')
            },
            'agc_enabled_count': sum(1 for c in configurations if c.agc_enabled),
            'agc_disabled_count': sum(1 for c in configurations if not c.agc_enabled)
        }
        
        return stats


if __name__ == "__main__":
    # Example usage and testing
    matrix = TestMatrix()
    
    # Generate full matrix for specific devices
    print("Generating test matrix for common devices...")
    configs = matrix.generate_matrix(
        device_types=['usb_generic', 'respeaker_2mic', 'jabra_410'],
        optimize_for_speed=True
    )
    
    # Filter for safe configurations
    safe_configs = matrix.filter_safe_configurations(configs, max_cumulative_gain=10.0)
    
    # Prioritize configurations
    prioritized = matrix.prioritize_configurations(safe_configs)
    
    # Get statistics
    stats = matrix.get_statistics(prioritized)
    
    print(f"\nTest Matrix Statistics:")
    print(f"Total configurations: {stats['total_configurations']}")
    print(f"Gain range: {stats['gain_range'][0]:.1f} - {stats['gain_range'][1]:.1f}")
    print(f"Risk distribution: {stats['risk_distribution']}")
    
    # Export top 10 configurations
    print(f"\nTop 10 priority configurations:")
    for i, config in enumerate(prioritized[:10], 1):
        print(f"{i}. Device: {config.device_type}, "
              f"Gain: {config.cumulative_gain:.1f}x, "
              f"Risk: {config.get_risk_level()}, "
              f"Volume: {config.volume_scenario}")
    
    # Export matrix
    matrix.export_matrix(prioritized, '/tmp/test_matrix.json', format='json')
    print(f"\nFull matrix exported to /tmp/test_matrix.json")