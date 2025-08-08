"""
Audio diagnostics and system configuration validation utilities
"""
import subprocess
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
import sounddevice as sd

from utils.logger import get_logger


class AudioDiagnostics:
    """Diagnostic utilities for audio system configuration"""
    
    # Predefined list of common ALSA control names (allowlist)
    VALID_ALSA_CONTROLS: Set[str] = {
        'Master', 'PCM', 'Line', 'Mic', 'Capture', 'Headphone',
        'Speaker', 'Front', 'Rear', 'Center', 'Bass', 'Treble',
        'Synth', 'Wave', 'Music', 'Digital', 'Monitor', 'IEC958',
        'Microphone', 'Mic Boost', 'Input Source', 'CD', 'Video',
        'Phone', 'Aux', 'Mono', 'Stereo', 'Surround', 'LFE',
        'Side', 'Beep', 'Auto-Mute Mode', 'Loopback', 'Internal Mic',
        'External Mic', 'Dock Mic', 'Headset Mic', 'Internal Mic Boost',
        'External Mic Boost', 'Dock Mic Boost', 'Headset Mic Boost'
    }
    
    def __init__(self):
        self.logger = get_logger("AudioDiagnostics")
    
    def validate_system_audio_config(self) -> Dict[str, any]:
        """
        Comprehensive system audio configuration validation
        
        Returns:
            Dictionary with validation results and recommendations
        """
        results = {
            'devices': self._get_audio_devices(),
            'alsa_mixers': self._get_alsa_mixer_levels(),
            'usb_audio_devices': self._get_usb_audio_devices(),
            'recommendations': []
        }
        
        # Analyze results and generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _get_audio_devices(self) -> List[Dict]:
        """Get available audio devices from sounddevice"""
        try:
            devices = sd.query_devices()
            device_list = []
            
            for i, device in enumerate(devices):
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'max_input_channels': device.get('max_input_channels', 0),
                    'max_output_channels': device.get('max_output_channels', 0),
                    'default_samplerate': device.get('default_samplerate', 0),
                    'is_default_input': i == sd.default.device[0],
                    'is_default_output': i == sd.default.device[1]
                }
                device_list.append(device_info)
            
            return device_list
            
        except Exception as e:
            self.logger.error(f"Error getting audio devices: {e}")
            return []
    
    def _get_alsa_mixer_levels(self) -> Dict[str, any]:
        """Get ALSA mixer levels for audio input devices"""
        try:
            # Get list of available mixer controls
            result = subprocess.run(['amixer', 'scontrols'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {'error': 'amixer not available or failed'}
            
            mixers = {}
            
            # Parse mixer controls
            for line in result.stdout.split('\n'):
                if line.strip():
                    # Extract control name
                    match = re.search(r"'([^']+)'", line)
                    if match:
                        control_name = match.group(1)
                        
                        # Get detailed info for this control
                        mixer_info = self._get_mixer_control_info(control_name)
                        if mixer_info:
                            mixers[control_name] = mixer_info
            
            return mixers
            
        except subprocess.TimeoutExpired:
            return {'error': 'amixer command timed out'}
        except Exception as e:
            self.logger.error(f"Error getting ALSA mixer levels: {e}")
            return {'error': str(e)}
    
    def _validate_mixer_control_name(self, control_name: str) -> str:
        """
        Validate ALSA mixer control name for security.
        
        Args:
            control_name: The control name to validate
            
        Returns:
            Validated control name
            
        Raises:
            ValueError: If control name is invalid
        """
        if not control_name:
            raise ValueError("Control name cannot be empty")
        
        # First check against known valid controls
        if control_name in self.VALID_ALSA_CONTROLS:
            return control_name
        
        # For custom controls, apply strict validation
        # Allow only alphanumeric, spaces, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9\s\-_]+$'
        if not re.match(pattern, control_name):
            raise ValueError(f"Invalid control name format: contains forbidden characters")
        
        # Limit length to prevent buffer issues
        if len(control_name) > 64:
            raise ValueError(f"Control name too long: {len(control_name)} characters")
        
        # Check for suspicious patterns
        suspicious_patterns = ['..', '/', '\\', ';', '|', '&', '$', '`', '(', ')', '<', '>', '\n', '\r']
        for pattern in suspicious_patterns:
            if pattern in control_name:
                raise ValueError(f"Control name contains suspicious pattern: {pattern}")
        
        return control_name
    
    def _get_mixer_control_info(self, control_name: str) -> Optional[Dict]:
        """Get detailed information for a specific mixer control"""
        try:
            # Validate control name for security
            try:
                safe_control = self._validate_mixer_control_name(control_name)
            except ValueError as e:
                self.logger.warning(f"Invalid control name rejected: {control_name} - {e}")
                return None
            
            # Use list format for subprocess (safer than shell=True)
            result = subprocess.run(['amixer', 'sget', safe_control], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            info = {
                'name': control_name,
                'type': 'unknown',
                'values': [],
                'is_capture': False,
                'is_playback': False
            }
            
            # Parse the output
            for line in result.stdout.split('\n'):
                line = line.strip()
                
                # Check for capture/playback
                if 'Capture' in line:
                    info['is_capture'] = True
                if 'Playback' in line:
                    info['is_playback'] = True
                
                # Extract volume/gain values
                if '[' in line and ']' in line:
                    # Look for percentage values
                    percentages = re.findall(r'\[(\d+)%\]', line)
                    if percentages:
                        info['values'].extend([int(p) for p in percentages])
                    
                    # Look for on/off status
                    if '[on]' in line:
                        info['enabled'] = True
                    elif '[off]' in line:
                        info['enabled'] = False
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting mixer control info for {control_name}: {e}")
            return None
    
    def _get_usb_audio_devices(self) -> List[Dict]:
        """Get USB audio devices information"""
        try:
            # Use lsusb to get USB devices
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
            
            usb_audio_devices = []
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    # Look for audio-related USB devices
                    if any(keyword in line.lower() for keyword in ['audio', 'microphone', 'mic', 'sound', 'headset']):
                        # Extract device info
                        parts = line.split()
                        if len(parts) >= 6:
                            device_info = {
                                'bus': parts[1],
                                'device': parts[3].rstrip(':'),
                                'vendor_id': parts[5].split(':')[0],
                                'product_id': parts[5].split(':')[1],
                                'description': ' '.join(parts[6:])
                            }
                            usb_audio_devices.append(device_info)
            
            return usb_audio_devices
            
        except subprocess.TimeoutExpired:
            self.logger.error("lsusb command timed out")
            return []
        except Exception as e:
            self.logger.error(f"Error getting USB audio devices: {e}")
            return []
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Check for USB audio devices
        if not results['usb_audio_devices']:
            recommendations.append("No USB audio devices detected. Ensure USB microphone is connected.")
        
        # Check ALSA mixer levels
        if 'error' in results['alsa_mixers']:
            recommendations.append(f"ALSA mixer error: {results['alsa_mixers']['error']}")
        else:
            # Check for low capture levels
            for control_name, info in results['alsa_mixers'].items():
                if info.get('is_capture', False):
                    values = info.get('values', [])
                    if values and max(values) < 50:
                        recommendations.append(f"Low capture level for {control_name}: {max(values)}% (consider increasing)")
                    
                    if info.get('enabled', True) == False:
                        recommendations.append(f"Capture control {control_name} is disabled (consider enabling)")
        
        # Check for default input device
        input_devices = [d for d in results['devices'] if d['max_input_channels'] > 0]
        if not input_devices:
            recommendations.append("No input devices found")
        else:
            default_input = next((d for d in input_devices if d['is_default_input']), None)
            if not default_input:
                recommendations.append("No default input device set")
        
        # Check sample rates
        for device in results['devices']:
            if device['max_input_channels'] > 0 and device['default_samplerate'] not in [16000, 44100, 48000]:
                recommendations.append(f"Unusual sample rate for {device['name']}: {device['default_samplerate']}Hz")
        
        return recommendations
    
    def print_diagnostic_report(self, results: Dict) -> None:
        """Print a formatted diagnostic report"""
        print("\n" + "="*60)
        print("AUDIO SYSTEM DIAGNOSTIC REPORT")
        print("="*60)
        
        # Audio devices
        print("\nAUDIO DEVICES:")
        for device in results['devices']:
            status = []
            if device['is_default_input']:
                status.append("DEFAULT INPUT")
            if device['is_default_output']:
                status.append("DEFAULT OUTPUT")
            
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Input channels: {device['max_input_channels']}")
            print(f"      Output channels: {device['max_output_channels']}")
            print(f"      Sample rate: {device['default_samplerate']}Hz")
            if status:
                print(f"      Status: {', '.join(status)}")
            print()
        
        # USB audio devices
        print("USB AUDIO DEVICES:")
        if results['usb_audio_devices']:
            for device in results['usb_audio_devices']:
                print(f"  {device['description']}")
                print(f"    Bus: {device['bus']}, Device: {device['device']}")
                print(f"    Vendor ID: {device['vendor_id']}, Product ID: {device['product_id']}")
                print()
        else:
            print("  No USB audio devices found")
        
        # ALSA mixer levels
        print("ALSA MIXER LEVELS:")
        if 'error' in results['alsa_mixers']:
            print(f"  Error: {results['alsa_mixers']['error']}")
        else:
            for control_name, info in results['alsa_mixers'].items():
                if info.get('is_capture', False):
                    values = info.get('values', [])
                    enabled = info.get('enabled', 'unknown')
                    print(f"  {control_name}: {values}% (enabled: {enabled})")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if results['recommendations']:
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("  No specific recommendations - configuration looks good")
        
        print("\n" + "="*60)


def run_audio_diagnostics() -> None:
    """Run comprehensive audio diagnostics"""
    diagnostics = AudioDiagnostics()
    results = diagnostics.validate_system_audio_config()
    diagnostics.print_diagnostic_report(results)


if __name__ == "__main__":
    run_audio_diagnostics()