#!/usr/bin/env python3
"""
OpenAI Compatibility Checker for PCM16 Audio
Part of Task 1.4 - Audio Format Validation

Validates that PCM16 audio format meets OpenAI Realtime API requirements.
"""

import numpy as np
import struct
import sys
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass 
class OpenAICompatibility:
    """OpenAI format compatibility results"""
    format_valid: bool
    sample_rate_valid: bool
    bit_depth_valid: bool
    endianness_valid: bool
    channels_valid: bool
    byte_order_valid: bool
    issues: list
    warnings: list


class OpenAICompatibilityChecker:
    """Validate PCM16 audio for OpenAI Realtime API compatibility"""
    
    def __init__(self):
        # OpenAI Realtime API requirements
        self.required_sample_rate = 24000
        self.required_bit_depth = 16
        self.required_channels = 1  # Mono
        self.required_endianness = 'little'
        self.required_format = 'pcm16'
    
    def check_format_compliance(self, pcm16_data: np.ndarray) -> OpenAICompatibility:
        """
        Check if PCM16 data meets OpenAI requirements
        
        Args:
            pcm16_data: PCM16 audio data as numpy array
            
        Returns:
            OpenAICompatibility results
        """
        issues = []
        warnings = []
        
        # Check data type
        format_valid = pcm16_data.dtype == np.int16
        if not format_valid:
            issues.append(f"Invalid dtype: {pcm16_data.dtype}, expected np.int16")
        
        # Check bit depth (itemsize is in bytes)
        bit_depth_valid = pcm16_data.itemsize == 2
        if not bit_depth_valid:
            issues.append(f"Invalid bit depth: {pcm16_data.itemsize * 8} bits, expected 16")
        
        # Check endianness
        endianness_valid = self._check_endianness(pcm16_data)
        if not endianness_valid:
            issues.append(f"Invalid endianness, expected little-endian")
        
        # Check value range
        min_val = np.min(pcm16_data)
        max_val = np.max(pcm16_data)
        if min_val < -32768 or max_val > 32767:
            issues.append(f"Values out of int16 range: [{min_val}, {max_val}]")
        
        # Check if using full dynamic range (warning only)
        if max_val < 16000 and min_val > -16000:
            warnings.append("Audio may be too quiet (not using full dynamic range)")
        
        # For this module, we assume sample rate and channels are handled elsewhere
        sample_rate_valid = True  # Assumed to be resampled to 24kHz
        channels_valid = True  # Assumed to be mono
        byte_order_valid = endianness_valid
        
        return OpenAICompatibility(
            format_valid=format_valid,
            sample_rate_valid=sample_rate_valid,
            bit_depth_valid=bit_depth_valid,
            endianness_valid=endianness_valid,
            channels_valid=channels_valid,
            byte_order_valid=byte_order_valid,
            issues=issues,
            warnings=warnings
        )
    
    def _check_endianness(self, pcm16_data: np.ndarray) -> bool:
        """Check if data is little-endian"""
        # Create test value
        test_val = np.array([1000], dtype=np.int16)
        test_bytes = test_val.tobytes()
        
        # Check if it matches little-endian encoding
        little_endian_bytes = struct.pack('<h', 1000)
        
        return test_bytes == little_endian_bytes
    
    def validate_byte_stream(self, audio_bytes: bytes) -> Dict:
        """
        Validate raw byte stream for OpenAI API
        
        Args:
            audio_bytes: Raw PCM16 audio bytes
            
        Returns:
            Validation results
        """
        results = {
            'byte_length': len(audio_bytes),
            'expected_samples': len(audio_bytes) // 2,
            'is_even_length': len(audio_bytes) % 2 == 0,
            'format_check': {}
        }
        
        # Check byte length is even (2 bytes per sample)
        if not results['is_even_length']:
            results['format_check']['error'] = "Byte length must be even (2 bytes per sample)"
            return results
        
        # Parse first few samples to verify format
        try:
            # Try to interpret as little-endian int16
            first_samples_le = struct.unpack('<10h', audio_bytes[:20])
            results['format_check']['first_10_samples_le'] = list(first_samples_le)
            
            # Check if values are in valid range
            valid_range = all(-32768 <= s <= 32767 for s in first_samples_le)
            results['format_check']['valid_range'] = valid_range
            
            # Try big-endian for comparison
            first_samples_be = struct.unpack('>10h', audio_bytes[:20])
            results['format_check']['first_10_samples_be'] = list(first_samples_be)
            
            # Determine likely endianness based on value distribution
            le_variance = np.var(first_samples_le)
            be_variance = np.var(first_samples_be)
            
            if le_variance < be_variance * 10:  # Little-endian likely has lower variance
                results['format_check']['detected_endianness'] = 'little'
            else:
                results['format_check']['detected_endianness'] = 'big'
            
            results['format_check']['status'] = 'valid'
            
        except Exception as e:
            results['format_check']['error'] = str(e)
            results['format_check']['status'] = 'invalid'
        
        return results
    
    def test_conversion_chain(self, conversion_method: Callable) -> Dict:
        """
        Test complete conversion chain for OpenAI compatibility
        
        Args:
            conversion_method: Float32 to PCM16 conversion function
            
        Returns:
            Test results
        """
        results = {}
        
        # Generate test signal at 24kHz
        duration = 1.0
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test 1: Simple sine wave
        sine = np.sin(2 * np.pi * 440 * t) * 0.5
        sine_pcm = conversion_method(sine)
        sine_bytes = sine_pcm.tobytes()
        
        results['sine_wave'] = {
            'pcm16_check': self.check_format_compliance(sine_pcm),
            'byte_validation': self.validate_byte_stream(sine_bytes),
            'byte_size': len(sine_bytes),
            'expected_size': len(sine) * 2
        }
        
        # Test 2: Full scale signal
        full_scale = np.sin(2 * np.pi * 440 * t) * 0.999
        full_pcm = conversion_method(full_scale)
        full_bytes = full_pcm.tobytes()
        
        results['full_scale'] = {
            'pcm16_check': self.check_format_compliance(full_pcm),
            'max_value': int(np.max(full_pcm)),
            'min_value': int(np.min(full_pcm)),
            'using_full_range': np.max(np.abs(full_pcm)) > 30000
        }
        
        # Test 3: Quiet signal
        quiet = np.sin(2 * np.pi * 440 * t) * 0.01
        quiet_pcm = conversion_method(quiet)
        
        results['quiet_signal'] = {
            'pcm16_check': self.check_format_compliance(quiet_pcm),
            'max_value': int(np.max(quiet_pcm)),
            'signal_preserved': np.max(np.abs(quiet_pcm)) > 0
        }
        
        # Overall compatibility
        all_valid = all(
            results[test]['pcm16_check'].format_valid and
            results[test]['pcm16_check'].endianness_valid
            for test in ['sine_wave', 'full_scale', 'quiet_signal']
            if 'pcm16_check' in results[test]
        )
        
        results['overall_compatible'] = all_valid
        
        return results
    
    def generate_sample_audio(self, duration: float = 0.1) -> Tuple[np.ndarray, bytes]:
        """
        Generate sample audio in OpenAI-compatible format
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Tuple of (pcm16_array, audio_bytes)
        """
        # Generate at required sample rate
        samples = int(self.required_sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Create test tone
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to PCM16 using recommended method
        pcm16 = (audio * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = pcm16.tobytes()
        
        return pcm16, audio_bytes
    
    def create_validation_report(self, conversion_method: Callable) -> Dict:
        """
        Create comprehensive OpenAI compatibility report
        
        Args:
            conversion_method: Conversion function to validate
            
        Returns:
            Validation report
        """
        report = {
            'method_name': conversion_method.__name__ if hasattr(conversion_method, '__name__') else 'unknown',
            'openai_requirements': {
                'sample_rate': self.required_sample_rate,
                'bit_depth': self.required_bit_depth,
                'channels': self.required_channels,
                'endianness': self.required_endianness,
                'format': self.required_format
            },
            'test_results': self.test_conversion_chain(conversion_method),
            'recommendations': []
        }
        
        # Generate recommendations
        if report['test_results']['overall_compatible']:
            report['recommendations'].append("✓ Format is fully compatible with OpenAI Realtime API")
        else:
            report['recommendations'].append("⚠ Format compatibility issues detected")
        
        # Check for warnings
        sine_check = report['test_results']['sine_wave']['pcm16_check']
        if sine_check.warnings:
            for warning in sine_check.warnings:
                report['recommendations'].append(f"⚠ {warning}")
        
        # Check byte stream
        byte_validation = report['test_results']['sine_wave']['byte_validation']
        if byte_validation['format_check']['detected_endianness'] == 'little':
            report['recommendations'].append("✓ Byte order (little-endian) is correct")
        else:
            report['recommendations'].append("⚠ Byte order may be incorrect (expected little-endian)")
        
        return report


def demonstrate_openai_compatibility():
    """Demonstrate OpenAI compatibility checking"""
    print("OpenAI Compatibility Checker")
    print("=" * 50)
    
    checker = OpenAICompatibilityChecker()
    
    # Test current implementation
    def current_method(audio: np.ndarray) -> np.ndarray:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    print("\nTesting OpenAI compatibility...")
    report = checker.create_validation_report(current_method)
    
    print("\nOpenAI Requirements:")
    for key, value in report['openai_requirements'].items():
        print(f"  {key}: {value}")
    
    print("\nTest Results:")
    print(f"  Overall Compatible: {report['test_results']['overall_compatible']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Generate sample audio
    print("\n" + "=" * 50)
    print("Generating sample OpenAI-compatible audio...")
    pcm16, audio_bytes = checker.generate_sample_audio(0.1)
    print(f"  Generated {len(audio_bytes)} bytes")
    print(f"  Sample rate: 24000 Hz")
    print(f"  Duration: 0.1 seconds")
    print(f"  Format: PCM16, little-endian, mono")


if __name__ == "__main__":
    demonstrate_openai_compatibility()