"""
Dependency validation utilities for audio libraries
"""
import sys
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger


class DependencyValidationError(Exception):
    """Exception raised when dependencies are missing or invalid"""
    pass


def validate_audio_dependencies() -> Dict[str, Optional[str]]:
    """
    Validate that all required audio libraries are available and functional.
    
    Returns:
        Dict mapping library names to version strings (None if not available)
        
    Raises:
        DependencyValidationError: If critical dependencies are missing
    """
    logger = get_logger("DependencyValidator")
    results = {}
    errors = []
    
    # Check numpy
    try:
        import numpy as np
        results["numpy"] = np.__version__
        logger.debug(f"numpy {np.__version__} available")
    except ImportError as e:
        results["numpy"] = None
        errors.append(f"numpy: {e}")
    
    # Check scipy
    try:
        import scipy
        results["scipy"] = scipy.__version__
        logger.debug(f"scipy {scipy.__version__} available")
    except ImportError as e:
        results["scipy"] = None
        errors.append(f"scipy: {e}")
    
    # Check sounddevice
    try:
        import sounddevice as sd
        results["sounddevice"] = sd.__version__
        logger.debug(f"sounddevice {sd.__version__} available")
        
        # Test basic functionality
        try:
            devices = sd.query_devices()
            logger.debug(f"Found {len(devices)} audio devices")
        except Exception as e:
            logger.warning(f"sounddevice available but device query failed: {e}")
            
    except ImportError as e:
        results["sounddevice"] = None
        errors.append(f"sounddevice: {e}")
    
    # Check if any critical dependencies are missing
    if errors:
        error_msg = "Missing required audio dependencies:\n" + "\n".join(errors)
        error_msg += "\n\nTo install: pip install numpy scipy sounddevice"
        raise DependencyValidationError(error_msg)
    
    return results


def validate_audio_device_access() -> Tuple[List[dict], List[dict]]:
    """
    Validate that audio devices are accessible.
    
    Returns:
        Tuple of (input_devices, output_devices)
        
    Raises:
        DependencyValidationError: If no audio devices are available
    """
    logger = get_logger("DependencyValidator")
    
    try:
        import sounddevice as sd
        
        # Get all devices
        devices = sd.query_devices()
        
        # Separate input and output devices
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                "index": i,
                "name": device["name"],
                "channels": device.get("max_input_channels", 0),
                "sample_rate": device.get("default_samplerate", 44100)
            }
            
            if device.get("max_input_channels", 0) > 0:
                input_devices.append(device_info)
            if device.get("max_output_channels", 0) > 0:
                output_devices.append(device_info)
        
        # Check if we have at least one input device
        if not input_devices:
            raise DependencyValidationError(
                "No audio input devices found. Please check microphone connection and permissions."
            )
        
        # Check if we have at least one output device
        if not output_devices:
            logger.warning("No audio output devices found. Audio playback will not be available.")
        
        logger.info(f"Found {len(input_devices)} input devices and {len(output_devices)} output devices")
        
        return input_devices, output_devices
        
    except ImportError:
        raise DependencyValidationError("sounddevice not available - cannot validate audio devices")
    except Exception as e:
        raise DependencyValidationError(f"Audio device validation failed: {e}")


def test_microphone_recording(device_index: Optional[int] = None, duration: float = 1.0) -> bool:
    """
    Test microphone recording functionality.
    
    Args:
        device_index: Audio device index (None for default)
        duration: Test duration in seconds
        
    Returns:
        True if microphone recording works, False otherwise
    """
    logger = get_logger("DependencyValidator")
    
    try:
        import numpy as np
        import sounddevice as sd
        
        # Record for the specified duration
        logger.debug(f"Testing microphone recording for {duration} seconds...")
        
        recording = sd.rec(
            frames=int(44100 * duration),
            samplerate=44100,
            channels=1,
            device=device_index,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        # Check if we got actual audio data
        if recording is not None and len(recording) > 0:
            max_amplitude = np.max(np.abs(recording))
            logger.debug(f"Microphone test successful, max amplitude: {max_amplitude:.4f}")
            
            if max_amplitude < 0.0001:  # Very quiet
                logger.warning("Microphone recording very quiet - may need adjustment")
                return False
            
            return True
        else:
            logger.error("Microphone recording returned no data")
            return False
            
    except ImportError as e:
        logger.error(f"Required libraries not available for microphone test: {e}")
        return False
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        return False


def print_dependency_status() -> None:
    """Print dependency status for debugging"""
    logger = get_logger("DependencyValidator")
    
    try:
        # Validate audio dependencies
        deps = validate_audio_dependencies()
        
        print("Audio Dependency Status:")
        print("=" * 40)
        for lib, version in deps.items():
            status = f"[OK] {version}" if version else "[MISSING] Missing"
            print(f"{lib:15} {status}")
        
        # Test device access
        input_devices, output_devices = validate_audio_device_access()
        
        print(f"\nAudio Devices:")
        print("=" * 40)
        print(f"Input devices: {len(input_devices)}")
        print(f"Output devices: {len(output_devices)}")
        
        if input_devices:
            print("\nInput Devices:")
            for device in input_devices[:3]:  # Show first 3
                print(f"  {device['index']}: {device['name']} ({device['channels']} ch)")
        
        # Test microphone
        print(f"\nMicrophone Test:")
        print("=" * 40)
        mic_works = test_microphone_recording(duration=0.5)
        status = "[OK] Working" if mic_works else "[FAIL] Failed"
        print(f"Recording test: {status}")
        
    except DependencyValidationError as e:
        print(f"[FAIL] Dependency validation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in dependency validation: {e}")
        print(f"[FAIL] Unexpected error: {e}")


if __name__ == "__main__":
    print_dependency_status()