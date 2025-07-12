"""
Wake word detection module with support for multiple engines
"""
from typing import Union
from config import WakeWordConfig
from utils.logger import get_logger

# Import detector implementations
from .detector import WakeWordDetector
from .porcupine_detector import PorcupineDetector


def create_wake_word_detector(config: WakeWordConfig) -> Union[WakeWordDetector, PorcupineDetector]:
    """
    Factory method to create appropriate wake word detector based on config
    
    Args:
        config: Wake word configuration
        
    Returns:
        Wake word detector instance
        
    Raises:
        ValueError: If unknown engine specified
    """
    logger = get_logger("WakeWordFactory")
    
    # Get engine from config, default to openwakeword for backwards compatibility
    engine = getattr(config, 'engine', 'openwakeword').lower()
    
    logger.info(f"Creating wake word detector with engine: {engine}")
    
    if engine == 'openwakeword':
        logger.info("Using OpenWakeWord engine")
        return WakeWordDetector(config)
    elif engine == 'porcupine':
        logger.info("Using Picovoice Porcupine engine")
        return PorcupineDetector(config)
    else:
        raise ValueError(f"Unknown wake word engine: {engine}. Supported: openwakeword, porcupine")


# Export the factory function and detector classes
__all__ = ['create_wake_word_detector', 'WakeWordDetector', 'PorcupineDetector']